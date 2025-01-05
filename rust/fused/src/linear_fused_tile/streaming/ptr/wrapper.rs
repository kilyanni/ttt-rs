//! FusedKernel implementation for the pointer-based streaming TTT-Linear kernel.
//!
//! This implements the FusedKernel trait for `TttPtrStreamingKernel`, which uses
//! a persistent GPU kernel with true zero-copy input via pointer tables.

use std::{fmt::Debug, ops::Range, sync::Arc};

use burn::{
    module::Ignored,
    tensor::{Tensor, TensorPrimitive},
};
use burn_cubecl::{CubeRuntime, FloatElement, kernel::into_contiguous, tensor::CubeTensor};
use tracing::trace;
use ttt_core::{ModelConfig, TTTInnerModel, TTTInputsInner, TTTLinear, TTTLinearState};
use ttt_kernels::kernel::FusedKernel;

use super::host::{
    PtrStreamingConfig, get_or_create_ptr_streaming_state, remove_ptr_streaming_state_by_id,
};
use crate::{
    Fused, FusedTttBackend, PtrStreamingKernel,
    ttt::{TttInputs, TttOutputs},
};

/// Inner handle that cleans up the streaming state on drop.
#[derive(Debug)]
struct PtrStreamHandleInner(u64);

impl Drop for PtrStreamHandleInner {
    fn drop(&mut self) {
        remove_ptr_streaming_state_by_id(self.0);
    }
}

/// Handle that cleans up the streaming state when the last clone is dropped.
#[derive(Debug, Clone)]
pub struct PtrStreamHandle(Arc<PtrStreamHandleInner>);

impl PtrStreamHandle {
    pub fn new(stream_id: u64) -> Self {
        Self(Arc::new(PtrStreamHandleInner(stream_id)))
    }

    pub fn id(&self) -> u64 {
        self.0.0
    }
}

/// State for FusedTilePtrStreaming that wraps TTTLinearState and adds stream_id.
#[derive(burn::module::Module, Debug)]
pub struct FusedTilePtrStreamingState<B: FusedTttBackend> {
    /// The underlying linear state (weight and bias)
    pub inner: TTTLinearState<B>,
    /// Handle that cleans up on drop (not a module parameter)
    pub stream_handle: Ignored<PtrStreamHandle>,
}

impl<B: FusedTttBackend> FusedTilePtrStreamingState<B> {
    pub fn stream_id(&self) -> u64 {
        self.stream_handle.0.id()
    }
}

impl<B: FusedTttBackend> AsRef<TTTLinearState<B>> for FusedTilePtrStreamingState<B> {
    fn as_ref(&self) -> &TTTLinearState<B> {
        &self.inner
    }
}

/// Configuration for the ptr streaming kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtrStreamingKernelConfig {
    /// Unique stream identifier for registry lookup
    pub stream_id: u64,
    /// Mini-batch sequence length (CS)
    pub mini_batch_len: usize,
    /// Head dimension (F)
    pub head_dim: usize,
    /// Layer norm epsilon, stored as scaled integer
    pub epsilon_scaled: u32,
    /// Number of threads per cube
    pub threads: usize,
}

impl PtrStreamingKernelConfig {
    pub fn new(
        stream_id: u64,
        mini_batch_len: usize,
        head_dim: usize,
        epsilon: f32,
        threads: usize,
    ) -> Self {
        Self {
            stream_id,
            mini_batch_len,
            head_dim,
            epsilon_scaled: (epsilon / 1e-9) as u32,
            threads,
        }
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon_scaled as f32 * 1e-9
    }
}

/// Marker type for the pointer-based streaming TTT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttPtrStreamingKernel;

impl FusedKernel for TttPtrStreamingKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type SavedState<T: Debug + Clone + Send> = TttInputs<T>;
    type Config = PtrStreamingKernelConfig;

    fn forward_launch<R: CubeRuntime + 'static, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: PtrStreamingKernelConfig,
    ) -> (TttOutputs<CubeTensor<R>>, TttInputs<CubeTensor<R>>) {
        let saved = inputs.clone();
        let [batch_size, num_heads, _seq_len, head_dim] = inputs.xq.shape.dims();

        let ptr_config = PtrStreamingConfig::new(
            batch_size,
            num_heads,
            config.mini_batch_len,
            head_dim,
            config.epsilon(),
            config.threads,
        );

        let client = inputs.xq.client.clone();
        let device = inputs.xq.device.clone();

        // Get or create the streaming state from the global registry
        let state = get_or_create_ptr_streaming_state::<R, F>(
            config.stream_id,
            ptr_config,
            client.clone(),
            device.clone(),
            inputs.weight.clone(),
            inputs.bias.clone(),
            inputs.token_eta.clone(),
            inputs.ln_weight.clone(),
            inputs.ln_bias.clone(),
        );

        trace!("ptr streaming forward start");

        // Make input tensors contiguous if they have non-contiguous strides.
        // Sliced tensors with contiguous strides but an offset are handled by
        // feed_mini_batch which adds handle.offset_start to the GPU addresses.
        let xq = into_contiguous(inputs.xq);
        let xk = into_contiguous(inputs.xk);
        let xv = into_contiguous(inputs.xv);
        let ttt_lr_eta = into_contiguous(inputs.ttt_lr_eta);

        // Sync the default stream to ensure contiguous copies are complete
        // before the persistent kernel (on a different stream) reads from them.
        use thundercube::util::wait_for_sync;
        if let Err(e) = wait_for_sync(&client) {
            trace!("forward_launch sync warning: {:?}", e);
        }

        let output = state.forward_tensor(&xq, &xk, &xv, &ttt_lr_eta);

        trace!("ptr streaming forward complete");
        // Make true copies of output, weight, and bias; the kernel reuses its buffers
        // and burn's tensor tracking can cause issues if we share memory with the kernel.
        // We force copies using mul_scalar by 1.0 which allocates new output buffers.
        use burn_cubecl::ops::numeric::mul_scalar;
        use cubecl::prelude::InputScalar;
        let dtype = output.dtype;
        let output = mul_scalar(output.clone(), InputScalar::new(1.0f32, dtype));
        let weight_out = mul_scalar(
            state.tensors.weight.clone(),
            InputScalar::new(1.0f32, dtype),
        );
        let bias_out = mul_scalar(state.tensors.bias.clone(), InputScalar::new(1.0f32, dtype));

        let outputs = TttOutputs {
            output,
            weight: weight_out,
            bias: bias_out,
        };
        (outputs, saved)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        _saved: TttInputs<CubeTensor<R>>,
        _grad_outputs: TttOutputs<CubeTensor<R>>,
        _config: PtrStreamingKernelConfig,
    ) -> TttInputs<CubeTensor<R>> {
        panic!("Ptr streaming kernel backward not yet implemented")
    }
}

// ============================================================================
// High-level API
// ============================================================================

// Use shared stream ID counter from parent module
use super::super::next_stream_id;

/// High-level API for the ptr streaming TTT-Linear forward pass.
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_ptr_streaming_forward<B: FusedTttBackend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    xv: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Tensor<B, 3>,
    token_eta: Tensor<B, 1>,
    ttt_lr_eta: Tensor<B, 3>,
    ln_weight: Tensor<B, 2>,
    ln_bias: Tensor<B, 2>,
    stream_id: u64,
    mini_batch_len: usize,
    head_dim: usize,
    epsilon: f32,
    threads: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
    use ttt_kernels::FusedKernelBackend;

    let inputs = TttInputs {
        xq: xq.into_primitive().tensor(),
        xk: xk.into_primitive().tensor(),
        xv: xv.into_primitive().tensor(),
        weight: weight.into_primitive().tensor(),
        bias: bias.into_primitive().tensor(),
        token_eta: token_eta.into_primitive().tensor(),
        ttt_lr_eta: ttt_lr_eta.into_primitive().tensor(),
        ln_weight: ln_weight.into_primitive().tensor(),
        ln_bias: ln_bias.into_primitive().tensor(),
    };

    let config =
        PtrStreamingKernelConfig::new(stream_id, mini_batch_len, head_dim, epsilon, threads);

    let (outputs, _saved) =
        <B as FusedKernelBackend<TttPtrStreamingKernel>>::forward(inputs, config);

    (
        Tensor::from_primitive(TensorPrimitive::Float(outputs.output)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.weight)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.bias)),
    )
}

// ============================================================================
// TTTInnerModel implementation for ptr streaming kernel
// ============================================================================

/// TTTInnerModel implementation for the ptr streaming fused kernel.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, PtrStreamingKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = FusedTilePtrStreamingState<B>;

    fn name() -> &'static str {
        "FusedPtrStreamingTTTLinear"
    }

    fn new(general_config: &ModelConfig, config: &Arc<Self::Config>, device: &B::Device) -> Self {
        Fused::new(TTTLinear::new(general_config, config, device))
    }

    fn get_config(&self) -> &ModelConfig {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        FusedTilePtrStreamingState {
            inner: self.inner.init_state(batch_size),
            stream_handle: Ignored(PtrStreamHandle::new(next_stream_id())),
        }
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        let inputs = inputs.slice_seq(range);

        let inner = &self.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let inner_config = inner.get_config();
        let threads = inner_config
            .ttt
            .threads
            .unwrap_or_else(|| super::super::super::api::default_threads(seq_len, head_dim));

        let (output, weight_updated, bias_updated) = fused_ttt_ptr_streaming_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.inner.weight.clone(),
            state.inner.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            state.stream_id(),
            inner_config.ttt.mini_batch_size,
            head_dim,
            epsilon,
            threads,
        );

        state.inner.weight = weight_updated;
        state.inner.bias = bias_updated;

        output
    }
}
