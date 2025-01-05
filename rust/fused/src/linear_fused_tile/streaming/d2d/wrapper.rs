//! FusedKernel implementation for the D2D streaming TTT-Linear kernel.
//!
//! This implements the FusedKernel trait for `TttD2dStreamingKernel`, which uses
//! a persistent GPU kernel with a global registry for state management.

use std::{fmt::Debug, ops::Range, sync::Arc};

use burn::{
    module::Ignored,
    tensor::{Tensor, TensorPrimitive},
};
use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};
use tracing::trace;
use ttt_core::{ModelConfig, TTTInnerModel, TTTInputsInner, TTTLinear, TTTLinearState};
use ttt_kernels::kernel::FusedKernel;

use super::host::{
    D2dStreamingConfig, get_or_create_d2d_streaming_state, remove_d2d_streaming_state_by_id,
};
use crate::{
    D2dStreamingKernel, Fused, FusedTttBackend,
    ttt::{TttInputs, TttOutputs},
};

/// Inner handle that cleans up the streaming state on drop.
#[derive(Debug)]
struct StreamHandleInner(u64);

impl Drop for StreamHandleInner {
    fn drop(&mut self) {
        remove_d2d_streaming_state_by_id(self.0);
    }
}

/// Handle that cleans up the streaming state when the last clone is dropped.
#[derive(Debug, Clone)]
pub struct StreamHandle(Arc<StreamHandleInner>);

impl StreamHandle {
    pub fn new(stream_id: u64) -> Self {
        Self(Arc::new(StreamHandleInner(stream_id)))
    }

    pub fn id(&self) -> u64 {
        self.0.0
    }
}

/// State for FusedTileD2dStreaming that wraps TTTLinearState and adds stream_id.
#[derive(burn::module::Module, Debug)]
pub struct FusedTileD2dStreamingState<B: FusedTttBackend> {
    /// The underlying linear state (weight and bias)
    pub inner: TTTLinearState<B>,
    /// Handle that cleans up on drop (not a module parameter)
    pub stream_handle: Ignored<StreamHandle>,
}

impl<B: FusedTttBackend> FusedTileD2dStreamingState<B> {
    pub fn stream_id(&self) -> u64 {
        self.stream_handle.0.id()
    }
}

impl<B: FusedTttBackend> AsRef<TTTLinearState<B>> for FusedTileD2dStreamingState<B> {
    fn as_ref(&self) -> &TTTLinearState<B> {
        &self.inner
    }
}

/// Configuration for the D2D streaming kernel.
/// Extends FusedTttConfig with a stream_id for registry lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct D2dStreamingKernelConfig {
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

impl D2dStreamingKernelConfig {
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

/// Marker type for the D2D streaming TTT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttD2dStreamingKernel;

impl FusedKernel for TttD2dStreamingKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type SavedState<T: Debug + Clone + Send> = TttInputs<T>;
    type Config = D2dStreamingKernelConfig;

    fn forward_launch<R: CubeRuntime + 'static, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: D2dStreamingKernelConfig,
    ) -> (TttOutputs<CubeTensor<R>>, TttInputs<CubeTensor<R>>) {
        let saved = inputs.clone();
        let [batch_size, num_heads, _seq_len, head_dim] = inputs.xq.shape.dims();

        let streaming_config = D2dStreamingConfig::new(
            config.stream_id,
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
        let state = get_or_create_d2d_streaming_state::<R, F>(
            streaming_config,
            client.clone(),
            device.clone(),
            inputs.weight.clone(),
            inputs.bias.clone(),
            inputs.token_eta.clone(),
            inputs.ln_weight.clone(),
            inputs.ln_bias.clone(),
        );

        trace!("D2D streaming forward_d2d start");
        // Use D2D copy to feed inputs to the streaming kernel (no CPU round-trip)
        let output = state.forward_d2d(&inputs.xq, &inputs.xk, &inputs.xv, &inputs.ttt_lr_eta);

        trace!("D2D streaming forward_d2d complete, cloning output");
        // Clone output tensor since we're returning ownership
        let output = output.clone();

        // Return outputs - use result buffers which can be read without blocking
        let result = TttOutputs {
            output,
            weight: state.tensors.result_weight.clone(),
            bias: state.tensors.result_bias.clone(),
        };
        trace!(
            "D2D streaming forward complete, output handle stream: {:?}",
            result.output.handle.stream
        );
        (result, saved)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        _saved: TttInputs<CubeTensor<R>>,
        _grad_outputs: TttOutputs<CubeTensor<R>>,
        _config: D2dStreamingKernelConfig,
    ) -> TttInputs<CubeTensor<R>> {
        panic!("D2D streaming kernel backward not yet implemented")
    }
}

// ============================================================================
// High-level API for streaming kernel
// ============================================================================

// Use shared stream ID counter from parent module
use super::super::next_stream_id;

/// High-level API for the D2D streaming TTT-Linear forward pass.
///
/// This function takes burn Tensors, converts them to CubeTensors,
/// calls the streaming kernel, and returns burn Tensors.
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_d2d_streaming_forward<B: FusedTttBackend>(
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
        D2dStreamingKernelConfig::new(stream_id, mini_batch_len, head_dim, epsilon, threads);

    let (outputs, _saved) =
        <B as FusedKernelBackend<TttD2dStreamingKernel>>::forward(inputs, config);

    (
        Tensor::from_primitive(TensorPrimitive::Float(outputs.output)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.weight)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.bias)),
    )
}

// ============================================================================
// TTTInnerModel implementation for streaming kernel
// ============================================================================

/// TTTInnerModel implementation for the D2D streaming fused kernel.
///
/// The streaming kernel maintains a persistent GPU kernel that processes
/// mini-batches incrementally, keeping weight/bias in shared memory between calls.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, D2dStreamingKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = FusedTileD2dStreamingState<B>;

    fn name() -> &'static str {
        "FusedD2dStreamingTTTLinear"
    }

    fn new(general_config: &ModelConfig, config: &Arc<Self::Config>, device: &B::Device) -> Self {
        Fused::new(TTTLinear::new(general_config, config, device))
    }

    fn get_config(&self) -> &ModelConfig {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        FusedTileD2dStreamingState {
            inner: self.inner.init_state(batch_size),
            stream_handle: Ignored(StreamHandle::new(next_stream_id())),
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

        let (output, weight_updated, bias_updated) = fused_ttt_d2d_streaming_forward(
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
