//! Launch functions for the tiled TTT-Linear backward kernel.

use std::fmt::Debug;

use burn_backend::Element;
use burn_cubecl::{
    CubeRuntime, FloatElement, kernel::cast, ops::numeric::zeros_client, tensor::CubeTensor,
};
use cubecl::prelude::*;
use thundercube::prelude::{D4, D8, D16, D32, D64, LINE_SIZE};
use ttt_kernels::util::empty_like;

use super::{
    super::helpers::Params,
    kernel::{fused_ttt_backward_kernel, fused_ttt_backward_kernel_multi},
    types::{GradOutputsLaunch, RecomputationInputsLaunch, SavedTensorsLaunch},
};
use crate::FusedTttConfig;

/// Saved tensors needed for backward pass (for recomputation).
#[derive(Debug, Clone)]
pub struct TttSavedTensors<T> {
    pub xq: T,
    pub xk: T,
    pub xv: T,
    pub weight_init: T,
    pub bias: T,
    pub token_eta: T,
    pub ttt_lr_eta: T,
    pub ln_weight: T,
    pub ln_bias: T,
}

/// Gradient inputs for backward pass.
#[derive(Debug, Clone)]
pub struct TttGradInputs<T> {
    pub grad_xq: T,
    pub grad_xk: T,
    pub grad_xv: T,
    pub grad_weight: T,
    pub grad_bias: T,
    pub grad_ttt_lr_eta: T,
    pub grad_token_eta: T,
    pub grad_ln_weight: T,
    pub grad_ln_bias: T,
}

/// Launch the tiled TTT backward kernel.
///
/// Supports same tile configurations as forward.
/// Recomputes forward intermediates from saved inputs instead of loading them.
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_backward<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    // Saved tensors from forward (for backward computation)
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    weight_init: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    // Additional inputs for recomputation
    xv: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    // Upstream gradient
    grad_output: TensorHandleRef<R>,
    // Output gradients
    grad_xq: TensorHandleRef<R>,
    grad_xk: TensorHandleRef<R>,
    grad_xv: TensorHandleRef<R>,
    grad_weight: TensorHandleRef<R>,
    grad_bias: TensorHandleRef<R>,
    grad_ttt_lr_eta: TensorHandleRef<R>,
    grad_token_eta: TensorHandleRef<R>,
    grad_ln_weight: TensorHandleRef<R>,
    grad_ln_bias: TensorHandleRef<R>,
    config: FusedTttConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let seq_len = xq.shape[2];
    let head_dim = xq.shape[3];

    // Each cube handles one (batch, head) pair
    let cube_count = CubeCount::Static(batch_size, num_heads, 1);

    // Vectorization factor for Line<F>
    let vectorization = LINE_SIZE;

    let saved_launch = SavedTensorsLaunch::<F, R>::new(
        xq.as_tensor_arg(vectorization),
        xk.as_tensor_arg(vectorization),
        weight_init.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
    );

    let recompute_launch = RecomputationInputsLaunch::<F, R>::new(
        xv.as_tensor_arg(vectorization),
        bias.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let grad_output_arg = grad_output.as_tensor_arg(vectorization);

    let grads_launch = GradOutputsLaunch::<F, R>::new(
        grad_xq.as_tensor_arg(vectorization),
        grad_xk.as_tensor_arg(vectorization),
        grad_xv.as_tensor_arg(vectorization),
        grad_weight.as_tensor_arg(vectorization),
        grad_bias.as_tensor_arg(vectorization),
        grad_ttt_lr_eta.as_tensor_arg(vectorization),
        // Atomic tensors use scalar (vectorization=1), not Line<F>
        grad_token_eta.as_tensor_arg(1),
        grad_ln_weight.as_tensor_arg(1),
        grad_ln_bias.as_tensor_arg(1),
    );

    tile_dispatch!(
        fused_ttt_backward_kernel,
        client,
        cube_count,
        seq_len,
        head_dim,
        config.threads,
        saved_launch,
        recompute_launch,
        grad_output_arg,
        grads_launch,
        config
    );
}

/// Backward pass using the tiled kernel.
///
/// Recomputes forward intermediates from saved inputs.
/// Takes saved tensors and upstream gradients, returns gradients w.r.t. all inputs.
pub fn backward<R: CubeRuntime, F: FloatElement>(
    saved: TttSavedTensors<CubeTensor<R>>,
    grad_output: CubeTensor<R>,
    epsilon: f32,
    threads: usize,
) -> TttGradInputs<CubeTensor<R>> {
    let shape = saved.xq.shape.clone();
    let [batch_size, num_heads, seq_len, head_dim] = shape.dims();

    // Allocate output gradient tensors (xk, xv have same shape as xq)
    let grad_xq = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xk = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xv = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(&saved.ttt_lr_eta, saved.ttt_lr_eta.shape.clone());

    // Parameter gradients for weight/bias need batch dimension (separate per cube)
    let grad_weight_batched = empty_like::<R, F>(
        &saved.weight_init,
        [batch_size, num_heads, head_dim, head_dim],
    );
    let grad_bias_batched =
        empty_like::<R, F>(&saved.weight_init, [batch_size, num_heads, head_dim]);

    // Atomic gradients are unbatched (accumulation across batches/heads)
    // Must be zero-initialized since kernel uses atomic adds
    // NOTE: These use f32 because HIP/ROCm doesn't support bf16 atomics
    let grad_token_eta = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [seq_len].into(),
        f32::dtype(),
    );
    let grad_ln_weight = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );
    let grad_ln_bias = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );

    let config = FusedTttConfig::new(seq_len, head_dim, epsilon, threads);

    launch_tile_backward::<R, F>(
        &saved.xq.client,
        saved.xq.as_handle_ref(),
        saved.xk.as_handle_ref(),
        saved.weight_init.as_handle_ref(),
        saved.token_eta.as_handle_ref(),
        saved.ttt_lr_eta.as_handle_ref(),
        saved.ln_weight.as_handle_ref(),
        saved.xv.as_handle_ref(),
        saved.bias.as_handle_ref(),
        saved.ln_bias.as_handle_ref(),
        grad_output.as_handle_ref(),
        grad_xq.as_handle_ref(),
        grad_xk.as_handle_ref(),
        grad_xv.as_handle_ref(),
        grad_weight_batched.as_handle_ref(),
        grad_bias_batched.as_handle_ref(),
        grad_ttt_lr_eta.as_handle_ref(),
        grad_token_eta.as_handle_ref(),
        grad_ln_weight.as_handle_ref(),
        grad_ln_bias.as_handle_ref(),
        config,
    );

    // These are in f32 (see above), so we need to cast them back
    let grad_token_eta = cast(grad_token_eta, F::dtype());
    let grad_ln_weight = cast(grad_ln_weight, F::dtype());
    let grad_ln_bias = cast(grad_ln_bias, F::dtype());

    TttGradInputs {
        grad_xq,
        grad_xk,
        grad_xv,
        grad_weight: grad_weight_batched,
        grad_bias: grad_bias_batched,
        grad_ttt_lr_eta,
        grad_token_eta,
        grad_ln_weight,
        grad_ln_bias,
    }
}

/// Launch the multi-stage tiled TTT backward kernel.
///
/// Processes `num_stages` mini-batches in reverse order (backward through time).
/// Recomputes forward intermediates from saved inputs.
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_backward_multi<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    // Saved tensors from forward
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    weight_init: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    // Additional inputs for recomputation
    xv: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    // Per-stage weight/bias checkpoints from forward
    weight_checkpoints: TensorHandleRef<R>,
    bias_checkpoints: TensorHandleRef<R>,
    // Per-(batch,head) scratch for storing reconstructed W[stage_idx]
    weight_stage_buf: TensorHandleRef<R>,
    // Upstream gradient
    grad_output: TensorHandleRef<R>,
    // Output gradients
    grad_xq: TensorHandleRef<R>,
    grad_xk: TensorHandleRef<R>,
    grad_xv: TensorHandleRef<R>,
    grad_weight: TensorHandleRef<R>,
    grad_bias: TensorHandleRef<R>,
    grad_ttt_lr_eta: TensorHandleRef<R>,
    grad_token_eta: TensorHandleRef<R>,
    grad_ln_weight: TensorHandleRef<R>,
    grad_ln_bias: TensorHandleRef<R>,
    config: FusedTttConfig,
    num_stages: usize,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let mini_batch_len = config.mini_batch_len;
    let head_dim = config.head_dim;

    // Each cube handles one (batch, head) pair
    let cube_count = CubeCount::Static(batch_size, num_heads, 1);

    // Vectorization factor for Line<F>
    let vectorization = LINE_SIZE;

    let saved_launch = SavedTensorsLaunch::<F, R>::new(
        xq.as_tensor_arg(vectorization),
        xk.as_tensor_arg(vectorization),
        weight_init.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
    );

    let recompute_launch = RecomputationInputsLaunch::<F, R>::new(
        xv.as_tensor_arg(vectorization),
        bias.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let grad_output_arg = grad_output.as_tensor_arg(vectorization);

    let grads_launch = GradOutputsLaunch::<F, R>::new(
        grad_xq.as_tensor_arg(vectorization),
        grad_xk.as_tensor_arg(vectorization),
        grad_xv.as_tensor_arg(vectorization),
        grad_weight.as_tensor_arg(vectorization),
        grad_bias.as_tensor_arg(vectorization),
        grad_ttt_lr_eta.as_tensor_arg(vectorization),
        // Atomic tensors use scalar (vectorization=1), not Line<F>
        grad_token_eta.as_tensor_arg(1),
        grad_ln_weight.as_tensor_arg(1),
        grad_ln_bias.as_tensor_arg(1),
    );

    tile_dispatch!(
        fused_ttt_backward_kernel_multi,
        client,
        cube_count,
        mini_batch_len,
        head_dim,
        config.threads,
        saved_launch,
        recompute_launch,
        weight_checkpoints.as_tensor_arg(vectorization),
        bias_checkpoints.as_tensor_arg(vectorization),
        weight_stage_buf.as_tensor_arg(vectorization),
        grad_output_arg,
        grads_launch,
        ScalarArg::new(num_stages as u32),
        config
    );
}

/// Backward pass using the multi-stage tiled kernel.
///
/// Processes the full sequence in reverse order by dividing it into
/// mini-batches and processing them backward through time.
/// Recomputes forward intermediates from saved inputs.
pub fn backward_multi<R: CubeRuntime, F: FloatElement>(
    saved: TttSavedTensors<CubeTensor<R>>,
    weight_checkpoints: CubeTensor<R>,
    bias_checkpoints: CubeTensor<R>,
    grad_output: CubeTensor<R>,
    config: FusedTttConfig,
) -> TttGradInputs<CubeTensor<R>> {
    let shape = saved.xq.shape.clone();
    let [batch_size, num_heads, seq_len, head_dim] = shape.dims();
    let mini_batch_len = config.mini_batch_len;

    assert_eq!(
        seq_len % mini_batch_len,
        0,
        "seq_len ({seq_len}) must be divisible by mini_batch_len ({mini_batch_len})"
    );
    let num_stages = seq_len / mini_batch_len;

    // Allocate output gradient tensors (xk, xv have same shape as xq)
    let grad_xq = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xk = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_xv = empty_like::<R, F>(&saved.xq, shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(&saved.ttt_lr_eta, saved.ttt_lr_eta.shape.clone());

    // Parameter gradients for weight/bias need batch dimension (separate per cube)
    let grad_weight_batched = empty_like::<R, F>(
        &saved.weight_init,
        [batch_size, num_heads, head_dim, head_dim],
    );
    let grad_bias_batched =
        empty_like::<R, F>(&saved.weight_init, [batch_size, num_heads, head_dim]);

    // Atomic gradients are unbatched (accumulation across batches/heads)
    // Must be zero-initialized since kernel uses atomic adds
    // NOTE: These use f32 because HIP/ROCm doesn't support bf16 atomics
    // token_eta is [seq_len]: each stage writes to its own offset for per-position gradients
    let grad_token_eta = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [seq_len].into(),
        f32::dtype(),
    );
    let grad_ln_weight = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );
    let grad_ln_bias = zeros_client::<R>(
        saved.ln_weight.client.clone(),
        saved.ln_weight.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );

    // Scratch buffer for storing reconstructed W[stage_idx] before backward_stage
    // overwrites it. Reused each stage (only one (batch,head) pair per cube).
    let weight_stage_buf = empty_like::<R, F>(
        &saved.weight_init,
        [batch_size, num_heads, head_dim, head_dim],
    );

    launch_tile_backward_multi::<R, F>(
        &saved.xq.client,
        saved.xq.as_handle_ref(),
        saved.xk.as_handle_ref(),
        saved.weight_init.as_handle_ref(),
        saved.token_eta.as_handle_ref(),
        saved.ttt_lr_eta.as_handle_ref(),
        saved.ln_weight.as_handle_ref(),
        saved.xv.as_handle_ref(),
        saved.bias.as_handle_ref(),
        saved.ln_bias.as_handle_ref(),
        weight_checkpoints.as_handle_ref(),
        bias_checkpoints.as_handle_ref(),
        weight_stage_buf.as_handle_ref(),
        grad_output.as_handle_ref(),
        grad_xq.as_handle_ref(),
        grad_xk.as_handle_ref(),
        grad_xv.as_handle_ref(),
        grad_weight_batched.as_handle_ref(),
        grad_bias_batched.as_handle_ref(),
        grad_ttt_lr_eta.as_handle_ref(),
        grad_token_eta.as_handle_ref(),
        grad_ln_weight.as_handle_ref(),
        grad_ln_bias.as_handle_ref(),
        config,
        num_stages,
    );

    // These are in f32 (see above), so we need to cast them back
    let grad_token_eta = cast(grad_token_eta, F::dtype());
    let grad_ln_weight = cast(grad_ln_weight, F::dtype());
    let grad_ln_bias = cast(grad_ln_bias, F::dtype());

    TttGradInputs {
        grad_xq,
        grad_xk,
        grad_xv,
        grad_weight: grad_weight_batched,
        grad_bias: grad_bias_batched,
        grad_ttt_lr_eta,
        grad_token_eta,
        grad_ln_weight,
        grad_ln_bias,
    }
}
