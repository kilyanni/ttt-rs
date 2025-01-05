//! Launch functions for the tiled TTT-Linear forward kernel.

use burn_cubecl::{CubeRuntime, FloatElement, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::prelude::*;
use thundercube::prelude::{D4, D8, D16, D32, D64, LINE_SIZE};
use ttt_kernels::{TensorBundle, util::empty_like};

use super::{
    super::helpers::Params, InputsLaunch, OutputsLaunch, fused_ttt_forward_kernel,
    fused_ttt_forward_kernel_multi,
};
use crate::{
    FusedTttConfig,
    ttt::{TttInputs, TttOutputs},
};

/// Launch the tiled TTT forward kernel with automatic tile size dispatch.
///
/// Supports multiple tile configurations based on (seq_len, head_dim):
/// - 8x32, 8x64, 16x32, 16x64, 16x128, 32x32, 32x64
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_forward<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    weight_out: TensorHandleRef<R>,
    bias_out: TensorHandleRef<R>,
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

    let inputs_launch = InputsLaunch::<F, R>::new(
        xq.as_tensor_arg(vectorization),
        xk.as_tensor_arg(vectorization),
        xv.as_tensor_arg(vectorization),
        weight.as_tensor_arg(vectorization),
        bias.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let outputs_launch = OutputsLaunch::<F, R>::new(
        output.as_tensor_arg(vectorization),
        weight_out.as_tensor_arg(vectorization),
        bias_out.as_tensor_arg(vectorization),
    );

    tile_dispatch!(
        fused_ttt_forward_kernel,
        client,
        cube_count,
        seq_len,
        head_dim,
        config.threads,
        inputs_launch,
        outputs_launch,
        config
    );
}

/// Forward pass using the tiled kernel.
///
/// Supports multiple tile configurations based on (seq_len, head_dim):
/// - 8x32, 8x64, 16x32, 16x64, 16x128, 32x32, 32x64
///
/// Returns the outputs. Forward intermediates are recomputed during backward.
pub fn forward<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    config: FusedTttConfig,
) -> TttOutputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();

    // Allocate output tensors
    let output = empty_like::<R, F>(&inputs.xq, shape.clone());
    let weight_out = empty_like::<R, F>(&inputs.weight, inputs.weight.shape.clone());
    let bias_out = empty_like::<R, F>(&inputs.bias, inputs.bias.shape.clone());

    launch_tile_forward::<R, F>(
        &inputs.xq.client,
        inputs.xq.as_handle_ref(),
        inputs.xk.as_handle_ref(),
        inputs.xv.as_handle_ref(),
        inputs.weight.as_handle_ref(),
        inputs.bias.as_handle_ref(),
        inputs.token_eta.as_handle_ref(),
        inputs.ttt_lr_eta.as_handle_ref(),
        inputs.ln_weight.as_handle_ref(),
        inputs.ln_bias.as_handle_ref(),
        output.as_handle_ref(),
        weight_out.as_handle_ref(),
        bias_out.as_handle_ref(),
        config,
    );

    TttOutputs {
        output,
        weight: weight_out,
        bias: bias_out,
    }
}

/// Launch the multi-stage tiled TTT forward kernel.
///
/// Processes `num_stages` mini-batches in a single kernel launch.
/// Input seq_len should be `mini_batch_len * num_stages`.
#[allow(clippy::too_many_arguments)]
pub fn launch_tile_forward_multi<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    weight_out: TensorHandleRef<R>,
    bias_out: TensorHandleRef<R>,
    weight_checkpoints: TensorHandleRef<R>,
    bias_checkpoints: TensorHandleRef<R>,
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

    let inputs_launch = InputsLaunch::<F, R>::new(
        xq.as_tensor_arg(vectorization),
        xk.as_tensor_arg(vectorization),
        xv.as_tensor_arg(vectorization),
        weight.as_tensor_arg(vectorization),
        bias.as_tensor_arg(vectorization),
        token_eta.as_tensor_arg(vectorization),
        ttt_lr_eta.as_tensor_arg(vectorization),
        ln_weight.as_tensor_arg(vectorization),
        ln_bias.as_tensor_arg(vectorization),
    );

    let outputs_launch = OutputsLaunch::<F, R>::new(
        output.as_tensor_arg(vectorization),
        weight_out.as_tensor_arg(vectorization),
        bias_out.as_tensor_arg(vectorization),
    );

    tile_dispatch!(
        fused_ttt_forward_kernel_multi,
        client,
        cube_count,
        mini_batch_len,
        head_dim,
        config.threads,
        inputs_launch,
        outputs_launch,
        weight_checkpoints.as_tensor_arg(vectorization),
        bias_checkpoints.as_tensor_arg(vectorization),
        ScalarArg::new(num_stages as u32),
        config
    );
}

/// Forward multi return type including per-stage weight/bias checkpoints.
pub struct ForwardMultiResult<R: CubeRuntime> {
    pub outputs: TttOutputs<CubeTensor<R>>,
    pub weight_checkpoints: CubeTensor<R>,
    pub bias_checkpoints: CubeTensor<R>,
}

pub fn forward_multi<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    config: FusedTttConfig,
) -> ForwardMultiResult<R> {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();
    let [batch_size, num_heads, seq_len, head_dim] = shape.dims();
    let mini_batch_len = config.mini_batch_len;

    assert_eq!(
        seq_len % mini_batch_len,
        0,
        "seq_len ({seq_len}) must be divisible by mini_batch_len ({mini_batch_len})"
    );
    let num_stages = seq_len / mini_batch_len;

    // Allocate output tensors
    let output = empty_like::<R, F>(&inputs.xq, shape.clone());
    let weight_out = empty_like::<R, F>(&inputs.weight, inputs.weight.shape.clone());
    let bias_out = empty_like::<R, F>(&inputs.bias, inputs.bias.shape.clone());

    // Allocate checkpoints (one per checkpoint_interval stages)
    let checkpoint_interval = config.checkpoint_interval;
    let num_checkpoints = num_stages.div_ceil(checkpoint_interval);
    let ckpt_count = batch_size * num_heads * num_checkpoints;
    let weight_checkpoints = empty_like::<R, F>(&inputs.xq, [ckpt_count, head_dim, head_dim]);
    let bias_checkpoints = empty_like::<R, F>(&inputs.xq, [ckpt_count, head_dim]);

    launch_tile_forward_multi::<R, F>(
        &inputs.xq.client,
        inputs.xq.as_handle_ref(),
        inputs.xk.as_handle_ref(),
        inputs.xv.as_handle_ref(),
        inputs.weight.as_handle_ref(),
        inputs.bias.as_handle_ref(),
        inputs.token_eta.as_handle_ref(),
        inputs.ttt_lr_eta.as_handle_ref(),
        inputs.ln_weight.as_handle_ref(),
        inputs.ln_bias.as_handle_ref(),
        output.as_handle_ref(),
        weight_out.as_handle_ref(),
        bias_out.as_handle_ref(),
        weight_checkpoints.as_handle_ref(),
        bias_checkpoints.as_handle_ref(),
        config,
        num_stages,
    );

    ForwardMultiResult {
        outputs: TttOutputs {
            output,
            weight: weight_out,
            bias: bias_out,
        },
        weight_checkpoints,
        bias_checkpoints,
    }
}
