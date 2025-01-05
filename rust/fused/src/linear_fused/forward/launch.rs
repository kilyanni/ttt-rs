use burn_cubecl::{CubeRuntime, FloatElement, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::prelude::*;
use ttt_kernels::{TensorBundle, util::empty_like};

use super::{kernel::fused_ttt_forward_kernel, kernel_multi::fused_ttt_forward_kernel_multi};
use crate::{
    FusedTttConfig,
    ttt::{TttInputs, TttOutputs},
};

/// Launch configuration for the multi-stage fused TTT forward kernel.
pub fn launch_fused_ttt_forward_multi<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    weight_init: TensorHandleRef<R>,
    bias_init: TensorHandleRef<R>,
    weight_out: TensorHandleRef<R>,
    bias_out: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    weight_checkpoints: TensorHandleRef<R>,
    bias_checkpoints: TensorHandleRef<R>,
    num_stages: usize,
    config: FusedTttConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let mini_batch_len = config.mini_batch_len as u32;
    let head_dim = config.head_dim as u32;

    let cube_dim = CubeDim::new_2d(head_dim, mini_batch_len);

    unsafe {
        cube_launch!(fused_ttt_forward_kernel_multi::<F, R>(
            client,
            CubeCount::Static(batch_size, num_heads, 1),
            cube_dim,
            TensorArg::from_raw_parts::<F>(xq.handle, xq.strides, xq.shape, 1),
            TensorArg::from_raw_parts::<F>(xk.handle, xk.strides, xk.shape, 1),
            TensorArg::from_raw_parts::<F>(xv.handle, xv.strides, xv.shape, 1),
            TensorArg::from_raw_parts::<F>(token_eta.handle, token_eta.strides, token_eta.shape, 1),
            TensorArg::from_raw_parts::<F>(
                ttt_lr_eta.handle,
                ttt_lr_eta.strides,
                ttt_lr_eta.shape,
                1,
            ),
            TensorArg::from_raw_parts::<F>(ln_weight.handle, ln_weight.strides, ln_weight.shape, 1),
            TensorArg::from_raw_parts::<F>(ln_bias.handle, ln_bias.strides, ln_bias.shape, 1),
            TensorArg::from_raw_parts::<F>(
                weight_init.handle,
                weight_init.strides,
                weight_init.shape,
                1
            ),
            TensorArg::from_raw_parts::<F>(bias_init.handle, bias_init.strides, bias_init.shape, 1),
            TensorArg::from_raw_parts::<F>(
                weight_out.handle,
                weight_out.strides,
                weight_out.shape,
                1
            ),
            TensorArg::from_raw_parts::<F>(bias_out.handle, bias_out.strides, bias_out.shape, 1),
            TensorArg::from_raw_parts::<F>(output.handle, output.strides, output.shape, 1),
            TensorArg::from_raw_parts::<F>(
                weight_checkpoints.handle,
                weight_checkpoints.strides,
                weight_checkpoints.shape,
                1
            ),
            TensorArg::from_raw_parts::<F>(
                bias_checkpoints.handle,
                bias_checkpoints.strides,
                bias_checkpoints.shape,
                1
            ),
            ScalarArg::new(num_stages as u32),
            config,
        ));
    }
}

/// Launch configuration for the fused TTT forward kernel.
pub fn launch_fused_ttt_forward<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    weight_out: TensorHandleRef<R>,
    bias_out: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    config: FusedTttConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let seq_len = config.mini_batch_len as u32;
    let head_dim = config.head_dim as u32;

    // Each cube handles one (batch, head) pair
    // Threads within cube: head_dim x seq_len (or max allowed)
    let cube_dim = CubeDim::new_2d(head_dim, seq_len);

    unsafe {
        cube_launch!(fused_ttt_forward_kernel::<F, R>(
            client,
            CubeCount::Static(batch_size, num_heads, 1),
            cube_dim,
            TensorArg::from_raw_parts::<F>(xq.handle, xq.strides, xq.shape, 1),
            TensorArg::from_raw_parts::<F>(xk.handle, xk.strides, xk.shape, 1),
            TensorArg::from_raw_parts::<F>(xv.handle, xv.strides, xv.shape, 1),
            TensorArg::from_raw_parts::<F>(token_eta.handle, token_eta.strides, token_eta.shape, 1),
            TensorArg::from_raw_parts::<F>(
                ttt_lr_eta.handle,
                ttt_lr_eta.strides,
                ttt_lr_eta.shape,
                1,
            ),
            TensorArg::from_raw_parts::<F>(ln_weight.handle, ln_weight.strides, ln_weight.shape, 1),
            TensorArg::from_raw_parts::<F>(ln_bias.handle, ln_bias.strides, ln_bias.shape, 1),
            TensorArg::from_raw_parts::<F>(weight.handle, weight.strides, weight.shape, 1),
            TensorArg::from_raw_parts::<F>(bias.handle, bias.strides, bias.shape, 1),
            TensorArg::from_raw_parts::<F>(
                weight_out.handle,
                weight_out.strides,
                weight_out.shape,
                1
            ),
            TensorArg::from_raw_parts::<F>(bias_out.handle, bias_out.strides, bias_out.shape, 1),
            TensorArg::from_raw_parts::<F>(output.handle, output.strides, output.shape, 1),
            config,
        ));
    }
}

pub struct ForwardMultiResult<R: CubeRuntime> {
    pub outputs: TttOutputs<CubeTensor<R>>,
    pub weight_checkpoints: CubeTensor<R>,
    pub bias_checkpoints: CubeTensor<R>,
}

pub fn forward<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    epsilon: f32,
) -> TttOutputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);

    let shape = inputs.xq.shape.clone();
    let [_batch_size, _num_heads, seq_len, head_dim] = shape.dims();

    let output = empty_like::<R, F>(&inputs.xq, shape);
    let weight_out = empty_like::<R, F>(&inputs.weight, inputs.weight.shape.clone());
    let bias_out = empty_like::<R, F>(&inputs.bias, inputs.bias.shape.clone());

    let config = FusedTttConfig::new(seq_len, head_dim, epsilon, 0); // threads unused for non-tile kernel

    launch_fused_ttt_forward::<R, F>(
        &inputs.xq.client,
        inputs.xq.as_handle_ref(),
        inputs.xk.as_handle_ref(),
        inputs.xv.as_handle_ref(),
        inputs.token_eta.as_handle_ref(),
        inputs.ttt_lr_eta.as_handle_ref(),
        inputs.ln_weight.as_handle_ref(),
        inputs.ln_bias.as_handle_ref(),
        inputs.weight.as_handle_ref(),
        inputs.bias.as_handle_ref(),
        weight_out.as_handle_ref(),
        bias_out.as_handle_ref(),
        output.as_handle_ref(),
        config,
    );

    TttOutputs {
        output,
        weight: weight_out,
        bias: bias_out,
    }
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
        "seq_len must be divisible by mini_batch_len"
    );
    let num_stages = seq_len / mini_batch_len;

    let output = empty_like::<R, F>(&inputs.xq, shape);
    let weight_out = empty_like::<R, F>(&inputs.weight, inputs.weight.shape.clone());
    let bias_out = empty_like::<R, F>(&inputs.bias, inputs.bias.shape.clone());

    // Allocate checkpoints
    let checkpoint_interval = config.checkpoint_interval;
    let num_checkpoints = num_stages.div_ceil(checkpoint_interval);
    let ckpt_count = batch_size * num_heads * num_checkpoints;
    let weight_checkpoints = empty_like::<R, F>(&inputs.xq, [ckpt_count, head_dim, head_dim]);
    let bias_checkpoints = empty_like::<R, F>(&inputs.xq, [ckpt_count, head_dim]);

    launch_fused_ttt_forward_multi::<R, F>(
        &inputs.xq.client,
        inputs.xq.as_handle_ref(),
        inputs.xk.as_handle_ref(),
        inputs.xv.as_handle_ref(),
        inputs.token_eta.as_handle_ref(),
        inputs.ttt_lr_eta.as_handle_ref(),
        inputs.ln_weight.as_handle_ref(),
        inputs.ln_bias.as_handle_ref(),
        inputs.weight.as_handle_ref(),
        inputs.bias.as_handle_ref(),
        weight_out.as_handle_ref(),
        bias_out.as_handle_ref(),
        output.as_handle_ref(),
        weight_checkpoints.as_handle_ref(),
        bias_checkpoints.as_handle_ref(),
        num_stages,
        config,
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
