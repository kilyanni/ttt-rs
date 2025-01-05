use burn_backend::Element;
use burn_cubecl::{
    CubeRuntime, FloatElement,
    kernel::{cast, into_contiguous},
    ops::numeric::zeros_client,
    tensor::CubeTensor,
};
use cubecl::prelude::*;
use ttt_kernels::{TensorBundle, util::empty_like};

use super::{
    kernel::fused_ttt_backward_kernel,
    kernel_multi::fused_ttt_backward_kernel_multi,
    types::{ForwardInputsLaunch, GradOutputsLaunch},
};
use crate::{
    FusedTttConfig,
    ttt::{TttInputs, TttOutputs},
};

/// Launch the multi-stage fused TTT backward kernel.
pub fn launch_fused_ttt_backward_multi<R: Runtime, F: Float + CubeElement>(
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
    weight_checkpoints: TensorHandleRef<R>,
    bias_checkpoints: TensorHandleRef<R>,
    grad_output: TensorHandleRef<R>,
    grad_xq: TensorHandleRef<R>,
    grad_xk: TensorHandleRef<R>,
    grad_xv: TensorHandleRef<R>,
    grad_weight: TensorHandleRef<R>,
    grad_bias: TensorHandleRef<R>,
    grad_ttt_lr_eta: TensorHandleRef<R>,
    grad_token_eta: TensorHandleRef<R>,
    grad_ln_weight: TensorHandleRef<R>,
    grad_ln_bias: TensorHandleRef<R>,
    num_stages: usize,
    config: FusedTttConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let mini_batch_len = config.mini_batch_len as u32;
    let head_dim = config.head_dim as u32;

    let cube_dim = CubeDim::new_2d(head_dim, mini_batch_len);

    unsafe {
        cube_launch!(fused_ttt_backward_kernel_multi::<F, R>(
            client,
            CubeCount::Static(batch_size, num_heads, 1),
            cube_dim,
            ForwardInputsLaunch::new(
                TensorArg::from_raw_parts::<F>(xq.handle, xq.strides, xq.shape, 1),
                TensorArg::from_raw_parts::<F>(xk.handle, xk.strides, xk.shape, 1),
                TensorArg::from_raw_parts::<F>(xv.handle, xv.strides, xv.shape, 1),
                TensorArg::from_raw_parts::<F>(weight.handle, weight.strides, weight.shape, 1),
                TensorArg::from_raw_parts::<F>(bias.handle, bias.strides, bias.shape, 1),
                TensorArg::from_raw_parts::<F>(
                    token_eta.handle,
                    token_eta.strides,
                    token_eta.shape,
                    1
                ),
                TensorArg::from_raw_parts::<F>(
                    ttt_lr_eta.handle,
                    ttt_lr_eta.strides,
                    ttt_lr_eta.shape,
                    1
                ),
                TensorArg::from_raw_parts::<F>(
                    ln_weight.handle,
                    ln_weight.strides,
                    ln_weight.shape,
                    1
                ),
                TensorArg::from_raw_parts::<F>(ln_bias.handle, ln_bias.strides, ln_bias.shape, 1),
            ),
            TensorArg::from_raw_parts::<F>(
                grad_output.handle,
                grad_output.strides,
                grad_output.shape,
                1
            ),
            GradOutputsLaunch::new(
                TensorArg::from_raw_parts::<F>(grad_xq.handle, grad_xq.strides, grad_xq.shape, 1),
                TensorArg::from_raw_parts::<F>(grad_xk.handle, grad_xk.strides, grad_xk.shape, 1),
                TensorArg::from_raw_parts::<F>(grad_xv.handle, grad_xv.strides, grad_xv.shape, 1),
                TensorArg::from_raw_parts::<F>(
                    grad_weight.handle,
                    grad_weight.strides,
                    grad_weight.shape,
                    1
                ),
                TensorArg::from_raw_parts::<F>(
                    grad_bias.handle,
                    grad_bias.strides,
                    grad_bias.shape,
                    1
                ),
                TensorArg::from_raw_parts::<F>(
                    grad_ttt_lr_eta.handle,
                    grad_ttt_lr_eta.strides,
                    grad_ttt_lr_eta.shape,
                    1
                ),
                TensorArg::from_raw_parts::<f32>(
                    grad_token_eta.handle,
                    grad_token_eta.strides,
                    grad_token_eta.shape,
                    1
                ),
                TensorArg::from_raw_parts::<f32>(
                    grad_ln_weight.handle,
                    grad_ln_weight.strides,
                    grad_ln_weight.shape,
                    1
                ),
                TensorArg::from_raw_parts::<f32>(
                    grad_ln_bias.handle,
                    grad_ln_bias.strides,
                    grad_ln_bias.shape,
                    1
                ),
            ),
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

/// Launch the fused TTT backward kernel.
pub fn launch_fused_ttt_backward<R: Runtime, F: Float + CubeElement>(
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
    grad_output: TensorHandleRef<R>,
    grad_xq: TensorHandleRef<R>,
    grad_xk: TensorHandleRef<R>,
    grad_xv: TensorHandleRef<R>,
    grad_weight: TensorHandleRef<R>,
    grad_bias: TensorHandleRef<R>,
    grad_ttt_lr_eta: TensorHandleRef<R>,
    // Atomic tensors (f32, zero-initialized, unbatched)
    grad_token_eta: TensorHandleRef<R>,
    grad_ln_weight: TensorHandleRef<R>,
    grad_ln_bias: TensorHandleRef<R>,
    config: FusedTttConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let seq_len = config.mini_batch_len as u32;
    let head_dim = config.head_dim as u32;

    let cube_dim = CubeDim::new_2d(head_dim, seq_len);

    unsafe {
        cube_launch!(fused_ttt_backward_kernel::<F, R>(
            client,
            CubeCount::Static(batch_size, num_heads, 1),
            cube_dim,
            ForwardInputsLaunch::new(
                TensorArg::from_raw_parts::<F>(xq.handle, xq.strides, xq.shape, 1),
                TensorArg::from_raw_parts::<F>(xk.handle, xk.strides, xk.shape, 1),
                TensorArg::from_raw_parts::<F>(xv.handle, xv.strides, xv.shape, 1),
                TensorArg::from_raw_parts::<F>(weight.handle, weight.strides, weight.shape, 1),
                TensorArg::from_raw_parts::<F>(bias.handle, bias.strides, bias.shape, 1),
                TensorArg::from_raw_parts::<F>(
                    token_eta.handle,
                    token_eta.strides,
                    token_eta.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    ttt_lr_eta.handle,
                    ttt_lr_eta.strides,
                    ttt_lr_eta.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    ln_weight.handle,
                    ln_weight.strides,
                    ln_weight.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(ln_bias.handle, ln_bias.strides, ln_bias.shape, 1),
            ),
            TensorArg::from_raw_parts::<F>(
                grad_output.handle,
                grad_output.strides,
                grad_output.shape,
                1,
            ),
            GradOutputsLaunch::new(
                TensorArg::from_raw_parts::<F>(grad_xq.handle, grad_xq.strides, grad_xq.shape, 1),
                TensorArg::from_raw_parts::<F>(grad_xk.handle, grad_xk.strides, grad_xk.shape, 1),
                TensorArg::from_raw_parts::<F>(grad_xv.handle, grad_xv.strides, grad_xv.shape, 1),
                TensorArg::from_raw_parts::<F>(
                    grad_weight.handle,
                    grad_weight.strides,
                    grad_weight.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    grad_bias.handle,
                    grad_bias.strides,
                    grad_bias.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    grad_ttt_lr_eta.handle,
                    grad_ttt_lr_eta.strides,
                    grad_ttt_lr_eta.shape,
                    1,
                ),
                // Atomic tensors use scalar access (vectorization=1) and f32 type
                TensorArg::from_raw_parts::<f32>(
                    grad_token_eta.handle,
                    grad_token_eta.strides,
                    grad_token_eta.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<f32>(
                    grad_ln_weight.handle,
                    grad_ln_weight.strides,
                    grad_ln_weight.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<f32>(
                    grad_ln_bias.handle,
                    grad_ln_bias.strides,
                    grad_ln_bias.shape,
                    1,
                ),
            ),
            config,
        ));
    }
}

pub fn backward<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    grad_outputs: TttOutputs<CubeTensor<R>>,
    epsilon: f32,
) -> TttInputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);
    let grad_output = into_contiguous(grad_outputs.output);

    let [_batch_size, num_heads, seq_len, head_dim] = inputs.xq.shape.dims();
    let t = &inputs.xq; // template for empty_like

    let grad_xq = empty_like::<R, F>(t, inputs.xq.shape.clone());
    let grad_xk = empty_like::<R, F>(t, inputs.xk.shape.clone());
    let grad_xv = empty_like::<R, F>(t, inputs.xv.shape.clone());
    let grad_weight = empty_like::<R, F>(t, inputs.weight.shape.clone());
    let grad_bias = empty_like::<R, F>(t, inputs.bias.shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(t, inputs.ttt_lr_eta.shape.clone());

    // Atomic gradients are unbatched (accumulation across batches/heads via atomics)
    // Must be zero-initialized since kernel uses atomic adds
    // NOTE: These use f32 because HIP/ROCm doesn't support bf16 atomics
    let grad_token_eta = zeros_client::<R>(
        inputs.xq.client.clone(),
        inputs.xq.device.clone(),
        [seq_len].into(),
        f32::dtype(),
    );
    let grad_ln_weight = zeros_client::<R>(
        inputs.xq.client.clone(),
        inputs.xq.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );
    let grad_ln_bias = zeros_client::<R>(
        inputs.xq.client.clone(),
        inputs.xq.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );

    let config = FusedTttConfig::new(seq_len, head_dim, epsilon, 0); // threads unused for non-tile kernel

    launch_fused_ttt_backward::<R, F>(
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
        grad_output.as_handle_ref(),
        grad_xq.as_handle_ref(),
        grad_xk.as_handle_ref(),
        grad_xv.as_handle_ref(),
        grad_weight.as_handle_ref(),
        grad_bias.as_handle_ref(),
        grad_ttt_lr_eta.as_handle_ref(),
        grad_token_eta.as_handle_ref(),
        grad_ln_weight.as_handle_ref(),
        grad_ln_bias.as_handle_ref(),
        config,
    );

    // Cast atomic f32 tensors back to F
    let grad_token_eta = cast(grad_token_eta, F::dtype());
    let grad_ln_weight = cast(grad_ln_weight, F::dtype());
    let grad_ln_bias = cast(grad_ln_bias, F::dtype());

    TttInputs {
        xq: grad_xq,
        xk: grad_xk,
        xv: grad_xv,
        weight: grad_weight,
        bias: grad_bias,
        token_eta: grad_token_eta,
        ttt_lr_eta: grad_ttt_lr_eta,
        ln_weight: grad_ln_weight,
        ln_bias: grad_ln_bias,
    }
}

pub fn backward_multi<R: CubeRuntime, F: FloatElement>(
    inputs: TttInputs<CubeTensor<R>>,
    weight_checkpoints: CubeTensor<R>,
    bias_checkpoints: CubeTensor<R>,
    grad_outputs: TttOutputs<CubeTensor<R>>,
    config: FusedTttConfig,
) -> TttInputs<CubeTensor<R>> {
    let inputs = inputs.map(into_contiguous);
    let grad_output = into_contiguous(grad_outputs.output);
    let weight_checkpoints = into_contiguous(weight_checkpoints);
    let bias_checkpoints = into_contiguous(bias_checkpoints);

    let [_batch_size, num_heads, seq_len, head_dim] = inputs.xq.shape.dims();
    let mini_batch_len = config.mini_batch_len;
    assert_eq!(
        seq_len % mini_batch_len,
        0,
        "seq_len must be divisible by mini_batch_len"
    );
    let num_stages = seq_len / mini_batch_len;

    let t = &inputs.xq;

    let grad_xq = empty_like::<R, F>(t, inputs.xq.shape.clone());
    let grad_xk = empty_like::<R, F>(t, inputs.xk.shape.clone());
    let grad_xv = empty_like::<R, F>(t, inputs.xv.shape.clone());
    let grad_weight = empty_like::<R, F>(t, inputs.weight.shape.clone());
    let grad_bias = empty_like::<R, F>(t, inputs.bias.shape.clone());
    let grad_ttt_lr_eta = empty_like::<R, F>(t, inputs.ttt_lr_eta.shape.clone());

    // Atomic gradients (f32, zero-initialized)
    let grad_token_eta = zeros_client::<R>(
        inputs.xq.client.clone(),
        inputs.xq.device.clone(),
        [seq_len].into(),
        f32::dtype(),
    );
    let grad_ln_weight = zeros_client::<R>(
        inputs.xq.client.clone(),
        inputs.xq.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );
    let grad_ln_bias = zeros_client::<R>(
        inputs.xq.client.clone(),
        inputs.xq.device.clone(),
        [num_heads, head_dim].into(),
        f32::dtype(),
    );

    launch_fused_ttt_backward_multi::<R, F>(
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
        weight_checkpoints.as_handle_ref(),
        bias_checkpoints.as_handle_ref(),
        grad_output.as_handle_ref(),
        grad_xq.as_handle_ref(),
        grad_xk.as_handle_ref(),
        grad_xv.as_handle_ref(),
        grad_weight.as_handle_ref(),
        grad_bias.as_handle_ref(),
        grad_ttt_lr_eta.as_handle_ref(),
        grad_token_eta.as_handle_ref(),
        grad_ln_weight.as_handle_ref(),
        grad_ln_bias.as_handle_ref(),
        num_stages,
        config,
    );

    let grad_token_eta = cast(grad_token_eta, F::dtype());
    let grad_ln_weight = cast(grad_ln_weight, F::dtype());
    let grad_ln_bias = cast(grad_ln_bias, F::dtype());

    TttInputs {
        xq: grad_xq,
        xk: grad_xk,
        xv: grad_xv,
        weight: grad_weight,
        bias: grad_bias,
        token_eta: grad_token_eta,
        ttt_lr_eta: grad_ttt_lr_eta,
        ln_weight: grad_ln_weight,
        ln_bias: grad_ln_bias,
    }
}
