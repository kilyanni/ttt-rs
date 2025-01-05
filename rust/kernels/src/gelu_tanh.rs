//! Fused GELU with tanh approximation kernel using thundercube.

use burn_cubecl::CubeRuntime;
use cubecl::prelude::*;
use thundercube::{
    LINE_SIZE,
    unary_ops::{gelu, gelu_bwd, gelu_bwd_bwd},
};

#[cube(launch, launch_unchecked)]
pub fn gelu_tanh_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let idx = ABSOLUTE_POS;

    if idx < input.len() {
        output[idx] = gelu::<F>(input[idx]);
    }
}

/// Computes `gelu_bwd(x)` - the derivative of gelu at x.
/// Used directly in TTT MLP forward pass for gradient computation.
#[cube(launch, launch_unchecked)]
pub fn gelu_bwd_forward_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
    let idx = ABSOLUTE_POS;

    if idx < input.len() {
        output[idx] = gelu_bwd::<F>(input[idx]);
    }
}

#[cube(launch, launch_unchecked)]
pub fn gelu_tanh_backward_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    grad_output: &Tensor<Line<F>>,
    grad_input: &mut Tensor<Line<F>>,
) {
    let idx = ABSOLUTE_POS;

    if idx < input.len() {
        grad_input[idx] = grad_output[idx] * gelu_bwd::<F>(input[idx]);
    }
}

#[cube(launch, launch_unchecked)]
pub fn gelu_tanh_backward_backward_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    grad_output: &Tensor<Line<F>>,
    grad_input: &mut Tensor<Line<F>>,
) {
    let idx = ABSOLUTE_POS;

    if idx < input.len() {
        grad_input[idx] = grad_output[idx] * gelu_bwd_bwd::<F>(input[idx]);
    }
}

pub fn launch_gelu_tanh<R: CubeRuntime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
) {
    let num_elements: usize = input.shape.iter().product();
    let cube_dim = CubeDim::new(client, num_elements);
    let cube_count = (num_elements as u32).div_ceil(cube_dim.num_elems());

    unsafe {
        cube_launch!(gelu_tanh_kernel::<F, R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            cube_dim,
            TensorArg::from_raw_parts::<F>(input.handle, input.strides, input.shape, LINE_SIZE),
            TensorArg::from_raw_parts::<F>(output.handle, output.strides, output.shape, LINE_SIZE),
        ));
    }
}

pub fn launch_gelu_bwd_forward<R: CubeRuntime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
) {
    let num_elements: usize = input.shape.iter().product();
    let cube_dim = CubeDim::new(client, num_elements);
    let cube_count = (num_elements as u32).div_ceil(cube_dim.num_elems());

    unsafe {
        cube_launch!(gelu_bwd_forward_kernel::<F, R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            cube_dim,
            TensorArg::from_raw_parts::<F>(input.handle, input.strides, input.shape, LINE_SIZE),
            TensorArg::from_raw_parts::<F>(output.handle, output.strides, output.shape, LINE_SIZE),
        ));
    }
}

pub fn launch_gelu_tanh_backward<R: CubeRuntime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorHandleRef<R>,
    grad_output: TensorHandleRef<R>,
    grad_input: TensorHandleRef<R>,
) {
    let num_elements: usize = input.shape.iter().product();

    let cube_dim = CubeDim::new(client, num_elements);
    let cube_count = (num_elements as u32).div_ceil(cube_dim.num_elems());

    unsafe {
        cube_launch!(gelu_tanh_backward_kernel::<F, R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            cube_dim,
            TensorArg::from_raw_parts::<F>(input.handle, input.strides, input.shape, LINE_SIZE),
            TensorArg::from_raw_parts::<F>(
                grad_output.handle,
                grad_output.strides,
                grad_output.shape,
                LINE_SIZE,
            ),
            TensorArg::from_raw_parts::<F>(
                grad_input.handle,
                grad_input.strides,
                grad_input.shape,
                LINE_SIZE,
            ),
        ));
    }
}

pub fn launch_gelu_tanh_backward_backward<R: CubeRuntime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorHandleRef<R>,
    grad_output: TensorHandleRef<R>,
    grad_input: TensorHandleRef<R>,
) {
    let num_elements: usize = input.shape.iter().product();

    let cube_dim = CubeDim::new(client, num_elements);
    let cube_count = (num_elements as u32).div_ceil(cube_dim.num_elems());

    unsafe {
        cube_launch!(gelu_tanh_backward_backward_kernel::<F, R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            cube_dim,
            TensorArg::from_raw_parts::<F>(input.handle, input.strides, input.shape, LINE_SIZE),
            TensorArg::from_raw_parts::<F>(
                grad_output.handle,
                grad_output.strides,
                grad_output.shape,
                LINE_SIZE,
            ),
            TensorArg::from_raw_parts::<F>(
                grad_input.handle,
                grad_input.strides,
                grad_input.shape,
                LINE_SIZE,
            ),
        ));
    }
}
