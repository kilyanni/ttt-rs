use burn::tensor::{Tensor, TensorPrimitive};

use super::types::{GeluBwdKernel, GeluInput, GeluTanhKernel};
use crate::kernel::FusedKernelBackend;

/// GELU activation with tanh approximation.
pub fn gelu_tanh<B: FusedKernelBackend<GeluTanhKernel>, const D: usize>(
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    let inputs = GeluInput {
        input: input.into_primitive().tensor(),
    };
    let (outputs, _saved) = B::forward(inputs, ());
    Tensor::from_primitive(TensorPrimitive::Float(outputs.output))
}

/// Computes d/dx gelu(x) directly.
/// Used in TTT MLP forward pass for gradient computation.
pub fn gelu_bwd<B: FusedKernelBackend<GeluBwdKernel>, const D: usize>(
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    let inputs = GeluInput {
        input: input.into_primitive().tensor(),
    };
    let (outputs, _saved) = B::forward(inputs, ());
    Tensor::from_primitive(TensorPrimitive::Float(outputs.output))
}
