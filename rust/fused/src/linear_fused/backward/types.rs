use cubecl::prelude::*;

/// Forward pass input tensors grouped into a struct.
#[derive(CubeType, CubeLaunch)]
pub struct ForwardInputs<F: Float> {
    pub xq: Tensor<F>,
    pub xk: Tensor<F>,
    pub xv: Tensor<F>,
    pub weight: Tensor<F>,
    pub bias: Tensor<F>,
    pub token_eta: Tensor<F>,
    pub ttt_lr_eta: Tensor<F>,
    pub ln_weight: Tensor<F>,
    pub ln_bias: Tensor<F>,
}

/// Gradient output tensors grouped into a struct.
#[derive(CubeType, CubeLaunch)]
pub struct GradOutputs<F: Float> {
    pub xq: Tensor<F>,
    pub xk: Tensor<F>,
    pub xv: Tensor<F>,
    pub weight: Tensor<F>,
    pub bias: Tensor<F>,
    pub ttt_lr_eta: Tensor<F>,
    /// Atomic tensor for accumulating token_eta gradients across batches/heads.
    /// Shape: [seq_len] — always f32 because HIP/ROCm doesn't support bf16 atomics.
    pub token_eta: Tensor<Atomic<f32>>,
    /// Atomic tensor for accumulating ln_weight gradients across batches.
    /// Shape: [num_heads, head_dim] — always f32.
    pub ln_weight: Tensor<Atomic<f32>>,
    /// Atomic tensor for accumulating ln_bias gradients across batches.
    /// Shape: [num_heads, head_dim] — always f32.
    pub ln_bias: Tensor<Atomic<f32>>,
}
