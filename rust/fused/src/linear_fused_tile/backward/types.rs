use cubecl::prelude::*;
use thundercube::prelude::*;

// =============================================================================
// Data structures
// =============================================================================

/// Saved tensors from forward pass needed for backward.
#[derive(CubeType, CubeLaunch)]
pub struct SavedTensors<F: Float> {
    pub xq: Tensor<Line<F>>,
    pub xk: Tensor<Line<F>>,
    pub weight_init: Tensor<Line<F>>,
    pub token_eta: Tensor<Line<F>>,
    pub ttt_lr_eta: Tensor<Line<F>>,
    pub ln_weight: Tensor<Line<F>>,
}

/// Gradient outputs from backward pass.
#[derive(CubeType, CubeLaunch)]
pub struct GradOutputs<F: Float> {
    pub grad_xq: Tensor<Line<F>>,
    pub grad_xk: Tensor<Line<F>>,
    pub grad_xv: Tensor<Line<F>>,
    pub grad_weight: Tensor<Line<F>>,
    pub grad_bias: Tensor<Line<F>>,
    pub grad_ttt_lr_eta: Tensor<Line<F>>,
    /// Atomic tensor for accumulating token_eta gradients across batches/heads.
    /// Shape: [seq_len] (unbatched, shared across batch and head dimensions)
    /// NOTE: Always f32 because HIP/ROCm doesn't support bf16 atomics.
    pub grad_token_eta: Tensor<Atomic<f32>>,
    /// Atomic tensor for accumulating ln_weight gradients across batches.
    /// Shape: [num_heads, head_dim] (unbatched, shared across batch dimension)
    /// NOTE: Always f32 because HIP/ROCm doesn't support bf16 atomics.
    pub grad_ln_weight: Tensor<Atomic<f32>>,
    /// Atomic tensor for accumulating ln_bias gradients across batches.
    /// Shape: [num_heads, head_dim] (unbatched, shared across batch dimension)
    /// NOTE: Always f32 because HIP/ROCm doesn't support bf16 atomics.
    pub grad_ln_bias: Tensor<Atomic<f32>>,
}

/// Additional inputs needed to recompute forward intermediates during backward.
/// These are inputs that weren't already in SavedTensors.
#[derive(CubeType, CubeLaunch)]
pub struct RecomputationInputs<F: Float> {
    pub xv: Tensor<Line<F>>,
    pub bias: Tensor<Line<F>>,
    pub ln_bias: Tensor<Line<F>>,
}

/// Atomically add a register value to an f32 atomic tensor.
/// Values are cast to f32 before the atomic add since HIP/ROCm doesn't support bf16 atomics.
/// Each element in the register is added to the corresponding position in the tensor.
#[cube]
pub fn atomic_add_rv<F: Float, L: Dim>(
    rv: &Rv<F, L>,
    tensor: &mut Tensor<Atomic<f32>>,
    base_offset: usize,
) {
    // Only one thread per cube does the atomic add (all threads have same data)
    if UNIT_POS == 0 {
        #[unroll]
        for line_idx in 0..L::LINES {
            let line = rv.data[line_idx];
            #[unroll]
            for elem_idx in 0..LINE_SIZE {
                let idx = base_offset + line_idx * LINE_SIZE + elem_idx;
                // Cast to f32 for atomic add (HIP doesn't support bf16 atomics)
                let val_f32: f32 = f32::cast_from(line[elem_idx]);
                tensor[idx].fetch_add(val_f32);
            }
        }
    }
}
