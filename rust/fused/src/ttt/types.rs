use ttt_kernels::tensor_bundle;

tensor_bundle! {
    /// Input tensors for the TTT fused kernel.
    pub struct TttInputs {
        xq, xk, xv, weight, bias, token_eta, ttt_lr_eta, ln_weight, ln_bias
    }
}

tensor_bundle! {
    /// Output tensors from the TTT fused kernel.
    pub struct TttOutputs { output, weight, bias }
}

/// Marker type for the TTT naive fused kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttNaiveKernel;

/// Marker type for the multi-stage TTT naive fused kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttNaiveMultiKernel;
