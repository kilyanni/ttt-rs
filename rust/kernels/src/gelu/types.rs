use crate::bundle::tensor_bundle;

tensor_bundle! {
    /// Input for GELU forward kernel.
    pub struct GeluInput { input }
}

tensor_bundle! {
    /// Output from GELU forward kernel.
    pub struct GeluOutput { output }
}

/// Marker type for the GELU tanh approximation forward kernel.
#[derive(Debug, Clone, Copy)]
pub struct GeluTanhKernel;

/// Marker type for the GELU backward derivative forward kernel.
/// Computes d/dx gelu(x) directly (used in TTT MLP forward pass).
#[derive(Debug, Clone, Copy)]
pub struct GeluBwdKernel;

/// Marker type for the GELU tanh backward kernel.
#[derive(Debug, Clone, Copy)]
pub struct GeluTanhBackwardKernel;

/// Marker type for the GELU tanh backward-backward kernel.
#[derive(Debug, Clone, Copy)]
pub struct GeluTanhBackwardBackwardKernel;
