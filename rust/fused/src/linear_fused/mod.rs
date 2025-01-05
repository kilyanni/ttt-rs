//! Fused TTT-Linear kernel (scalar/per-element implementation).
//!
//! This module contains the original fused TTT-Linear implementation that operates
//! per-element using global memory with thread cooperation within cubes.

mod api;
mod backward;
mod forward;
mod launch;
mod wrapper;

#[cfg(test)]
mod tests;

pub use api::{fused_ttt_naive_forward, fused_ttt_naive_forward_multi};
pub use backward::{
    backward, backward_multi, launch_fused_ttt_backward, launch_fused_ttt_backward_multi,
};
pub use forward::{
    ForwardMultiResult, forward, forward_multi, fused_ttt_forward_kernel,
    fused_ttt_forward_kernel_multi, launch_fused_ttt_forward, launch_fused_ttt_forward_multi,
};
pub use launch::TttSavedStateMulti;
pub use ttt_kernels::util::empty_like;
