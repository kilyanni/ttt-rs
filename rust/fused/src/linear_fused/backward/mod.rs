//! Fused TTT-Linear backward pass kernel.
//!
//! Given dL/d_output (gradient of loss w.r.t. output), computes gradients w.r.t.:
//! - xq, xk, xv (input projections)
//! - W, b (initial weights before TTT update)
//! - ttt_lr_eta (learned learning rate)
//! - ln_weight, ln_bias (layer norm parameters)

mod kernel;
mod kernel_multi;
mod launch;
mod types;

#[allow(unused_imports)]
pub use kernel::*;
#[allow(unused_imports)]
pub use kernel_multi::*;
#[allow(unused_imports)]
pub use launch::*;
#[allow(unused_imports)]
pub use types::*;
