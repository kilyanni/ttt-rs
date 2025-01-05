//! Fully fused TTT-Linear mini-batch forward kernel.
//!
//! This kernel performs the entire TTT-Linear forward pass in a single kernel launch,
//! keeping all intermediates in shared memory to minimize global memory traffic.

mod kernel;
mod kernel_multi;
mod launch;

#[allow(unused_imports)]
pub use kernel::*;
#[allow(unused_imports)]
pub use kernel_multi::*;
#[allow(unused_imports)]
pub use launch::*;
