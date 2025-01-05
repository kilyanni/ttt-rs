//! Thundercube - Tiled GPU compute primitives for CubeCL.
//!
//! This crate provides abstractions for writing tiled GPU kernels with CubeCL,
//! focusing on shared memory tiles and register tiles with compile-time dimensions.
//!
//! # Core Abstractions
//!
//! ## Tiles (`tiles` module)
//! - [`St<F, R, C>`] - Shared memory tile, R×C matrix in GPU shared memory
//! - [`Rt<F, R, C>`] - Register tile, R×C matrix in thread-local registers
//! - [`Sv<F, L>`] / [`Rv<F, L>`] - Vector aliases (column dimension = 1)
//! - [`Dim`] trait - Compile-time dimensions (D4, D8, D16, D32, D64, D128)
//!
//! ## Cooperative Operations (`cube` module)
//! - `load_*` - Cooperative loads from global/shared memory to register tiles
//! - `store_*` - Cooperative stores from register tiles to global/shared memory
//! - `mma_*` - Matrix multiply-accumulate (e.g., `mma_AB`, `mma_AtB`)
//! - `sum_*` / `reduce_*` - Parallel reductions across tiles
//!
//! ## Element-wise Operations
//! - `unary_ops` - Per-element operations (neg, abs, sqrt, exp, etc.)
//! - `binary_ops` - Two-operand operations (add, mul, etc.)
//! - `reduction_ops` - Reduction operations (sum, max, etc.)
//!
//! # Memory Model
//!
//! ```text
//! Global Memory ──load_st──► Shared Memory (St) ──load_rt──► Registers (Rt)
//!       ▲                           ▲                              │
//!       └──store_st_direct──────────┴────────store_rt_to_st────────┘
//! ```
//!
//! - **St** (shared memory): Visible to all threads in a block, used for inter-thread communication
//! - **Rt** (registers): Thread-local, used for computation. Each thread holds a fragment.
//!
//! # Example
//!
//! ```ignore
//! // Load A and B tiles from global to shared memory
//! cube::load_st(&a_global, &mut a_smem, offset, 0, 0);
//! cube::load_st(&b_global, &mut b_smem, offset, 0, 0);
//! sync_cube();
//!
//! // Matrix multiply: C += A @ B
//! let mut c_reg = Rt::<f32, D16, D16>::new();
//! cube::mma_AB(&mut c_reg, &a_smem, &b_smem);
//!
//! // Store result back
//! cube::store_rt_to_st(&c_reg, &mut c_smem);
//! ```

// #![warn(clippy::pedantic)]
#![allow(
    clippy::identity_op,
    reason = "For `addr + 0`, it makes some stuff cleaner to read"
)]
#![allow(
    clippy::len_without_is_empty,
    reason = "Empty tiles aren't a thing, so this method would be confusing"
)]
#![allow(clippy::needless_range_loop)]

// #[cfg(all(
//     any(test, feature = "test-utils"),
//     not(any(feature = "cuda", feature = "rocm", feature = "wgpu", feature = "cpu"))
// ))]
// compile_error!(
//     "At least one backend must be enabled for test-utils, please run with `--features cuda/rocm/wgpu/cpu`"
// );

pub mod binary_ops;
pub mod cube;
pub mod reduction_ops;
#[cfg(feature = "rocm")]
pub mod streaming;
pub mod tiles;
pub mod unary_ops;
pub mod util;

#[cfg(any(test, feature = "test-utils"))]
#[macro_use]
pub mod test_utils;

// We could parametrize this
// but right now it's not worth the effort
pub const LINE_SIZE: usize = 4;

/// Maximum loop iteration count to unconditionally unroll (general/outer loops)
pub const UNROLL_LIMIT: usize = 1;
/// Maximum loop iteration count to unconditionally unroll (hot/inner loops)
pub const UNROLL_LIMIT_HOT: usize = 4;

pub mod prelude {
    #[cfg(test)]
    pub use crate::test_kernel;
    #[cfg(feature = "rocm")]
    pub use crate::util::gpu_sleep;
    pub use crate::{LINE_SIZE, UNROLL_LIMIT, UNROLL_LIMIT_HOT, cube, tiles::*, util::sync_planes};
}
