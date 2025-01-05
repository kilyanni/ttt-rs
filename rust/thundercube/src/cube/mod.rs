pub mod broadcast;
mod load;
mod mma;
mod reduce;
mod store;

#[cfg(test)]
mod tests;

use cubecl::cube;

#[cube]
pub fn swizzle(row: usize, vec_col: usize, mask: usize) -> usize {
    vec_col ^ (row & mask)
}

pub use load::*;
pub use mma::*;
pub use reduce::*;
pub use store::*;
