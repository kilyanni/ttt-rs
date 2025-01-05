use std::marker::PhantomData;

use crate::LINE_SIZE;

/// Marker trait for compile-time dimensions, including D1.
/// CubeCL doesn't like const generics, so we improvise.
///
/// Use `Dim` (which excludes D1) for most cases.
/// Only use `DimOrOne` when D1 is explicitly needed (e.g., vector column dimension).
pub trait DimOrOne: Send + Sync + 'static {
    const VALUE: usize;
    const LINES: usize = Self::VALUE / crate::LINE_SIZE;
}

/// Marker trait for tile dimensions (excludes D1).
/// This is the standard dimension trait - use this for matrix/tile dimensions.
/// D1 is excluded because D1::LINES = 0, which breaks most tile operations.
///
/// For vectors (Rv, Sv), the column dimension uses `DimOrOne` to allow D1.
pub trait Dim: DimOrOne {}

/// Dimension equal to LINE_SIZE. Use when a dimension intentionally matches the line size.
pub type DLine = D4;
const _: () = assert!(DLine::VALUE == LINE_SIZE, "DLine must equal LINE_SIZE");

/// Compile-time dimension of 1 (for vectors only, do not use for tiles).
/// Only implements `DimOrOne`, not `Dim`.
pub struct D1;
impl DimOrOne for D1 {
    const VALUE: usize = 1;
}

/// Compile-time dimension of 4.
pub struct D4;
impl DimOrOne for D4 {
    const VALUE: usize = 4;
}
impl Dim for D4 {}

/// Compile-time dimension of 8.
pub struct D8;
impl DimOrOne for D8 {
    const VALUE: usize = 8;
}
impl Dim for D8 {}

/// Compile-time dimension of 16.
pub struct D16;
impl DimOrOne for D16 {
    const VALUE: usize = 16;
}
impl Dim for D16 {}

/// Compile-time dimension of 32.
pub struct D32;
impl DimOrOne for D32 {
    const VALUE: usize = 32;
}
impl Dim for D32 {}

/// Compile-time dimension of 64.
pub struct D64;
impl DimOrOne for D64 {
    const VALUE: usize = 64;
}
impl Dim for D64 {}

/// Compile-time dimension of 128.
pub struct D128;
impl DimOrOne for D128 {
    const VALUE: usize = 128;
}
impl Dim for D128 {}

/// Compile-time dimension of 256.
pub struct D256;
impl DimOrOne for D256 {
    const VALUE: usize = 256;
}
impl Dim for D256 {}

/// Zero-sized type for carrying dimension info without runtime cost.
pub type DimPhantom<R, C> = PhantomData<(R, C)>;
