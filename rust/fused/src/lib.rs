#![warn(clippy::pedantic)]
#![allow(clippy::too_many_arguments)]
#![allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "erroneous false positives on #[cube] functions"
)]
#![allow(
    clippy::used_underscore_binding,
    clippy::pub_underscore_fields,
    reason = "False positive on Module derive"
)]
#![allow(non_camel_case_types, non_snake_case)]
#![allow(
    clippy::similar_names,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::default_trait_access,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::format_push_string
)]
//! TTT Fused Kernels
//!
//! This crate provides fused GPU kernel implementations for TTT:
//! - `TttNaiveKernel` - basic naive fused TTT kernel
//! - `TttTileKernel` - tiled TTT kernel (single-stage)
//! - `TttTileMultiKernel` - multi-stage tiled kernel
//! - Streaming kernels (ROCm only)

// Compile-time check: streaming requires rocm
#[cfg(all(feature = "streaming", not(feature = "rocm")))]
compile_error!(
    "The 'streaming' feature requires the 'rocm' feature to be enabled. Streaming kernels are ROCm-only."
);

use std::marker::PhantomData;

use burn::prelude::*;
use ttt_core::TTTInnerModel;
use ttt_kernels::{FusedKernelBackend, GeluBwdKernel, GeluTanhKernel};

/// Launch a CubeCL kernel with bounds checking in debug builds,
/// unchecked in release builds for performance. Must be called inside `unsafe`.
macro_rules! cube_launch {
    ($kernel:ident :: < $($ty:ty),+ > ( $($args:expr),* $(,)? )) => {{
        #[cfg(debug_assertions)]
        { $kernel::launch::< $($ty),+ >( $($args),* ).unwrap() }
        #[cfg(not(debug_assertions))]
        { $kernel::launch_unchecked::< $($ty),+ >( $($args),* ).unwrap() }
    }};
}

pub mod linear_fused;
pub mod linear_fused_tile;
pub mod ttt;

// Re-export commonly used items from ttt module
pub use linear_fused::{fused_ttt_naive_forward, fused_ttt_naive_forward_multi};
#[cfg(all(feature = "rocm", feature = "streaming"))]
pub use linear_fused_tile::{TttD2dStreamingKernel, TttPtrStreamingKernel};
// Re-export kernel types from linear_fused_tile
pub use linear_fused_tile::{TttTileKernel, TttTileMultiKernel};
pub use ttt::{TttInputs, TttNaiveKernel, TttNaiveMultiKernel, TttOutputs};

// ============================================================================
// Kernel marker types
// ============================================================================

/// Marker for the basic fused TTT-Linear kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveKernel;

/// Marker for the multi-stage fused TTT-Linear kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct NaiveMultiKernel;

/// Marker for the tiled fused TTT-Linear kernel (single-stage).
#[derive(Debug, Clone, Copy, Default)]
pub struct TileKernel;

/// Marker for the multi-stage tiled fused TTT-Linear kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct TileMultiKernel;

/// Marker for the D2D streaming tiled fused TTT-Linear kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct D2dStreamingKernel;

/// Marker for the pointer-based streaming TTT-Linear kernel.
#[derive(Debug, Clone, Copy, Default)]
pub struct PtrStreamingKernel;

// ============================================================================
// Type aliases
// ============================================================================

/// Basic naive fused TTT-Linear kernel.
pub type FusedNaive<B> = Fused<B, ttt_core::TTTLinear<B>, NaiveKernel>;

/// Multi-stage naive fused TTT-Linear kernel.
pub type FusedNaiveMulti<B> = Fused<B, ttt_core::TTTLinear<B>, NaiveMultiKernel>;

/// Tiled fused TTT-Linear kernel (single-stage).
pub type FusedTile<B> = Fused<B, ttt_core::TTTLinear<B>, TileKernel>;

/// Multi-stage tiled fused TTT-Linear kernel.
pub type FusedTileMulti<B> = Fused<B, ttt_core::TTTLinear<B>, TileMultiKernel>;

/// D2D streaming tiled fused TTT-Linear kernel.
#[cfg(all(feature = "rocm", feature = "streaming"))]
pub type FusedTileD2dStreaming<B> = Fused<B, ttt_core::TTTLinear<B>, D2dStreamingKernel>;

/// Pointer-based streaming TTT-Linear kernel.
#[cfg(all(feature = "rocm", feature = "streaming"))]
pub type FusedTilePtrStreaming<B> = Fused<B, ttt_core::TTTLinear<B>, PtrStreamingKernel>;

// ============================================================================
// FusedTttBackend trait
// ============================================================================

/// Unified backend trait for fused TTT kernels.
#[cfg(all(feature = "rocm", feature = "streaming"))]
pub trait FusedTttBackend:
    FusedKernelBackend<TttNaiveKernel>
    + FusedKernelBackend<TttNaiveMultiKernel>
    + FusedKernelBackend<TttTileKernel>
    + FusedKernelBackend<TttTileMultiKernel>
    + FusedKernelBackend<TttD2dStreamingKernel>
    + FusedKernelBackend<TttPtrStreamingKernel>
    + FusedKernelBackend<GeluTanhKernel>
    + FusedKernelBackend<GeluBwdKernel>
{
}

#[cfg(not(all(feature = "rocm", feature = "streaming")))]
pub trait FusedTttBackend:
    FusedKernelBackend<TttNaiveKernel>
    + FusedKernelBackend<TttNaiveMultiKernel>
    + FusedKernelBackend<TttTileKernel>
    + FusedKernelBackend<TttTileMultiKernel>
    + FusedKernelBackend<GeluTanhKernel>
    + FusedKernelBackend<GeluBwdKernel>
{
}

#[cfg(all(feature = "rocm", feature = "streaming"))]
impl<B> FusedTttBackend for B where
    B: Backend
        + FusedKernelBackend<TttNaiveKernel>
        + FusedKernelBackend<TttNaiveMultiKernel>
        + FusedKernelBackend<TttTileKernel>
        + FusedKernelBackend<TttTileMultiKernel>
        + FusedKernelBackend<TttD2dStreamingKernel>
        + FusedKernelBackend<TttPtrStreamingKernel>
        + FusedKernelBackend<GeluTanhKernel>
        + FusedKernelBackend<GeluBwdKernel>
{
}

#[cfg(not(all(feature = "rocm", feature = "streaming")))]
impl<B> FusedTttBackend for B where
    B: Backend
        + FusedKernelBackend<TttNaiveKernel>
        + FusedKernelBackend<TttNaiveMultiKernel>
        + FusedKernelBackend<TttTileKernel>
        + FusedKernelBackend<TttTileMultiKernel>
        + FusedKernelBackend<GeluTanhKernel>
        + FusedKernelBackend<GeluBwdKernel>
{
}

// ============================================================================
// Fused wrapper struct
// ============================================================================

/// Wrapper for fused TTT layers.
#[derive(Debug)]
pub struct Fused<B: Backend, Inner, Kernel> {
    pub inner: Inner,
    _phantom: PhantomData<(B, Kernel)>,
}

impl<B: Backend, Inner: Clone, Kernel> Clone for Fused<B, Inner, Kernel> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, Inner: burn::module::Module<B>, Kernel: Send + Sync + std::fmt::Debug>
    burn::module::Module<B> for Fused<B, Inner, Kernel>
{
    type Record = Inner::Record;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        self.inner.collect_devices(devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self::new(self.inner.fork(device))
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self::new(self.inner.to_device(device))
    }

    fn visit<V: burn::module::ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.inner.visit(visitor);
    }

    fn map<M: burn::module::ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        Self::new(self.inner.map(mapper))
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self::new(self.inner.load_record(record))
    }

    fn into_record(self) -> Self::Record {
        self.inner.into_record()
    }
}

impl<B: Backend, Inner: burn::module::ModuleDisplay, Kernel> burn::module::ModuleDisplay
    for Fused<B, Inner, Kernel>
{
    fn custom_settings(&self) -> Option<burn::module::DisplaySettings> {
        self.inner.custom_settings()
    }

    fn custom_content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        self.inner.custom_content(content)
    }
}

impl<B: Backend, Inner: burn::module::ModuleDisplayDefault, Kernel>
    burn::module::ModuleDisplayDefault for Fused<B, Inner, Kernel>
{
    fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        self.inner.content(content)
    }
}

impl<
    B: burn::tensor::backend::AutodiffBackend,
    Inner: burn::module::AutodiffModule<B>,
    Kernel: Send + Sync + std::fmt::Debug + Clone,
> burn::module::AutodiffModule<B> for Fused<B, Inner, Kernel>
{
    type InnerModule = Fused<B::InnerBackend, Inner::InnerModule, Kernel>;

    fn valid(&self) -> Self::InnerModule {
        Fused::new(self.inner.valid())
    }
}

impl<B: Backend, Inner, Kernel> Fused<B, Inner, Kernel> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, T: TTTInnerModel<B>, K> From<T> for Fused<B, T, K> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

// ============================================================================
// FusedTttConfig
// ============================================================================

const EPSILON_SCALE_INV: f32 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FusedTttConfig {
    pub mini_batch_len: usize,
    pub head_dim: usize,
    pub epsilon_scaled: u32,
    pub threads: usize,
    /// Number of stages between checkpoints (1 = every stage, N = only initial).
    /// Higher values reduce memory but increase backward compute (re-simulation).
    pub checkpoint_interval: usize,
}

impl FusedTttConfig {
    #[must_use]
    pub fn new(mini_batch_len: usize, head_dim: usize, epsilon: f32, threads: usize) -> Self {
        Self {
            mini_batch_len,
            head_dim,
            epsilon_scaled: (epsilon / EPSILON_SCALE_INV) as u32,
            threads,
            checkpoint_interval: 1,
        }
    }

    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.epsilon_scaled as f32 * EPSILON_SCALE_INV
    }

    /// Check if this tile configuration is supported by the fused kernel.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        Self::is_config_supported(self.mini_batch_len, self.head_dim, self.threads)
    }

    /// Check if a (mini_batch_len, head_dim, threads) configuration is supported.
    #[must_use]
    pub fn is_config_supported(mini_batch_len: usize, head_dim: usize, threads: usize) -> bool {
        linear_fused_tile::is_tile_config_supported(mini_batch_len, head_dim, threads)
    }

    /// All supported tile configurations as (mini_batch_len, head_dim, threads) tuples.
    pub const SUPPORTED_CONFIGS: &[(usize, usize, usize)] =
        linear_fused_tile::SUPPORTED_TILE_CONFIGS;
}
