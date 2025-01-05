#![warn(clippy::pedantic)]
#![allow(
    clippy::too_many_arguments,
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
    clippy::type_complexity
)]

//! TTT Layer - Outer layer implementations
//!
//! This crate provides:
//! - `TTT` - the outer TTT layer that wraps inner model implementations
//! - `TTTBlock`, `TTTBlockWithInner` - block-level wrappers
//! - `TTTModel` - full language model implementation
//! - `AnyInner` / `AnyInnerState` - enum-based inner model polymorphism
//! - `dispatch_ttt_layer_type!` - macro for dispatching based on layer type

pub mod any_inner;
pub mod block;
pub mod lm;
pub mod ttt;

// Re-export commonly used items
pub use any_inner::{AnyInner, AnyInnerState};
pub use block::{TTTBlock, TTTBlockConfig, TTTBlockWithInner};
pub use lm::{ModelConfigModelExt, TTTModel};
pub use ttt::{ModelConfigExt, TTT};
// Re-export InnerModel for use with the dispatch macro
pub use ttt_config::InnerModel;

/// Dispatch a function call based on TTT layer type.
///
/// This macro takes a function call with a layer type and dispatches to the
/// appropriate inner model implementation. Useful for benchmarks and tests that
/// need monomorphized code paths for specific inner model types.
///
/// For the main model, prefer `AnyInner` which supports per-layer type mixing.
///
/// # Example
/// ```ignore
/// dispatch_ttt_layer_type!(bench_inner::<B, layer_type, _>(
///     device,
///     config,
/// ));
/// ```
#[macro_export]
macro_rules! dispatch_ttt_layer_type {
    ($f:ident :: < $backend:ident, $ty:tt $(, $other:ty)* > ($($args:expr),* $(,)?)) => {
        match $ty {
            $crate::InnerModel::Linear => {
                $f::<$backend, ttt_core::TTTLinear<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::LinearAdam => {
                $f::<$backend, ttt_core::TTTLinearAdam<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::Mlp => {
                $f::<$backend, ttt_core::TTTMLP<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::Mlp2 => {
                $f::<$backend, ttt_core::TTTMLP2<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::Mlp3 => {
                $f::<$backend, ttt_core::TTTMLP3<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::Mlp4 => {
                $f::<$backend, ttt_core::TTTMLP4<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::FusedNaiveLinear => {
                $f::<$backend, ttt_fused::FusedNaive<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::FusedNaiveMultiLinear => {
                $f::<$backend, ttt_fused::FusedNaiveMulti<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::FusedTileLinear => {
                $f::<$backend, ttt_fused::FusedTile<$backend> $(, $other)*>($($args),*)
            }
            $crate::InnerModel::FusedTileMultiLinear => {
                $f::<$backend, ttt_fused::FusedTileMulti<$backend> $(, $other)*>($($args),*)
            }
            #[cfg(feature = "streaming")]
            $crate::InnerModel::D2dStreamingLinear => {
                $f::<$backend, ttt_fused::FusedTileD2dStreaming<$backend> $(, $other)*>($($args),*)
            }
            #[cfg(feature = "streaming")]
            $crate::InnerModel::PtrStreamingLinear => {
                $f::<$backend, ttt_fused::FusedTilePtrStreaming<$backend> $(, $other)*>($($args),*)
            }
            #[cfg(not(feature = "streaming"))]
            $crate::InnerModel::D2dStreamingLinear | $crate::InnerModel::PtrStreamingLinear => {
                panic!("Streaming kernels require the 'streaming' feature to be enabled")
            }
        }
    };
}
