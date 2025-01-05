#![warn(clippy::pedantic)]
#![allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::default_trait_access,
    //
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    //
    clippy::too_many_lines,
    clippy::type_complexity,
)]

//! TTT Core
//!
//! This crate provides:
//! - `TTTConfig` - configuration for TTT layers
//! - `TTTInnerModel` trait - interface for inner TTT model implementations
//! - `Qkv`, `TTTInputsInner` - input data structures
//! - Linear and MLP implementations

pub mod config;
pub mod inner;
pub mod linear;
pub mod linear_adam;
pub mod mlp;
pub mod mlp2;
pub mod mlp3;
pub mod mlp4;
pub mod test_utils;
pub mod util;

pub use config::{GpuAutodiffBackend, GpuBackend, ModelConfig, TEST_VOCAB_SIZE, TrainingBackend};
pub use inner::{Qkv, TTTInnerModel, TTTInputsInner};
pub use linear::{TTTLinear, TTTLinearConfig, TTTLinearState};
pub use linear_adam::{TTTLinearAdam, TTTLinearAdamConfig};
pub use mlp::{TTTMLP, TTTMLPConfig};
pub use mlp2::{TTTMLP2, TTTMLP2Config};
pub use mlp3::{TTTMLP3, TTTMLP3Config};
pub use mlp4::{TTTMLP4, TTTMLP4Config};
pub use ttt_config::{
    InnerModel, ModelArch, ModelSize, PosEncoding, TTTConfig, TrainConfig, TrainParams,
};
