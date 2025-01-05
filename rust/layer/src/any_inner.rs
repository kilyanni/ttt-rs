//! Enum wrapper for all inner model types, enabling per-layer type selection.
//!
//! burn's `Module` trait is not object-safe (it has `type Record` and methods
//! returning `Self`), so we can't use `dyn TTTInnerModel`. Instead, we wrap all
//! concrete inner model types in an enum and derive `Module` on it. This lets
//! different layers in the same model use different inner types.

use std::sync::Arc;

use burn::{module::Module, tensor::Tensor};
use ttt_config::InnerModel;
use ttt_core::{
    TTTInnerModel, TTTInputsInner,
    config::ModelConfig,
    linear::{TTTLinear, TTTLinearConfig, TTTLinearState},
    linear_adam::{TTTLinearAdam, TTTLinearAdamConfig, TTTLinearAdamState},
    mlp::{TTTMLP, TTTMLPConfig, TTTMLPState},
    mlp2::{TTTMLP2, TTTMLP2Config, TTTMLP2State},
    mlp3::{TTTMLP3, TTTMLP3Config, TTTMLP3State},
    mlp4::{TTTMLP4, TTTMLP4Config, TTTMLP4State},
};
use ttt_fused::{FusedNaive, FusedNaiveMulti, FusedTile, FusedTileMulti, FusedTttBackend};

/// All possible inner model types. Each layer in a `TTTModel` holds one of these,
/// enabling per-layer type selection (e.g., Linear for even layers, MLP for odd).
#[derive(Module, Debug)]
pub enum AnyInner<B: FusedTttBackend> {
    Linear(TTTLinear<B>),
    LinearAdam(TTTLinearAdam<B>),
    Mlp(TTTMLP<B>),
    Mlp2(TTTMLP2<B>),
    Mlp3(TTTMLP3<B>),
    Mlp4(TTTMLP4<B>),
    FusedNaive(FusedNaive<B>),
    FusedNaiveMulti(FusedNaiveMulti<B>),
    FusedTile(FusedTile<B>),
    FusedTileMulti(FusedTileMulti<B>),
}

/// All possible inner model state types. Paired with `AnyInner` — the variant
/// must match the model variant (e.g., `AnyInner::Mlp` with `AnyInnerState::Mlp`).
///
/// Fused variants (FusedNaive, FusedNaiveMulti, FusedTile, FusedTileMulti) share
/// `TTTLinearState` with the `Linear` variant since they wrap TTTLinear internally.
#[derive(Module, Debug)]
pub enum AnyInnerState<B: FusedTttBackend> {
    Linear(TTTLinearState<B>),
    LinearAdam(TTTLinearAdamState<B>),
    Mlp(TTTMLPState<B>),
    Mlp2(TTTMLP2State<B>),
    Mlp3(TTTMLP3State<B>),
    Mlp4(TTTMLP4State<B>),
}

impl<B: FusedTttBackend> AnyInner<B> {
    /// Create an inner model from an `InnerModel` enum variant.
    pub fn from_type(inner_type: InnerModel, config: &ModelConfig, device: &B::Device) -> Self {
        match inner_type {
            InnerModel::Linear => {
                let cfg = Arc::new(TTTLinearConfig::default());
                Self::Linear(TTTLinear::new(config, &cfg, device))
            }
            InnerModel::LinearAdam => {
                let cfg = Arc::new(
                    TTTLinearAdamConfig::new()
                        .with_lr(config.ttt.adam_lr)
                        .with_beta1(config.ttt.adam_beta1)
                        .with_beta2(config.ttt.adam_beta2),
                );
                Self::LinearAdam(TTTLinearAdam::new(config, &cfg, device))
            }
            InnerModel::Mlp => {
                let cfg = Arc::new(TTTMLPConfig::default());
                Self::Mlp(TTTMLP::new(config, &cfg, device))
            }
            InnerModel::Mlp2 => {
                let cfg = Arc::new(TTTMLP2Config::default());
                Self::Mlp2(TTTMLP2::new(config, &cfg, device))
            }
            InnerModel::Mlp3 => {
                let cfg = Arc::new(TTTMLP3Config::default());
                Self::Mlp3(TTTMLP3::new(config, &cfg, device))
            }
            InnerModel::Mlp4 => {
                let cfg = Arc::new(TTTMLP4Config::default());
                Self::Mlp4(TTTMLP4::new(config, &cfg, device))
            }
            InnerModel::FusedNaiveLinear => {
                let cfg = Arc::new(TTTLinearConfig::default());
                Self::FusedNaive(TTTLinear::new(config, &cfg, device).into())
            }
            InnerModel::FusedNaiveMultiLinear => {
                let cfg = Arc::new(TTTLinearConfig::default());
                Self::FusedNaiveMulti(TTTLinear::new(config, &cfg, device).into())
            }
            InnerModel::FusedTileLinear => {
                let cfg = Arc::new(TTTLinearConfig::default());
                Self::FusedTile(TTTLinear::new(config, &cfg, device).into())
            }
            InnerModel::FusedTileMultiLinear => {
                let cfg = Arc::new(TTTLinearConfig::default());
                Self::FusedTileMulti(TTTLinear::new(config, &cfg, device).into())
            }
            InnerModel::D2dStreamingLinear | InnerModel::PtrStreamingLinear => {
                panic!(
                    "Streaming inner model types ({inner_type}) require the 'streaming' feature \
                     and are not supported in AnyInner"
                )
            }
        }
    }

    pub fn init_state(&self, batch_size: usize) -> AnyInnerState<B> {
        match self {
            Self::Linear(m) => AnyInnerState::Linear(m.init_state(batch_size)),
            Self::LinearAdam(m) => AnyInnerState::LinearAdam(m.init_state(batch_size)),
            Self::Mlp(m) => AnyInnerState::Mlp(m.init_state(batch_size)),
            Self::Mlp2(m) => AnyInnerState::Mlp2(m.init_state(batch_size)),
            Self::Mlp3(m) => AnyInnerState::Mlp3(m.init_state(batch_size)),
            Self::Mlp4(m) => AnyInnerState::Mlp4(m.init_state(batch_size)),
            // Fused variants share TTTLinearState with Linear
            Self::FusedNaive(m) => AnyInnerState::Linear(m.init_state(batch_size)),
            Self::FusedNaiveMulti(m) => AnyInnerState::Linear(m.init_state(batch_size)),
            Self::FusedTile(m) => AnyInnerState::Linear(m.init_state(batch_size)),
            Self::FusedTileMulti(m) => AnyInnerState::Linear(m.init_state(batch_size)),
        }
    }

    pub fn forward(&self, state: &mut AnyInnerState<B>, inputs: TTTInputsInner<B>) -> Tensor<B, 4> {
        match (self, state) {
            (Self::Linear(m), AnyInnerState::Linear(s)) => m.forward(s, inputs),
            (Self::LinearAdam(m), AnyInnerState::LinearAdam(s)) => m.forward(s, inputs),
            (Self::Mlp(m), AnyInnerState::Mlp(s)) => m.forward(s, inputs),
            (Self::Mlp2(m), AnyInnerState::Mlp2(s)) => m.forward(s, inputs),
            (Self::Mlp3(m), AnyInnerState::Mlp3(s)) => m.forward(s, inputs),
            (Self::Mlp4(m), AnyInnerState::Mlp4(s)) => m.forward(s, inputs),
            (Self::FusedNaive(m), AnyInnerState::Linear(s)) => m.forward(s, inputs),
            (Self::FusedNaiveMulti(m), AnyInnerState::Linear(s)) => m.forward(s, inputs),
            (Self::FusedTile(m), AnyInnerState::Linear(s)) => m.forward(s, inputs),
            (Self::FusedTileMulti(m), AnyInnerState::Linear(s)) => m.forward(s, inputs),
            _ => panic!("AnyInner/AnyInnerState variant mismatch"),
        }
    }
}
