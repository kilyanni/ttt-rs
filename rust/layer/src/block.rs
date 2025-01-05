use burn::{
    module::{Ignored, Module},
    nn::{RmsNorm, RmsNormConfig},
    tensor::Tensor,
};
use ttt_core::{
    config::ModelConfig,
    util::{CausalConv, CausalConvConfig, SwiGluMlp, SwiGluMlpConfig},
};
use ttt_fused::FusedTttBackend;

use crate::{
    any_inner::{AnyInner, AnyInnerState},
    ttt::{ModelConfigExt, TTT},
};

/// A TTT block with its inner model.
#[derive(Module, Debug)]
pub struct TTTBlockWithInner<B: FusedTttBackend> {
    pub block: TTTBlock<B>,
    pub inner: AnyInner<B>,
}

impl<B: FusedTttBackend> TTTBlockWithInner<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: &mut AnyInnerState<B>,
        start_idx: usize,
    ) -> Tensor<B, 3> {
        self.block.forward(input, state, &self.inner, start_idx)
    }

    pub fn init_state(&self, batch_size: usize) -> AnyInnerState<B> {
        self.inner.init_state(batch_size)
    }
}

#[derive(Module, Debug)]
pub struct TTTBlock<B: FusedTttBackend> {
    pub layer_idx: usize,
    pub conv: Option<(CausalConv<B>, RmsNorm<B>)>,
    pub config: Ignored<ModelConfig>,
    pub seq_norm: RmsNorm<B>,
    pub ffn_norm: RmsNorm<B>,
    pub ttt: TTT<B>,
    pub swi_glu_mlp: SwiGluMlp<B>,
}

#[derive(Clone, Debug)]
pub struct TTTBlockConfig {
    pub model_config: ModelConfig,
    pub layer_idx: usize,
}

impl TTTBlockConfig {
    #[must_use]
    pub fn new(model_config: ModelConfig, layer_idx: usize) -> Self {
        Self {
            model_config,
            layer_idx,
        }
    }

    pub fn init<B: FusedTttBackend>(self, device: &B::Device) -> TTTBlock<B> {
        let hidden_size = self.model_config.arch.hidden_size;
        let intermediate_size = self.model_config.arch.intermediate_size;
        let conv_kernel_size = self.model_config.ttt.conv_kernel_size;

        TTTBlock {
            layer_idx: self.layer_idx,
            conv: if self.model_config.ttt.conv_before_ttt {
                Some((
                    CausalConvConfig::new(hidden_size, conv_kernel_size).init(device),
                    RmsNormConfig::new(hidden_size).init(device),
                ))
            } else {
                None
            },
            seq_norm: RmsNormConfig::new(hidden_size).init(device),
            ffn_norm: RmsNormConfig::new(hidden_size).init(device),
            ttt: self.model_config.init_ttt_seq(device),
            swi_glu_mlp: SwiGluMlpConfig::new(hidden_size, intermediate_size).init(device),
            config: Ignored(self.model_config),
        }
    }

    pub fn init_with_inner<B: FusedTttBackend>(
        self,
        inner: AnyInner<B>,
        device: &B::Device,
    ) -> TTTBlockWithInner<B> {
        let block = self.init(device);
        TTTBlockWithInner { block, inner }
    }
}

impl<B: FusedTttBackend> TTTBlock<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        state: &mut AnyInnerState<B>,
        inner: &AnyInner<B>,
        start_idx: usize,
    ) -> Tensor<B, 3> {
        let mut hidden_states = input;

        if let Some((conv, conv_norm)) = &self.conv {
            let residual = hidden_states.clone();
            hidden_states = conv_norm.forward(hidden_states);
            hidden_states = conv.forward(hidden_states);
            hidden_states = residual + hidden_states;
        }

        let residual = hidden_states.clone();
        hidden_states = self.seq_norm.forward(hidden_states);
        hidden_states = self.ttt.forward(hidden_states, inner, state, start_idx);
        hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        hidden_states = self.ffn_norm.forward(hidden_states);
        hidden_states = self.swi_glu_mlp.forward(hidden_states);
        hidden_states = residual + hidden_states;

        hidden_states
    }
}
