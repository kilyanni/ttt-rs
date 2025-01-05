use std::{ops::Range, sync::Arc};

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{Tensor, s},
};

use crate::{
    config::ModelConfig,
    inner::{TTTInnerModel, TTTInputsInner},
    util::{MultiHeadLayerNorm, MultiHeadLayerNormConfig},
};

#[derive(Module, Debug)]
pub struct TTTLinear<B: Backend> {
    /// [num_heads, head_dim, head_dim]
    pub weight_init: Param<Tensor<B, 3>>,
    /// [num_heads, head_dim]
    pub bias_init: Param<Tensor<B, 2>>,
    pub layer_norm: MultiHeadLayerNorm<B>,
    pub config: Ignored<ModelConfig>,
}

#[derive(Module, Debug)]
pub struct TTTLinearState<B: Backend> {
    /// [batch_size, num_heads, head_dim, head_dim]
    pub weight: Tensor<B, 4>,
    /// [batch_size, num_heads, head_dim]
    pub bias: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub struct TTTLinearConfig {
    #[config(default = "Initializer::Normal{mean:0.0, std:0.02}")]
    pub initializer: Initializer,
}

impl Default for TTTLinearConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TTTInnerModel<B> for TTTLinear<B> {
    type Config = TTTLinearConfig;
    type State = TTTLinearState<B>;

    fn name() -> &'static str {
        "TTTLinear"
    }

    fn new(config: &ModelConfig, inner_config: &Arc<Self::Config>, device: &B::Device) -> Self {
        let len = config.arch.hidden_size;
        let num_heads = config.arch.num_heads;
        let head_dim = config.head_dim();
        Self {
            weight_init: inner_config.initializer.init_with(
                [num_heads, head_dim, head_dim],
                Some(len),
                Some(len),
                device,
            ),
            bias_init: inner_config.initializer.init_with(
                [num_heads, head_dim],
                Some(len),
                Some(len),
                device,
            ),
            layer_norm: MultiHeadLayerNormConfig::new(num_heads, head_dim)
                .with_initializer(inner_config.initializer.clone())
                .with_epsilon(config.ttt.epsilon)
                .init(device),
            config: Ignored(config.clone()),
        }
    }

    fn get_config(&self) -> &ModelConfig {
        &self.config.0
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        let weight = self.weight_init.val().unsqueeze().repeat_dim(0, batch_size);
        let bias = self.bias_init.val().unsqueeze().repeat_dim(0, batch_size);

        TTTLinearState { weight, bias }
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        let inputs = inputs.slice_seq(range);

        let qkv = inputs.qkv;

        let x1 = qkv.xk.clone();

        let z1 = x1.clone().matmul(state.weight.clone()) + state.bias.clone().unsqueeze_dim(2);

        let reconstruction_target = qkv.xv - qkv.xk;

        let (_ln_out, grad_l_wrt_z1) = self
            .layer_norm
            .forward_and_l2_grad(z1, reconstruction_target);

        let token_eta = inputs.token_eta.unsqueeze_dims::<4>(&[0, 0, -1]); // [1, 1, seq_len, 1]

        let ttt_lr_eta = inputs.ttt_lr_eta.unsqueeze_dim::<4>(2); // [B, H, 1, seq_len]

        let eta_combined = token_eta * ttt_lr_eta;

        let eta_mini_batch = eta_combined.tril(0); // [B, H, seq_len, seq_len]

        let attn_scores = qkv.xq.clone().matmul(x1.clone().transpose());

        let attn1 = attn_scores.tril(0);

        let b1_bar = state.bias.clone().unsqueeze_dim(2)
            - eta_mini_batch.clone().matmul(grad_l_wrt_z1.clone());

        let z1_bar = qkv.xq.clone().matmul(state.weight.clone())
            - (eta_mini_batch.clone() * attn1).matmul(grad_l_wrt_z1.clone())
            + b1_bar;

        let last_eta_row = eta_mini_batch.slice(s![.., .., -1.., ..]);

        let last_eta_col = last_eta_row.transpose(); // [B, H, K, 1]

        state.weight.inplace(|x| {
            x - (last_eta_col.clone() * x1)
                .transpose()
                .matmul(grad_l_wrt_z1.clone())
        });

        state.bias.inplace(|x| {
            x - (last_eta_col * grad_l_wrt_z1)
                .sum_dim(2)
                .squeeze_dim::<3>(2)
        });

        let z1_bar_normalized = self.layer_norm.forward(z1_bar);

        qkv.xq + z1_bar_normalized
    }
}
