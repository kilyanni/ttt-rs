use std::{ops::Range, sync::Arc};

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{Tensor, s},
};
use ttt_kernels::{FusedKernelBackend, GeluBwdKernel, GeluTanhKernel, gelu_bwd, gelu_tanh};

/// Backend trait that supports GELU kernels.
pub trait GeluBackend:
    Backend + FusedKernelBackend<GeluTanhKernel> + FusedKernelBackend<GeluBwdKernel>
{
}

impl<B> GeluBackend for B where
    B: Backend + FusedKernelBackend<GeluTanhKernel> + FusedKernelBackend<GeluBwdKernel>
{
}

use crate::{
    config::ModelConfig,
    inner::{TTTInnerModel, TTTInputsInner},
    util::{MultiHeadLayerNorm, MultiHeadLayerNormConfig},
};

#[derive(Module, Debug)]
pub struct TTTMLP<B: GeluBackend> {
    /// First layer weight: [num_heads, head_dim, mlp_dim]
    pub w1_init: Param<Tensor<B, 3>>,
    /// First layer bias: [num_heads, mlp_dim]
    pub b1_init: Param<Tensor<B, 2>>,
    /// Second layer weight: [num_heads, mlp_dim, head_dim]
    pub w2_init: Param<Tensor<B, 3>>,
    /// Second layer bias: [num_heads, head_dim]
    pub b2_init: Param<Tensor<B, 2>>,
    pub layer_norm: MultiHeadLayerNorm<B>,
    pub config: Ignored<ModelConfig>,
}

#[derive(Module, Debug)]
pub struct TTTMLPState<B: GeluBackend> {
    /// First layer weight: [batch_size, num_heads, head_dim, mlp_dim]
    pub w1: Tensor<B, 4>,
    /// First layer bias: [batch_size, num_heads, mlp_dim]
    pub b1: Tensor<B, 3>,
    /// Second layer weight: [batch_size, num_heads, mlp_dim, head_dim]
    pub w2: Tensor<B, 4>,
    /// Second layer bias: [batch_size, num_heads, head_dim]
    pub b2: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub struct TTTMLPConfig {
    #[config(default = "Initializer::Normal{mean:0.0, std:0.02}")]
    pub initializer: Initializer,
}

impl Default for TTTMLPConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: GeluBackend> TTTInnerModel<B> for TTTMLP<B> {
    type Config = TTTMLPConfig;
    type State = TTTMLPState<B>;

    fn name() -> &'static str {
        "TTTMLP"
    }

    fn new(config: &ModelConfig, inner_config: &Arc<Self::Config>, device: &B::Device) -> Self {
        let len = config.arch.hidden_size;
        let num_heads = config.arch.num_heads;
        let head_dim = config.head_dim();
        let mlp_dim = config.ttt.mlp_expansion_factor * head_dim;

        Self {
            w1_init: inner_config.initializer.init_with(
                [num_heads, head_dim, mlp_dim],
                Some(len),
                Some(len),
                device,
            ),
            b1_init: inner_config.initializer.init_with(
                [num_heads, mlp_dim],
                Some(len),
                Some(len),
                device,
            ),
            w2_init: inner_config.initializer.init_with(
                [num_heads, mlp_dim, head_dim],
                Some(len),
                Some(len),
                device,
            ),
            b2_init: inner_config.initializer.init_with(
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
        let w1 = self.w1_init.val().unsqueeze().repeat_dim(0, batch_size);
        let b1 = self.b1_init.val().unsqueeze().repeat_dim(0, batch_size);
        let w2 = self.w2_init.val().unsqueeze().repeat_dim(0, batch_size);
        let b2 = self.b2_init.val().unsqueeze().repeat_dim(0, batch_size);

        TTTMLPState { w1, b1, w2, b2 }
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

        let z1 = x1.clone().matmul(state.w1.clone()) + state.b1.clone().unsqueeze_dim(2);

        let x2 = gelu_tanh(z1.clone());

        let z2 = x2.clone().matmul(state.w2.clone()) + state.b2.clone().unsqueeze_dim(2);

        let reconstruction_target = qkv.xv - qkv.xk;

        let (_ln_out, grad_l_wrt_z2) = self
            .layer_norm
            .forward_and_l2_grad(z2, reconstruction_target);

        let grad_l_wrt_z1 =
            grad_l_wrt_z2.clone().matmul(state.w2.clone().transpose()) * gelu_bwd(z1);

        let token_eta = inputs.token_eta.unsqueeze_dims::<4>(&[0, 0, -1]); // [1, 1, K, 1]
        let ttt_lr_eta = inputs.ttt_lr_eta.unsqueeze_dim::<4>(2); // [B, H, 1, K]
        let eta_combined = token_eta * ttt_lr_eta;

        let eta_batch = eta_combined.tril(0); // [B, H, K, K]

        let attn_scores1 = qkv.xq.clone().matmul(x1.clone().transpose());

        let attn1 = attn_scores1.tril(0);

        let b1_bar =
            state.b1.clone().unsqueeze_dim(2) - eta_batch.clone().matmul(grad_l_wrt_z1.clone());

        let z1_bar = qkv.xq.clone().matmul(state.w1.clone())
            - (eta_batch.clone() * attn1).matmul(grad_l_wrt_z1.clone())
            + b1_bar;

        let x2_bar = gelu_tanh(z1_bar);

        let attn2 = x2_bar.clone().matmul(x2.clone().transpose()).tril(0);

        let b2_bar =
            state.b2.clone().unsqueeze_dim(2) - eta_batch.clone().matmul(grad_l_wrt_z2.clone());

        let z2_bar = x2_bar.matmul(state.w2.clone())
            - (eta_batch.clone() * attn2).matmul(grad_l_wrt_z2.clone())
            + b2_bar;

        let last_eta_row = eta_batch.slice(s![.., .., -1.., ..]);
        let last_eta_col = last_eta_row.transpose(); // [B, H, K, 1]

        state.w1 = state.w1.clone()
            - (last_eta_col.clone() * x1)
                .transpose()
                .matmul(grad_l_wrt_z1.clone());

        state.b1 = state.b1.clone()
            - (last_eta_col.clone() * grad_l_wrt_z1)
                .sum_dim(2)
                .squeeze_dim::<3>(2);

        state.w2 = state.w2.clone()
            - (last_eta_col.clone() * x2)
                .transpose()
                .matmul(grad_l_wrt_z2.clone());

        state.b2 = state.b2.clone()
            - (last_eta_col * grad_l_wrt_z2)
                .sum_dim(2)
                .squeeze_dim::<3>(2);

        let z2_bar_normalized = self.layer_norm.forward(z2_bar);

        qkv.xq + z2_bar_normalized
    }
}
