use std::{ops::Range, sync::Arc};

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    tensor::{Tensor, s},
};
use ttt_kernels::{gelu_bwd, gelu_tanh};

use crate::{
    config::ModelConfig,
    inner::{TTTInnerModel, TTTInputsInner},
    mlp::GeluBackend,
    util::{MultiHeadLayerNorm, MultiHeadLayerNormConfig},
};

/// MLP with 3 hidden layers (4 weight layers total)
/// input(D) -> hidden1(4D) -> hidden2(4D) -> hidden3(4D) -> output(D)
#[derive(Module, Debug)]
pub struct TTTMLP3<B: GeluBackend> {
    /// First layer weight: [num_heads, head_dim, 4*head_dim]
    pub w1_init: Param<Tensor<B, 3>>,
    /// First layer bias: [num_heads, 4*head_dim]
    pub b1_init: Param<Tensor<B, 2>>,
    /// Second layer weight: [num_heads, 4*head_dim, 4*head_dim]
    pub w2_init: Param<Tensor<B, 3>>,
    /// Second layer bias: [num_heads, 4*head_dim]
    pub b2_init: Param<Tensor<B, 2>>,
    /// Third layer weight: [num_heads, 4*head_dim, 4*head_dim]
    pub w3_init: Param<Tensor<B, 3>>,
    /// Third layer bias: [num_heads, 4*head_dim]
    pub b3_init: Param<Tensor<B, 2>>,
    /// Fourth layer weight: [num_heads, 4*head_dim, head_dim]
    pub w4_init: Param<Tensor<B, 3>>,
    /// Fourth layer bias: [num_heads, head_dim]
    pub b4_init: Param<Tensor<B, 2>>,
    pub layer_norm: MultiHeadLayerNorm<B>,
    pub config: Ignored<ModelConfig>,
}

#[derive(Module, Debug)]
pub struct TTTMLP3State<B: GeluBackend> {
    pub w1: Tensor<B, 4>,
    pub b1: Tensor<B, 3>,
    pub w2: Tensor<B, 4>,
    pub b2: Tensor<B, 3>,
    pub w3: Tensor<B, 4>,
    pub b3: Tensor<B, 3>,
    pub w4: Tensor<B, 4>,
    pub b4: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub struct TTTMLP3Config {
    #[config(default = "Initializer::Normal{mean:0.0, std:0.02}")]
    pub initializer: Initializer,
}

impl Default for TTTMLP3Config {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: GeluBackend> TTTInnerModel<B> for TTTMLP3<B> {
    type Config = TTTMLP3Config;
    type State = TTTMLP3State<B>;

    fn name() -> &'static str {
        "TTTMLP3"
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
                [num_heads, mlp_dim, mlp_dim],
                Some(len),
                Some(len),
                device,
            ),
            b2_init: inner_config.initializer.init_with(
                [num_heads, mlp_dim],
                Some(len),
                Some(len),
                device,
            ),
            w3_init: inner_config.initializer.init_with(
                [num_heads, mlp_dim, mlp_dim],
                Some(len),
                Some(len),
                device,
            ),
            b3_init: inner_config.initializer.init_with(
                [num_heads, mlp_dim],
                Some(len),
                Some(len),
                device,
            ),
            w4_init: inner_config.initializer.init_with(
                [num_heads, mlp_dim, head_dim],
                Some(len),
                Some(len),
                device,
            ),
            b4_init: inner_config.initializer.init_with(
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
        let w3 = self.w3_init.val().unsqueeze().repeat_dim(0, batch_size);
        let b3 = self.b3_init.val().unsqueeze().repeat_dim(0, batch_size);
        let w4 = self.w4_init.val().unsqueeze().repeat_dim(0, batch_size);
        let b4 = self.b4_init.val().unsqueeze().repeat_dim(0, batch_size);

        TTTMLP3State {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            w4,
            b4,
        }
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
        let x3 = gelu_tanh(z2.clone());

        let z3 = x3.clone().matmul(state.w3.clone()) + state.b3.clone().unsqueeze_dim(2);
        let x4 = gelu_tanh(z3.clone());

        let z4 = x4.clone().matmul(state.w4.clone()) + state.b4.clone().unsqueeze_dim(2);

        let reconstruction_target = qkv.xv - qkv.xk;

        let (_ln_out, grad_l_wrt_z4) = self
            .layer_norm
            .forward_and_l2_grad(z4, reconstruction_target);

        let grad_l_wrt_z3 =
            grad_l_wrt_z4.clone().matmul(state.w4.clone().transpose()) * gelu_bwd(z3);

        let grad_l_wrt_z2 =
            grad_l_wrt_z3.clone().matmul(state.w3.clone().transpose()) * gelu_bwd(z2);

        let grad_l_wrt_z1 =
            grad_l_wrt_z2.clone().matmul(state.w2.clone().transpose()) * gelu_bwd(z1);

        let token_eta = inputs.token_eta.unsqueeze_dims::<4>(&[0, 0, -1]);
        let ttt_lr_eta = inputs.ttt_lr_eta.unsqueeze_dim::<4>(2);
        let eta_combined = token_eta * ttt_lr_eta;
        let eta_batch = eta_combined.tril(0);

        let attn1 = qkv.xq.clone().matmul(x1.clone().transpose()).tril(0);
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
        let x3_bar = gelu_tanh(z2_bar);

        let attn3 = x3_bar.clone().matmul(x3.clone().transpose()).tril(0);
        let b3_bar =
            state.b3.clone().unsqueeze_dim(2) - eta_batch.clone().matmul(grad_l_wrt_z3.clone());
        let z3_bar = x3_bar.matmul(state.w3.clone())
            - (eta_batch.clone() * attn3).matmul(grad_l_wrt_z3.clone())
            + b3_bar;
        let x4_bar = gelu_tanh(z3_bar);

        let attn4 = x4_bar.clone().matmul(x4.clone().transpose()).tril(0);
        let b4_bar =
            state.b4.clone().unsqueeze_dim(2) - eta_batch.clone().matmul(grad_l_wrt_z4.clone());
        let z4_bar = x4_bar.matmul(state.w4.clone())
            - (eta_batch.clone() * attn4).matmul(grad_l_wrt_z4.clone())
            + b4_bar;

        let last_eta_row = eta_batch.slice(s![.., .., -1.., ..]);
        let last_eta_col = last_eta_row.transpose();

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
            - (last_eta_col.clone() * grad_l_wrt_z2)
                .sum_dim(2)
                .squeeze_dim::<3>(2);

        state.w3 = state.w3.clone()
            - (last_eta_col.clone() * x3)
                .transpose()
                .matmul(grad_l_wrt_z3.clone());
        state.b3 = state.b3.clone()
            - (last_eta_col.clone() * grad_l_wrt_z3)
                .sum_dim(2)
                .squeeze_dim::<3>(2);

        state.w4 = state.w4.clone()
            - (last_eta_col.clone() * x4)
                .transpose()
                .matmul(grad_l_wrt_z4.clone());
        state.b4 = state.b4.clone()
            - (last_eta_col * grad_l_wrt_z4)
                .sum_dim(2)
                .squeeze_dim::<3>(2);

        let z4_bar_normalized = self.layer_norm.forward(z4_bar);

        qkv.xq + z4_bar_normalized
    }
}
