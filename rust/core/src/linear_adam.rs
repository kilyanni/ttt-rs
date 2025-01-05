use std::{ops::Range, sync::Arc};

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{Tensor, s},
};
use burn_backend::FloatDType;

use crate::{
    config::ModelConfig,
    inner::{TTTInnerModel, TTTInputsInner},
    util::{MultiHeadLayerNorm, MultiHeadLayerNormConfig},
};

#[derive(Module, Debug)]
pub struct TTTLinearAdam<B: Backend> {
    /// [num_heads, head_dim, head_dim]
    pub weight_init: Param<Tensor<B, 3>>,
    /// [num_heads, head_dim]
    pub bias_init: Param<Tensor<B, 2>>,
    pub layer_norm: MultiHeadLayerNorm<B>,
    pub config: Ignored<ModelConfig>,
    pub adam_config: Ignored<Arc<TTTLinearAdamConfig>>,
}

#[derive(Module, Debug)]
pub struct TTTLinearAdamState<B: Backend> {
    /// Current weight parameters [batch_size, num_heads, head_dim, head_dim]
    pub weight: Tensor<B, 4>,
    /// Current bias parameters [batch_size, num_heads, head_dim]
    pub bias: Tensor<B, 3>,
    /// First moment estimate for weight [batch_size, num_heads, head_dim, head_dim]
    pub weight_m: Tensor<B, 4>,
    /// First moment estimate for bias [batch_size, num_heads, head_dim]
    pub bias_m: Tensor<B, 3>,
    /// Second moment estimate for weight [batch_size, num_heads, head_dim, head_dim]
    pub weight_v: Tensor<B, 4>,
    /// Second moment estimate for bias [batch_size, num_heads, head_dim]
    pub bias_v: Tensor<B, 3>,
    /// Step counter
    pub step: i32,
}

#[derive(Config, Debug)]
pub struct TTTLinearAdamConfig {
    #[config(default = "Initializer::Normal{mean:0.0, std:0.02}")]
    pub initializer: Initializer,
    /// Exponential decay rate for first moment estimates
    #[config(default = 0.9)]
    pub beta1: f32,
    /// Exponential decay rate for second moment estimates
    #[config(default = 0.999)]
    pub beta2: f32,
    /// Small constant for numerical stability
    #[config(default = 1e-8)]
    pub epsilon: f32,
    /// Inner Adam learning rate (scales the adaptive update)
    #[config(default = 0.001)]
    pub lr: f32,
}

impl Default for TTTLinearAdamConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TTTLinearAdam<B> {
    /// Apply Adam update to the state parameters.
    ///
    /// The eta-weighted gradients (with full per-token token_eta and ttt_lr_eta)
    /// are passed in, same as the SGD version. A scalar `lr` from the config
    /// controls the step magnitude, compensating for Adam's normalization
    /// which would otherwise make steps ~sign(g) ≈ ±1 per element.
    ///
    /// m = β1*m + (1-β1)*g
    /// v = β2*v + (1-β2)*g²
    /// m̂ = m / (1 - β1^t)
    /// v̂ = v / (1 - β2^t)
    /// θ = θ - lr · m̂ / (√v̂ + ε)
    fn adam_update(
        &self,
        state: &mut TTTLinearAdamState<B>,
        weight_grad: Tensor<B, 4>,
        bias_grad: Tensor<B, 3>,
    ) {
        let beta1 = self.adam_config.beta1;
        let beta2 = self.adam_config.beta2;
        let eps = self.adam_config.epsilon;
        let lr = self.adam_config.lr;

        state.step += 1;

        // Detach gradients so the outer backward pass does not differentiate
        // through the Adam moment updates
        let weight_grad = weight_grad.detach();
        let bias_grad = bias_grad.detach();

        state.weight_m = state.weight_m.clone() * beta1 + weight_grad.clone() * (1.0 - beta1);
        state.bias_m = state.bias_m.clone() * beta1 + bias_grad.clone() * (1.0 - beta1);

        // Square and accumulate v in f32 to avoid bf16 overflow.
        state.weight_v = (state.weight_v.clone().cast(FloatDType::F32) * beta2
            + weight_grad.cast(FloatDType::F32).powf_scalar(2.0) * (1.0 - beta2))
            .no_grad();
        state.bias_v = (state.bias_v.clone().cast(FloatDType::F32) * beta2
            + bias_grad.cast(FloatDType::F32).powf_scalar(2.0) * (1.0 - beta2))
            .no_grad();

        let beta1_pow = 1.0 - beta1.powi(state.step);
        let beta2_pow = 1.0 - beta2.powi(state.step);

        let weight_m_hat = state.weight_m.clone() / beta1_pow;
        let bias_m_hat = state.bias_m.clone() / beta1_pow;

        // Bias-correct and normalize v in f32, then cast back to original dtype.
        let dtype = state.weight.dtype();
        let weight_v_hat = state.weight_v.clone().cast(FloatDType::F32) / beta2_pow;
        let bias_v_hat = state.bias_v.clone().cast(FloatDType::F32) / beta2_pow;

        state.weight =
            state.weight.clone() - weight_m_hat / (weight_v_hat.sqrt() + eps).cast(dtype) * lr;
        state.bias = state.bias.clone() - bias_m_hat / (bias_v_hat.sqrt() + eps).cast(dtype) * lr;
    }
}

impl<B: Backend> TTTInnerModel<B> for TTTLinearAdam<B> {
    type Config = TTTLinearAdamConfig;
    type State = TTTLinearAdamState<B>;

    fn name() -> &'static str {
        "TTTLinearAdam"
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
            adam_config: Ignored(inner_config.clone()),
        }
    }

    fn get_config(&self) -> &ModelConfig {
        &self.config.0
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        let weight = self.weight_init.val().unsqueeze().repeat_dim(0, batch_size);
        let bias = self.bias_init.val().unsqueeze().repeat_dim(0, batch_size);

        TTTLinearAdamState {
            weight_m: weight.zeros_like(),
            bias_m: bias.zeros_like(),
            weight_v: weight.zeros_like(),
            bias_v: bias.zeros_like(),
            step: 0,
            weight,
            bias,
        }
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        let inputs = inputs.slice_seq(range);

        // For reference:
        // qkv: [batch_size, num_heads, seq_len, head_dim]
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
        let eta_batch = eta_combined.tril(0); // [B, H, seq_len, seq_len]

        let attn1 = qkv.xq.clone().matmul(x1.clone().transpose()).tril(0);

        let b1_bar =
            state.bias.clone().unsqueeze_dim(2) - eta_batch.clone().matmul(grad_l_wrt_z1.clone());

        let z1_bar = qkv.xq.clone().matmul(state.weight.clone())
            - (eta_batch.clone() * attn1).matmul(grad_l_wrt_z1.clone())
            + b1_bar;

        let last_eta_row = eta_batch.slice(s![.., .., -1.., ..]);
        let last_eta_col = last_eta_row.transpose(); // [B, H, K, 1]

        let weight_grad = (last_eta_col.clone() * x1)
            .transpose()
            .matmul(grad_l_wrt_z1.clone());

        let bias_grad = (last_eta_col * grad_l_wrt_z1)
            .sum_dim(2)
            .squeeze_dim::<3>(2);

        self.adam_update(state, weight_grad, bias_grad);

        let z1_bar_normalized = self.layer_norm.forward(z1_bar);

        qkv.xq + z1_bar_normalized
    }
}
