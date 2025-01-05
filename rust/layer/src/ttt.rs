//! TTT struct - the outer layer that wraps inner model implementations.

use burn::{
    module::{Ignored, Module},
    nn::{
        Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    prelude::*,
    tensor::{Tensor, activation::sigmoid},
};
use ttt_config::PosEncoding;
use ttt_core::{
    Qkv, TTTInputsInner,
    config::ModelConfig,
    util::{RotaryEmbedding, RotaryEmbeddingConfig, causal_conv1d_fn},
};
use ttt_fused::FusedTttBackend;
use ttt_kernels::gelu_tanh;

use crate::any_inner::{AnyInner, AnyInnerState};

/// Permute Q/K dimensions to match JAX/EasyLM rotary embedding format.
/// Required for byte-for-byte compatibility with the reference implementation.
/// Reference: https://github.com/young-geng/EasyLM/blob/981a2ed/EasyLM/models/llama/convert_hf_to_easylm.py#L33
fn permute_qk<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch_size, num_heads, seq_len, head_dim] = x.shape().dims();
    // [B, H, L, D] -> [B, H, L, D//2, 2] -> transpose(3,4) -> [B, H, L, 2, D//2] -> [B, H, L, D]
    x.reshape([batch_size, num_heads, seq_len, head_dim / 2, 2])
        .permute([0, 1, 2, 4, 3])
        .reshape([batch_size, num_heads, seq_len, head_dim])
}

/// Undo the Q/K permutation after rotary embedding.
fn undo_permute_qk<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch_size, num_heads, seq_len, head_dim] = x.shape().dims();
    // [B, H, L, D] -> [B, H, L, 2, D//2] -> transpose(3,4) -> [B, H, L, D//2, 2] -> [B, H, L, D]
    x.reshape([batch_size, num_heads, seq_len, 2, head_dim / 2])
        .permute([0, 1, 2, 4, 3])
        .reshape([batch_size, num_heads, seq_len, head_dim])
}

#[derive(Module, Debug)]
pub struct TTT<B: Backend> {
    pub q_proj: Linear<B>,
    /// Separate K projection (only present if share_qk is false)
    pub k_proj: Option<Linear<B>>,
    pub v_proj: Linear<B>,
    pub g_proj: Option<Linear<B>>, // Only present if use_gate is true
    pub o_proj: Linear<B>,
    pub q_conv: Conv1d<B>,
    pub k_conv: Conv1d<B>,
    /// Per-head learning rate weight: `[num_heads, token_size, 1]`
    pub learnable_ttt_lr_weight: Tensor<B, 3>,
    /// Per-head learning rate bias: `[num_heads, 1]`
    pub learnable_ttt_lr_bias: Tensor<B, 2>,
    /// Base token eta: `1/k` for `k=1..mini_batch_size`. `[mini_batch_size]`
    pub token_idx: Tensor<B, 1>,
    /// Learnable offset added to `token_idx`. `[mini_batch_size]`
    pub learnable_token_idx: Tensor<B, 1>,
    pub post_norm: LayerNorm<B>,
    pub config: Ignored<ModelConfig>,
    pub rot_enc: Option<RotaryEmbedding<B>>,
}

/// Extension trait for ModelConfig to initialize TTT layers.
pub trait ModelConfigExt {
    fn init_ttt_seq<B: Backend>(&self, device: &B::Device) -> TTT<B>;
}

impl ModelConfigExt for ModelConfig {
    fn init_ttt_seq<B: Backend>(&self, device: &B::Device) -> TTT<B> {
        let hidden_size = self.arch.hidden_size;
        let num_heads = self.arch.num_heads;
        let head_dim = self.head_dim();
        let mini_batch_size = self.ttt.mini_batch_size;

        let linear = |in_size, out_size, bias| {
            LinearConfig::new(in_size, out_size)
                .with_bias(bias)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device)
        };

        let conv = |size| {
            Conv1dConfig::new(size, size, self.ttt.conv_kernel_size)
                .with_groups(size)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(
                    self.ttt.conv_kernel_size - 1,
                ))
                .with_bias(true)
                .init(device)
        };

        let learnable_ttt_lr_weight = Tensor::random(
            [num_heads, hidden_size, 1],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        );

        let learnable_ttt_lr_bias = Tensor::zeros([num_heads, 1], device);

        let token_idx = Tensor::arange(1..(mini_batch_size as i64 + 1), device)
            .float()
            .recip();

        let learnable_token_idx = Tensor::zeros([mini_batch_size], device);

        TTT {
            q_proj: linear(hidden_size, hidden_size, false),
            k_proj: if self.ttt.share_qk {
                None
            } else {
                Some(linear(hidden_size, hidden_size, false))
            },
            v_proj: linear(hidden_size, hidden_size, false),
            g_proj: if self.ttt.use_gate {
                Some(linear(hidden_size, hidden_size, false))
            } else {
                None
            },
            o_proj: linear(hidden_size, hidden_size, false),
            q_conv: conv(hidden_size),
            k_conv: conv(hidden_size),
            learnable_ttt_lr_weight,
            learnable_ttt_lr_bias,
            token_idx,
            learnable_token_idx,
            post_norm: LayerNormConfig::new(hidden_size)
                .with_epsilon(1e-6)
                .init(device),
            rot_enc: match self.ttt.pos_encoding {
                PosEncoding::Rope | PosEncoding::RopeGlobal => Some(
                    RotaryEmbeddingConfig::new(head_dim)
                        .with_base(f64::from(self.ttt.rope_theta))
                        .init(device),
                ),
                _ => None,
            },
            config: Ignored(self.clone()),
        }
    }
}

impl<B: FusedTttBackend> TTT<B> {
    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, token_dim]`
    /// - `start_idx`: Starting position index for the sequence
    fn get_qkv(&self, x: Tensor<B, 3>, start_idx: usize) -> Qkv<B> {
        let [batch_size, seq_len, _token_dim] = x.shape().dims();

        let xv = self.v_proj.forward(x.clone());

        let (xq, xk) = if let Some(k_proj) = &self.k_proj {
            let xq = self.q_proj.forward(x.clone());
            let xk = k_proj.forward(x);
            (xq, xk)
        } else {
            let xqk = self.q_proj.forward(x);
            self.conv_qk(xqk)
        };

        // [B, seq_len, num_heads*dim] -> [B, num_heads, seq_len, dim]
        let [xq, xk, xv] = [xq, xk, xv].map(|x| {
            x.reshape([
                batch_size,
                seq_len,
                self.config.arch.num_heads,
                self.config.head_dim(),
            ])
            .permute([0, 2, 1, 3])
        });

        // Apply rotary position encoding
        // Use permute_qk/undo_permute_qk to match JAX/EasyLM format (see doc comment on permute_qk)
        let (xq, xk) = match &self.rot_enc {
            Some(rot_enc) => {
                let xq = permute_qk(xq);
                let xk = permute_qk(xk);
                let (offset, wrap) = match self.config.ttt.pos_encoding {
                    PosEncoding::RopeGlobal => (start_idx, None),
                    _ => (
                        start_idx % self.config.ttt.mini_batch_size,
                        Some(self.config.ttt.mini_batch_size),
                    ),
                };
                let (xq, xk) = rot_enc.apply(xq, xk, offset, wrap);
                (undo_permute_qk(xq), undo_permute_qk(xk))
            }
            None => (xq, xk),
        };

        Qkv { xq, xk, xv }
    }

    /// Gets the learning rate for each head of each token using per-head weights
    ///
    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, token_dim]`
    ///
    /// Returns a tensor of shape `[batch_size, num_heads, seq_len]`
    fn get_lr(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let lr_weight = self.learnable_ttt_lr_weight.clone().squeeze_dim::<2>(2); // [num_heads, token_size]

        let lr_weight_t = lr_weight.transpose(); // [token_size, num_heads]

        let lr_weight_t = lr_weight_t.unsqueeze_dim::<3>(0); // [1, token_size, num_heads]

        let lr = x.matmul(lr_weight_t); // [B, seq_len, num_heads]

        let lr_bias_expanded = self
            .learnable_ttt_lr_bias
            .clone()
            .squeeze_dim::<1>(1) // [num_heads]
            .unsqueeze_dim::<2>(0) // [1, num_heads]
            .unsqueeze_dim::<3>(0); // [1, 1, num_heads]

        let lr = lr + lr_bias_expanded; // [B, seq_len, num_heads]

        // [B, seq_len, num_heads] -> [B, num_heads, seq_len]
        let lr_sigmoid = sigmoid(lr.permute([0, 2, 1]));

        (self.config.ttt.base_lr * lr_sigmoid) / (self.config.head_dim() as f32)
    }

    fn conv_qk(&self, xqk: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let conv_q_weight = self.q_conv.weight.val();
        let conv_k_weight = self.k_conv.weight.val();

        let conv_q_bias = self.q_conv.bias.as_ref().map(burn::module::Param::val);
        let conv_k_bias = self.k_conv.bias.as_ref().map(burn::module::Param::val);

        let xqk_transposed = xqk.permute([0, 2, 1]);

        let xq_transposed = causal_conv1d_fn(xqk_transposed.clone(), conv_q_weight, conv_q_bias);
        let xk_transposed = causal_conv1d_fn(xqk_transposed, conv_k_weight, conv_k_bias);

        let xq = xq_transposed.permute([0, 2, 1]);
        let xk = xk_transposed.permute([0, 2, 1]);

        (xq, xk)
    }

    /// Parameters:
    /// - `x`: The input tensor of shape `[batch_size, seq_len, dim]`.
    /// - `start_idx`: Starting index for positional encoding
    fn get_inner_loop_inputs(&self, x: Tensor<B, 3>, start_idx: usize) -> TTTInputsInner<B> {
        let [_batch_size, seq_len, _] = x.shape().dims();

        let token_eta = (self.token_idx.clone() + self.learnable_token_idx.clone())
            .repeat_dim(0, seq_len.div_ceil(self.config.ttt.mini_batch_size))
            .slice(s![0..seq_len])
            .clamp_min(0.);

        let qkv = self.get_qkv(x.clone(), start_idx);
        let ttt_lr_eta = self.get_lr(x);

        TTTInputsInner {
            qkv,
            token_eta,
            ttt_lr_eta,
            start_idx,
        }
    }

    pub fn forward(
        &self,
        // [batch_size, seq_len, token_size]
        x: Tensor<B, 3>,
        inner: &AnyInner<B>,
        state: &mut AnyInnerState<B>,
        start_idx: usize,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _token_size] = x.shape().dims();

        let inputs = self.get_inner_loop_inputs(x.clone(), start_idx);

        debug_assert_eq!(batch_size, inputs.qkv.xq.shape().dims[0]);

        let out = inner.forward(state, inputs);

        let out =
            out.permute([0, 2, 1, 3])
                .reshape([batch_size, seq_len, self.config.arch.hidden_size]);

        let out = self.post_norm.forward(out);

        let out = if let Some(g_proj) = &self.g_proj {
            let gate = g_proj.forward(x);
            // Use tanh approximation for GELU to match JAX implementation
            gelu_tanh(gate) * out
        } else {
            out
        };

        self.o_proj.forward(out)
    }
}
