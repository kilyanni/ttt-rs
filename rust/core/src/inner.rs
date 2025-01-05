use std::{ops::Range, sync::Arc};

use burn::{
    config::Config,
    module::{Module, ModuleDisplay},
    prelude::*,
    tensor::Tensor,
};
use burn_backend::Distribution;

use crate::config::ModelConfig;

pub trait TTTInnerModel<B: Backend>: Module<B> + ModuleDisplay {
    type Config: Config + Default;
    type State: Module<B> + ModuleDisplay;

    fn name() -> &'static str;

    fn new(config: &ModelConfig, inner_config: &Arc<Self::Config>, device: &B::Device) -> Self;

    fn init_state(&self, batch_size: usize) -> Self::State;

    fn get_config(&self) -> &ModelConfig;

    fn forward(&self, state: &mut Self::State, inputs: TTTInputsInner<B>) -> Tensor<B, 4> {
        let mut output = inputs.qkv.xv.zeros_like();

        let [_batch_size, _num_heads, seq_len, _head_dim] = inputs.qkv.xv.shape().dims();

        let mini_batch_size = self.get_config().ttt.mini_batch_size;
        let num_mini_batch = seq_len / mini_batch_size;

        for i in 0..num_mini_batch {
            let start_idx = i * mini_batch_size;
            let r = start_idx..start_idx + mini_batch_size;
            let z = self.forward_mini_batch(state, &inputs, r.clone());
            output = output.slice_assign(s![.., .., r, ..,], z);
        }

        let last_mini_batch_end = num_mini_batch * mini_batch_size;

        // Process any remaining tokens in a single batch (should work for any seq_len < mini_batch_size)
        if last_mini_batch_end < seq_len {
            let r = last_mini_batch_end..seq_len;
            let z = self.forward_mini_batch(state, &inputs, r.clone());
            output = output.slice_assign(s![.., .., r, ..,], z);
        }

        output
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4>;
}

#[derive(Clone)]
pub struct Qkv<B: Backend> {
    /// [batch_size, num_heads, seq_len, head_dim]
    pub xq: Tensor<B, 4>,
    /// [batch_size, num_heads, seq_len, head_dim]
    pub xk: Tensor<B, 4>,
    /// [batch_size, num_heads, seq_len, head_dim]
    pub xv: Tensor<B, 4>,
}

impl<B: Backend> Qkv<B> {
    #[must_use]
    pub fn slice_seq(&self, range: Range<usize>) -> Self {
        Self {
            xq: self.xq.clone().slice(s![.., .., range.clone(), ..]),
            xk: self.xk.clone().slice(s![.., .., range.clone(), ..]),
            xv: self.xv.clone().slice(s![.., .., range, ..]),
        }
    }

    pub fn random(
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            xq: Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                Distribution::Default,
                device,
            ),
            xk: Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                Distribution::Default,
                device,
            ),
            xv: Tensor::random(
                [batch_size, num_heads, seq_len, head_dim],
                Distribution::Default,
                device,
            ),
        }
    }
}

#[derive(Clone)]
pub struct TTTInputsInner<B: Backend> {
    /// The key, query and value vectors
    pub qkv: Qkv<B>,
    /// Position-based token scale factor (1/k + learnable offset), clamped >= 0
    /// `[seq_len]`
    pub token_eta: Tensor<B, 1>,
    /// Per-head learning rate (base_lr * sigmoid(learned) / head_dim)
    /// `[batch_size, num_heads, seq_len]`
    pub ttt_lr_eta: Tensor<B, 3>,
    /// Index of the first token in the sequence
    pub start_idx: usize,
}

impl<B: Backend> TTTInputsInner<B> {
    #[must_use]
    pub fn slice_seq(&self, range: Range<usize>) -> Self {
        Self {
            qkv: self.qkv.slice_seq(range.clone()),
            token_eta: self.token_eta.clone().slice(s![range.clone()]),
            start_idx: range.start,
            ttt_lr_eta: self.ttt_lr_eta.clone().slice(s![.., .., range.clone()]),
        }
    }

    pub fn random(
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            qkv: Qkv::random(batch_size, num_heads, seq_len, head_dim, device),
            token_eta: Tensor::random(
                [seq_len],
                burn::tensor::Distribution::Normal(0.0, 0.1),
                device,
            ) + Tensor::arange(0..(seq_len as i64), device).float().recip(),
            start_idx: 0,
            ttt_lr_eta: Tensor::random(
                [batch_size, num_heads, seq_len],
                burn::tensor::Distribution::Default,
                device,
            ),
        }
    }
}
