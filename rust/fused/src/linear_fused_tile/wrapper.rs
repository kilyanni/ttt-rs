//! TTTInnerModel implementation for the tiled TTT-Linear kernel.

use std::{ops::Range, sync::Arc};

use burn::tensor::Tensor;
use ttt_core::{TTTInnerModel, TTTInputsInner, TTTLinear, config::ModelConfig};

use super::api::{fused_ttt_tile_forward, fused_ttt_tile_forward_multi};
use crate::{Fused, FusedTttBackend, FusedTttConfig, TileKernel, TileMultiKernel};

/// TTTInnerModel implementation for the tiled fused kernel.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, TileKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTileTTTLinear"
    }

    fn new(config: &ModelConfig, inner_config: &Arc<Self::Config>, device: &B::Device) -> Self {
        Fused::new(TTTLinear::new(config, inner_config, device))
    }

    fn get_config(&self) -> &ModelConfig {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.inner.init_state(batch_size)
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        let inputs = inputs.slice_seq(range);

        let inner = &self.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let inner_config = self.inner.get_config();
        let threads = inner_config
            .ttt
            .threads
            .unwrap_or_else(|| super::api::default_threads(seq_len, head_dim));
        let config =
            FusedTttConfig::new(inner_config.ttt.mini_batch_size, head_dim, epsilon, threads);

        let (output, weight_updated, bias_updated) = fused_ttt_tile_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.weight.clone(),
            state.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            config,
        );

        state.weight = weight_updated;
        state.bias = bias_updated;

        output
    }
}

/// TTTInnerModel implementation for the multi-stage tiled fused kernel.
///
/// This implementation overrides `forward()` to process all mini-batches in a
/// single kernel launch, rather than launching the kernel once per mini-batch.
impl<B: FusedTttBackend> TTTInnerModel<B> for Fused<B, TTTLinear<B>, TileMultiKernel> {
    type Config = <TTTLinear<B> as TTTInnerModel<B>>::Config;
    type State = <TTTLinear<B> as TTTInnerModel<B>>::State;

    fn name() -> &'static str {
        "FusedTileMultiTTTLinear"
    }

    fn new(config: &ModelConfig, inner_config: &Arc<Self::Config>, device: &B::Device) -> Self {
        Fused::new(TTTLinear::new(config, inner_config, device))
    }

    fn get_config(&self) -> &ModelConfig {
        self.inner.get_config()
    }

    fn init_state(&self, batch_size: usize) -> Self::State {
        self.inner.init_state(batch_size)
    }

    /// Override forward to use multi-stage kernel for full sequence processing.
    fn forward(&self, state: &mut Self::State, inputs: TTTInputsInner<B>) -> Tensor<B, 4> {
        let inner = &self.inner;
        let model_config = inner.get_config();
        let mini_batch_size = model_config.ttt.mini_batch_size;

        let [_batch_size, _num_heads, seq_len, head_dim] = inputs.qkv.xv.shape().dims();
        let num_full_batches = seq_len / mini_batch_size;
        let remainder = seq_len % mini_batch_size;

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let threads = model_config
            .ttt
            .threads
            .unwrap_or_else(|| super::api::default_threads(mini_batch_size, head_dim));
        let mut config = FusedTttConfig::new(mini_batch_size, head_dim, epsilon, threads);
        config.checkpoint_interval = model_config.ttt.checkpoint_interval;

        if num_full_batches > 0 {
            // Process full mini-batches with multi-stage kernel
            let full_seq_len = num_full_batches * mini_batch_size;
            let full_qkv = inputs.qkv.slice_seq(0..full_seq_len);
            let [batch_size, num_heads, _] = inputs.ttt_lr_eta.shape().dims();
            let full_ttt_lr_eta =
                inputs
                    .ttt_lr_eta
                    .clone()
                    .slice([0..batch_size, 0..num_heads, 0..full_seq_len]);

            // token_eta: pass full [seq_len] tensor so backward can output per-position gradients.
            // The kernel reads only [0..mini_batch_len] values (same across stages).
            let token_eta = inputs.token_eta.clone();
            let (output, weight_updated, bias_updated) = fused_ttt_tile_forward_multi::<B>(
                full_qkv.xq,
                full_qkv.xk,
                full_qkv.xv,
                state.weight.clone(),
                state.bias.clone(),
                token_eta,
                full_ttt_lr_eta,
                ln_weight.clone(),
                ln_bias.clone(),
                config,
            );

            state.weight = weight_updated;
            state.bias = bias_updated;

            if remainder == 0 {
                output
            } else {
                let remainder_output =
                    self.forward_mini_batch(state, &inputs, full_seq_len..seq_len);

                Tensor::cat(vec![output, remainder_output], 2)
            }
        } else {
            // Sequence shorter than mini_batch_size, use single-stage kernel
            self.forward_mini_batch(state, &inputs, 0..seq_len)
        }
    }

    fn forward_mini_batch(
        &self,
        state: &mut Self::State,
        inputs: &TTTInputsInner<B>,
        range: Range<usize>,
    ) -> Tensor<B, 4> {
        assert!(range.len() == self.inner.config.ttt.mini_batch_size, "Sequence length must be equal to mini_batch_size");

        let inputs = inputs.slice_seq(range);

        let inner = &self.inner;

        let qkv = inputs.qkv;
        let [_batch_size, _num_heads, seq_len, head_dim] = qkv.xq.shape().dims();

        let ln_weight = inner.layer_norm.weight.val();
        let ln_bias = inner.layer_norm.bias.val();
        let epsilon = inner.layer_norm.epsilon as f32;

        let threads = inner
            .config
            .ttt
            .threads
            .unwrap_or_else(|| super::api::default_threads(seq_len, head_dim));
        let config =
            FusedTttConfig::new(inner.config.ttt.mini_batch_size, head_dim, epsilon, threads);

        let (output, weight_updated, bias_updated) = fused_ttt_tile_forward(
            qkv.xq,
            qkv.xk,
            qkv.xv,
            state.weight.clone(),
            state.bias.clone(),
            inputs.token_eta,
            inputs.ttt_lr_eta,
            ln_weight,
            ln_bias,
            config,
        );

        state.weight = weight_updated;
        state.bias = bias_updated;

        output
    }
}
