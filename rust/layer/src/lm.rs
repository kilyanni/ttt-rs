use burn::{
    module::{Ignored, Module},
    nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{Int, Tensor},
};
use ttt_config::{InnerModel, MixPattern, PosEncoding};
use ttt_core::config::ModelConfig;
use ttt_fused::FusedTttBackend;

use crate::{
    any_inner::{AnyInner, AnyInnerState},
    block::{TTTBlockConfig, TTTBlockWithInner},
};

/// TTT language model with per-layer inner model selection.
#[derive(Module, Debug)]
pub struct TTTModel<B: FusedTttBackend> {
    pub config: Ignored<ModelConfig>,
    pub embedding: Embedding<B>,
    /// Optional absolute position embeddings (only present when pos_encoding is Absolute)
    pub position_embedding: Option<Embedding<B>>,
    pub layers: Vec<TTTBlockWithInner<B>>,
    pub norm: RmsNorm<B>,
    /// Optional separate lm_head (only present when tie_word_embeddings is false).
    /// When None, uses tied embedding weights via matmul.
    /// When Some, uses separate Linear layer (more performant, single kernel call).
    pub lm_head: Option<Linear<B>>,
}

/// Extension trait for ModelConfig to initialize TTT models.
pub trait ModelConfigModelExt {
    /// Initialize a TTT model with a factory function for creating inner models.
    ///
    /// The factory receives the layer index and returns the inner model type,
    /// allowing different inner model types per layer (e.g., Linear for even layers,
    /// MLP for odd layers).
    fn init_with_inner_by<B: FusedTttBackend>(
        &self,
        inner_factory: impl Fn(usize) -> InnerModel,
        device: &B::Device,
    ) -> TTTModel<B>;

    /// Initialize a TTT model with the same inner model type for all layers.
    fn init_uniform<B: FusedTttBackend>(
        &self,
        inner_type: InnerModel,
        device: &B::Device,
    ) -> TTTModel<B> {
        self.init_with_inner_by(|_| inner_type, device)
    }

    /// Initialize a TTT model using a [`MixPattern`] to select inner model types per layer.
    fn init_with_mix<B: FusedTttBackend>(
        &self,
        mix: &MixPattern,
        device: &B::Device,
    ) -> TTTModel<B> {
        self.init_with_inner_by(|idx| mix.get(idx), device)
    }
}

impl ModelConfigModelExt for ModelConfig {
    fn init_with_inner_by<B: FusedTttBackend>(
        &self,
        inner_factory: impl Fn(usize) -> InnerModel,
        device: &B::Device,
    ) -> TTTModel<B> {
        let hidden_size = self.arch.hidden_size;
        let vocab_size = self.arch.vocab_size;
        let num_hidden_layers = self.arch.num_hidden_layers;

        let embedding = EmbeddingConfig::new(vocab_size, hidden_size)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        let position_embedding = match self.ttt.pos_encoding {
            PosEncoding::Absolute => Some(
                EmbeddingConfig::new(self.ttt.max_seq_len, hidden_size)
                    .with_initializer(Initializer::Normal {
                        mean: 0.0,
                        std: 0.02,
                    })
                    .init(device),
            ),
            _ => None,
        };

        let layers = (0..num_hidden_layers)
            .map(|idx| {
                let inner_type = inner_factory(idx);
                let inner = AnyInner::from_type(inner_type, self, device);
                TTTBlockConfig::new(self.clone(), idx).init_with_inner(inner, device)
            })
            .collect();
        let norm = RmsNormConfig::new(hidden_size).init(device);

        let lm_head = if self.ttt.tie_word_embeddings {
            None
        } else {
            Some(
                LinearConfig::new(hidden_size, vocab_size)
                    .with_bias(false)
                    .with_initializer(Initializer::Normal {
                        mean: 0.0,
                        std: 0.02,
                    })
                    .init(device),
            )
        };

        TTTModel {
            config: Ignored(self.clone()),
            embedding,
            position_embedding,
            layers,
            norm,
            lm_head,
        }
    }
}

impl<B: FusedTttBackend> TTTModel<B> {
    /// Initialize fresh states for all layers
    pub fn init_states(&self, batch_size: usize) -> Vec<AnyInnerState<B>> {
        self.layers
            .iter()
            .map(|x| x.init_state(batch_size))
            .collect()
    }

    /// Forward pass with fresh states (convenience method that allocates states)
    pub fn forward(&self, input: Tensor<B, 2, Int>, start_idx: usize) -> Tensor<B, 3> {
        let [batch_size, _seq_len] = input.shape().dims();
        let mut states = self.init_states(batch_size);
        self.forward_with_states(input, start_idx, &mut states)
    }

    /// Forward pass with external states (for generation with state persistence)
    pub fn forward_with_states(
        &self,
        input: Tensor<B, 2, Int>,
        start_idx: usize,
        states: &mut [AnyInnerState<B>],
    ) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = input.shape().dims();
        let device = input.device();

        let embedded = self.embedding.forward(input);

        // Add absolute position embeddings if present
        let mut hidden_states = match &self.position_embedding {
            Some(pos_emb) => {
                let positions = Tensor::<B, 1, Int>::arange(
                    start_idx as i64..(start_idx + seq_len) as i64,
                    &device,
                )
                .unsqueeze_dim::<2>(0); // [1, seq_len]
                let pos_embedded = pos_emb.forward(positions); // [1, seq_len, hidden_size]
                embedded + pos_embedded
            }
            None => embedded,
        };

        for (layer, state) in self.layers.iter().zip(states.iter_mut()) {
            hidden_states = layer.forward(hidden_states, state, start_idx);
        }

        hidden_states = self.norm.forward(hidden_states);

        hidden_states = if let Some(lm_head) = &self.lm_head {
            lm_head.forward(hidden_states)
        } else {
            let weight = self.embedding.weight.val();
            let weight = weight.unsqueeze_dim::<3>(0).transpose();
            hidden_states.matmul(weight)
        };

        hidden_states
    }
}
