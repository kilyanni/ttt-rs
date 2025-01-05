use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    prelude::*,
    tensor::{Distribution, backend::AutodiffBackend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use ttt_config::MixPattern;
use ttt_core::config::ModelConfig;
use ttt_data::{TokenizerTrait, TrainingTextGenerationBatch};
use ttt_fused::FusedTttBackend;
use ttt_layer::{AnyInnerState, ModelConfigModelExt, TTTModel};

#[derive(Clone, Debug)]
pub struct TTTTextGenerationConfig {
    pub model_config: ModelConfig,
    pub pad_token: usize,
}

#[derive(Module, Debug)]
pub struct TTTTextGenerationModel<B: FusedTttBackend> {
    pub ttt_model: TTTModel<B>,
    pub pad_token: usize,
}

impl TTTTextGenerationConfig {
    #[must_use]
    pub fn new(model_config: ModelConfig, pad_token: usize) -> Self {
        Self {
            model_config,
            pad_token,
        }
    }

    pub fn from_tokenizer(model_config: ModelConfig, tokenizer: &impl TokenizerTrait) -> Self {
        assert_eq!(tokenizer.vocab_size(), model_config.arch.vocab_size);
        Self {
            model_config,
            pad_token: tokenizer.pad_token(),
        }
    }

    /// Initialize with pad token of zero, intended only for use in tests.
    #[must_use]
    pub fn new_testing(model_config: ModelConfig) -> Self {
        Self {
            model_config,
            pad_token: 0,
        }
    }

    /// Initialize using a mix pattern (single type = uniform, multiple = per-layer cycling).
    pub fn init<B: FusedTttBackend>(
        self,
        mix: &MixPattern,
        device: &B::Device,
    ) -> TTTTextGenerationModel<B> {
        let ttt_model = self.model_config.init_with_mix(mix, device);

        TTTTextGenerationModel {
            ttt_model,
            pad_token: self.pad_token,
        }
    }
}

impl<B: FusedTttBackend> TTTTextGenerationModel<B> {
    pub fn forward_training(
        &self,
        item: TrainingTextGenerationBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = self.ttt_model.embedding.weight.val().device();

        let inputs = item.tokens_inputs.to_device(&device);
        let targets = item.targets.to_device(&device);
        let _mask_pad = item.mask_pad.to_device(&device);

        let logits = self.ttt_model.forward(inputs, 0);

        let output_flatten = logits.reshape([
            batch_size * seq_length,
            self.ttt_model.config.0.arch.vocab_size,
        ]);
        let targets_flatten = targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output_flatten.device());
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }

    pub fn forward_inference(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.ttt_model.forward(input, 0)
    }

    /// Forward with external states for generation
    pub fn forward_with_states(
        &self,
        input: Tensor<B, 2, Int>,
        start_idx: usize,
        states: &mut [AnyInnerState<B>],
    ) -> Tensor<B, 3> {
        self.ttt_model.forward_with_states(input, start_idx, states)
    }

    /// Initialize states for generation
    pub fn init_states(&self, batch_size: usize) -> Vec<AnyInnerState<B>> {
        self.ttt_model.init_states(batch_size)
    }

    pub fn generate(
        &self,
        input_tokens: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Tensor<B, 2, Int> {
        let [batch_size, initial_length] = input_tokens.dims();
        let device = input_tokens.device();

        // Initialize TTT states once and persist across generation
        let mut states = self.init_states(batch_size);

        let initial_logits = self.forward_with_states(input_tokens.clone(), 0, &mut states);

        let total_length = initial_length + max_new_tokens;
        let mut generated: Tensor<B, 2, Int> = Tensor::zeros([batch_size, total_length], &device);
        generated = generated.slice_assign([0..batch_size, 0..initial_length], input_tokens);
        let mut current_pos = initial_length;

        let mut last_logits = initial_logits
            .slice(s![.., (initial_length - 1)..initial_length, ..,])
            .squeeze_dim::<2>(1);

        for _ in 0..max_new_tokens {
            let next_token = if temperature <= 0.0 {
                last_logits.clone().argmax(1).squeeze_dim::<1>(1)
            } else {
                let scaled_logits = last_logits.clone() / temperature;

                if let Some(k) = top_k {
                    Self::sample_top_k(scaled_logits, k, &device)
                } else {
                    Self::sample_multinomial(scaled_logits, &device)
                }
            };

            generated = generated.slice_assign(
                [0..batch_size, current_pos..current_pos + 1],
                next_token.clone().unsqueeze_dim(1),
            );
            current_pos += 1;

            let new_token_input = next_token.unsqueeze_dim(1);
            let new_logits =
                self.forward_with_states(new_token_input, current_pos - 1, &mut states);
            last_logits = new_logits.squeeze_dim::<2>(1);
        }

        generated
    }

    /// Sample from a categorical distribution
    fn sample_multinomial(logits: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 1, Int> {
        let [batch_size, vocab_size] = logits.dims();

        // Gumbel-max trick: argmax(logits + Gumbel noise) samples from softmax(logits)
        // Gumbel noise = -log(-log(uniform))
        // TODO: Precompute and store, right now this is more expensive than naive
        let uniform = Tensor::<B, 2>::random(
            [batch_size, vocab_size],
            Distribution::Uniform(1e-10, 1.0),
            device,
        );
        let gumbel_noise = uniform.log().neg().log().neg();

        let perturbed = logits + gumbel_noise;
        perturbed.argmax(1).squeeze_dim::<1>(1)
    }

    /// Sample from top-k tokens
    fn sample_top_k(logits: Tensor<B, 2>, k: usize, device: &B::Device) -> Tensor<B, 1, Int> {
        let [batch_size, vocab_size] = logits.dims();
        let k = k.min(vocab_size);

        let (top_k_values, top_k_indices) = logits.topk_with_indices(k, 1);

        let uniform =
            Tensor::<B, 2>::random([batch_size, k], Distribution::Uniform(1e-10, 1.0), device);
        let gumbel_noise = uniform.log().neg().log().neg();

        let perturbed = top_k_values + gumbel_noise;
        let selected_positions = perturbed.argmax(1); // [batch_size, 1] - position within top-k

        top_k_indices.gather(1, selected_positions).squeeze_dim(1)
    }
}

impl<B: AutodiffBackend + FusedTttBackend> TrainStep for TTTTextGenerationModel<B>
where
    Self: AutodiffModule<B>,
{
    type Input = TrainingTextGenerationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: FusedTttBackend> InferenceStep for TTTTextGenerationModel<B> {
    type Input = TrainingTextGenerationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_training(item)
    }
}
