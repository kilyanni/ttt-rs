use std::sync::Arc;

use burn::{
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::{Dataset, transform::SamplerDataset},
    },
    grad_clipping::GradientClippingConfig,
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::{AutodiffModule, DisplaySettings, ModuleDisplay},
    optim::AdamWConfig,
    prelude::*,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining,
        metric::{AccuracyMetric, LearningRateMetric, LossMetric, PerplexityMetric},
    },
};
use ttt_config::TrainConfig;
use ttt_core::config::ModelConfig;
use ttt_data::{
    TextDataset, TextGenerationBatcher, TokenBatcher, Tokenizer, TokenizerTrait,
    TrainingTextGenerationBatch, load_or_pretokenize,
};
use ttt_fused::FusedTttBackend;

use crate::text_generation::{TTTTextGenerationConfig, TTTTextGenerationModel};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TTTTrainingConfig {
    /// TTT model configuration
    pub model_config: ModelConfig,
    /// Training hyperparameters
    #[serde(default)]
    pub train: TrainConfig,
    /// Pad token ID
    pub pad_token: usize,
    /// Artifact directory for checkpoints
    pub artifact_dir: String,
    /// Resume from epoch (if any)
    #[serde(default)]
    pub resume_epoch: Option<usize>,
    /// If set, don't perform any training, just setup
    #[serde(default)]
    pub dry_run: bool,
}

impl TTTTrainingConfig {
    #[must_use]
    pub fn new(model_config: ModelConfig, pad_token: usize, artifact_dir: String) -> Self {
        Self {
            model_config,
            train: TrainConfig::default(),
            pad_token,
            artifact_dir,
            resume_epoch: None,
            dry_run: false,
        }
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }

    fn adam_config(&self) -> AdamWConfig {
        AdamWConfig::new()
            .with_beta_1(self.train.beta1)
            .with_beta_2(self.train.beta2)
            .with_weight_decay(self.train.weight_decay)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(
                self.train.grad_clip_norm,
            )))
    }
}

/// Common training loop
fn run_training<
    B: AutodiffBackend + FusedTttBackend,
    Item: Clone + Send + Sync + std::fmt::Debug + 'static,
    Bat: Batcher<B, Item, TrainingTextGenerationBatch<B>>
        + Batcher<B::InnerBackend, Item, TrainingTextGenerationBatch<B::InnerBackend>>
        + Clone
        + Send
        + Sync
        + 'static,
    D: Dataset<Item> + 'static,
>(
    model: TTTTextGenerationModel<B>,
    batcher: Bat,
    dataset_train: D,
    dataset_test: D,
    config: &TTTTrainingConfig,
) where
    TTTTextGenerationModel<B>: AutodiffModule<B>,
    <TTTTextGenerationModel<B> as AutodiffModule<B>>::InnerModule: InferenceStep<
            Input = TrainingTextGenerationBatch<B::InnerBackend>,
            Output = ClassificationOutput<B::InnerBackend>,
        >,
{
    let artifact_dir = &config.artifact_dir;

    if config.resume_epoch.is_none() {
        DefaultRecorder::new()
            .record(
                model.clone().into_record(),
                format!("{artifact_dir}/model").into(),
            )
            .unwrap();
    }

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.train.batch)
        .num_workers(config.train.workers)
        .build(SamplerDataset::new(dataset_train, config.train.samples));

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.train.batch)
        .num_workers(config.train.workers)
        .build(SamplerDataset::new(dataset_test, config.train.test_samples));

    let optim = config.adam_config().init();

    let hidden_size = config.model_config.arch.hidden_size;
    let warmup_steps = config.train.warmup_steps;
    let learning_rate = config.train.lr;

    let lr_scheduler = NoamLrSchedulerConfig::new(learning_rate)
        .with_warmup_steps(warmup_steps.max(1))
        .with_model_size(hidden_size)
        .init()
        .expect("Failed to initialize Noam LR scheduler");

    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train(PerplexityMetric::new())
        .metric_valid(PerplexityMetric::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(config.pad_token))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(config.pad_token))
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .grads_accumulation(config.train.grad_accum)
        .num_epochs(config.train.epochs)
        .summary();

    if let Some(epoch) = config.resume_epoch {
        training = training.checkpoint(epoch);
    }

    if config.dry_run {
        println!(
            "Model: {}",
            &model.format(DisplaySettings::new().with_show_num_parameters(true))
        );
        println!("Dry run completed");
        return;
    }

    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    DefaultRecorder::new()
        .record(
            result.model.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}

pub fn train_dataset<B: AutodiffBackend + FusedTttBackend>(
    device: &B::Device,
    config: &TTTTrainingConfig,
) where
    B::InnerBackend: FusedTttBackend,
{
    println!("Loading dataset...");
    let dataset_train = TextDataset::train();
    let dataset_test = TextDataset::test();

    println!("Starting TTT text generation training...");
    let mix = &config.model_config.ttt.layer_type;
    println!("Layer type: {mix}");

    // Batcher needs max_seq_len + 1 to account for next-token prediction
    let batcher = TextGenerationBatcher::new(
        Arc::new(Tokenizer::default()), // TODO: pass tokenizer through config
        config.model_config.ttt.max_seq_len + 1,
    );

    std::fs::create_dir_all(&config.artifact_dir).unwrap();
    config
        .save(format!("{}/config.json", config.artifact_dir))
        .unwrap();

    let model_config = TTTTextGenerationConfig::new(config.model_config.clone(), config.pad_token);
    let model: TTTTextGenerationModel<B> = model_config.init(mix, device);

    run_training(model, batcher, dataset_train, dataset_test, config);

    println!(
        "Training completed! Artifacts saved to: {}",
        config.artifact_dir
    );
}

/// Train using pre-tokenized dataset
///
/// This function will:
/// 1. Check for existing pre-tokenized data in ~/.cache
/// 2. If not found, download and tokenize the dataset
/// 3. Train using memory-mapped pre-tokenized data
pub fn train_dataset_pretokenized<B: AutodiffBackend + FusedTttBackend>(
    device: &B::Device,
    config: &TTTTrainingConfig,
    tokenizer: &Tokenizer,
    tokenizer_name: &str,
) where
    B::InnerBackend: FusedTttBackend,
{
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());

    println!("Loading pre-tokenized datasets...");
    // Load max_seq_len + 1 tokens to account for next-token prediction shift
    // (inputs: 0..max_seq_len, targets: 1..max_seq_len+1)
    let max_seq_len = config.model_config.ttt.max_seq_len;
    let dataset_train = load_or_pretokenize(tokenizer, tokenizer_name, "train", max_seq_len + 1)
        .expect("Failed to load/create pre-tokenized train dataset");
    let dataset_test =
        load_or_pretokenize(tokenizer, tokenizer_name, "validation", max_seq_len + 1)
            .expect("Failed to load/create pre-tokenized test dataset");

    println!(
        "Loaded {} train sequences, {} test sequences",
        dataset_train.len(),
        dataset_test.len()
    );

    println!("Starting TTT text generation training (pre-tokenized)...");
    let mix = &config.model_config.ttt.layer_type;
    println!("Layer type: {mix}");

    let batcher = TokenBatcher::new(config.pad_token, config.model_config.ttt.max_seq_len + 1);

    std::fs::create_dir_all(&config.artifact_dir).unwrap();
    config
        .save(format!("{}/config.json", config.artifact_dir))
        .unwrap();

    let model_config = TTTTextGenerationConfig::new(config.model_config.clone(), config.pad_token);
    let model: TTTTextGenerationModel<B> = model_config.init(mix, device);

    run_training(model, batcher, dataset_train, dataset_test, config);

    println!(
        "Training completed! Artifacts saved to: {}",
        config.artifact_dir
    );
}
