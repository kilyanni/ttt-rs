//! Training and TTT layer configuration.

use serde::{Deserialize, Serialize};

use crate::{DType, MixPattern, ModelArch, ModelSize, PosEncoding};

/// TTT layer behavioral configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct TTTConfig {
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, default_value = "linear"))]
    pub layer_type: MixPattern,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, default_value = "rope"))]
    pub pos_encoding: PosEncoding,
    #[serde(default = "default_base_lr")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1.0"))]
    pub base_lr: f32,
    #[serde(default = "default_mini_batch_size")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "16"))]
    pub mini_batch_size: usize,
    #[serde(default = "default_max_seq_len")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "2048"))]
    pub max_seq_len: usize,
    #[serde(default = "default_mlp_expansion")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "4"))]
    pub mlp_expansion_factor: usize,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long))]
    pub threads: Option<usize>,
    #[serde(default = "default_rope_theta")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "10000.0", hide = true))]
    pub rope_theta: f32,
    #[serde(default = "default_conv_kernel")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "4", hide = true))]
    pub conv_kernel_size: usize,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, hide = true))]
    pub conv_before_ttt: bool,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, hide = true))]
    pub use_gate: bool,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, hide = true))]
    pub share_qk: bool,
    #[serde(default = "default_true")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "true", hide = true))]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_epsilon")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1e-6", hide = true))]
    pub epsilon: f64,
    #[serde(default = "default_dtype")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "f32"))]
    pub dtype: DType,
    /// Number of stages between weight checkpoints in multi-stage backward.
    /// Higher values reduce memory but increase backward compute (re-simulation).
    /// Default: 1 (checkpoint every stage).
    #[serde(default = "default_checkpoint_interval")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1"))]
    pub checkpoint_interval: usize,
    /// Inner Adam learning rate (only used by linear-adam layers).
    #[serde(default = "default_adam_lr")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.001"))]
    pub adam_lr: f32,
    /// Inner Adam β₁ (only used by linear-adam layers).
    #[serde(default = "default_adam_beta1")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.9"))]
    pub adam_beta1: f32,
    /// Inner Adam β₂ (only used by linear-adam layers).
    #[serde(default = "default_adam_beta2")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.999"))]
    pub adam_beta2: f32,
}

fn default_base_lr() -> f32 {
    1.0
}
fn default_mini_batch_size() -> usize {
    16
}
fn default_max_seq_len() -> usize {
    2048
}
fn default_mlp_expansion() -> usize {
    4
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_conv_kernel() -> usize {
    4
}
fn default_true() -> bool {
    true
}
fn default_epsilon() -> f64 {
    1e-6
}
fn default_dtype() -> DType {
    DType::F32
}
fn default_checkpoint_interval() -> usize {
    1
}
fn default_adam_lr() -> f32 {
    0.001
}
fn default_adam_beta1() -> f32 {
    0.9
}
fn default_adam_beta2() -> f32 {
    0.999
}

impl Default for TTTConfig {
    fn default() -> Self {
        Self {
            layer_type: MixPattern::default(),
            pos_encoding: PosEncoding::default(),
            base_lr: default_base_lr(),
            mini_batch_size: default_mini_batch_size(),
            max_seq_len: default_max_seq_len(),
            mlp_expansion_factor: default_mlp_expansion(),
            threads: None,
            rope_theta: default_rope_theta(),
            conv_kernel_size: default_conv_kernel(),
            conv_before_ttt: false,
            use_gate: false,
            share_qk: false,
            tie_word_embeddings: true,
            epsilon: default_epsilon(),
            dtype: default_dtype(),
            checkpoint_interval: default_checkpoint_interval(),
            adam_lr: default_adam_lr(),
            adam_beta1: default_adam_beta1(),
            adam_beta2: default_adam_beta2(),
        }
    }
}

/// Training hyperparameters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "burn", derive(burn::config::Config))]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct TrainConfig {
    #[serde(default = "default_batch")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "32"))]
    pub batch: usize,
    #[serde(default = "default_epochs")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "10"))]
    pub epochs: usize,
    #[serde(default = "default_lr")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "2e-3"))]
    pub lr: f64,
    #[serde(default = "default_samples")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "10000"))]
    pub samples: usize,
    #[serde(default = "default_test_samples")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1000"))]
    pub test_samples: usize,
    #[serde(default = "default_grad_accum")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1"))]
    pub grad_accum: usize,
    #[serde(default = "default_workers")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "2"))]
    pub workers: usize,
    #[serde(default = "default_warmup_steps")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "5000"))]
    pub warmup_steps: usize,
    // Adam optimizer parameters
    #[serde(default = "default_beta1")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.9", hide = true))]
    pub beta1: f32,
    #[serde(default = "default_beta2")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.999", hide = true))]
    pub beta2: f32,
    #[serde(default = "default_weight_decay")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "0.01", hide = true))]
    pub weight_decay: f32,
    #[serde(default = "default_grad_clip_norm")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "1.0", hide = true))]
    pub grad_clip_norm: f32,
}

fn default_batch() -> usize {
    32
}
fn default_epochs() -> usize {
    10
}
fn default_lr() -> f64 {
    2e-3
}
fn default_samples() -> usize {
    10000
}
fn default_test_samples() -> usize {
    1000
}
fn default_grad_accum() -> usize {
    1
}
fn default_workers() -> usize {
    2
}
fn default_warmup_steps() -> usize {
    5000
}
fn default_beta1() -> f32 {
    0.9
}
fn default_beta2() -> f32 {
    0.999
}
fn default_weight_decay() -> f32 {
    0.01
}
fn default_grad_clip_norm() -> f32 {
    1.0
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch: default_batch(),
            epochs: default_epochs(),
            lr: default_lr(),
            samples: default_samples(),
            test_samples: default_test_samples(),
            grad_accum: default_grad_accum(),
            workers: default_workers(),
            warmup_steps: default_warmup_steps(),
            beta1: default_beta1(),
            beta2: default_beta2(),
            weight_decay: default_weight_decay(),
            grad_clip_norm: default_grad_clip_norm(),
        }
    }
}

impl TrainConfig {
    pub fn samples_per_epoch(&self) -> usize {
        self.samples + self.test_samples
    }

    pub fn total_samples(&self) -> usize {
        self.samples_per_epoch() * self.epochs
    }
}

/// Full training parameters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct TrainParams {
    #[serde(default = "default_tokenizer")]
    #[cfg_attr(feature = "clap", arg(long, default_value = "gpt2"))]
    pub tokenizer: String,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long, default_value = "60m"))]
    pub size: ModelSize,
    #[serde(default, flatten)]
    #[cfg_attr(feature = "clap", command(flatten))]
    pub ttt: TTTConfig,
    #[serde(default, flatten)]
    #[cfg_attr(feature = "clap", command(flatten))]
    pub train: TrainConfig,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long))]
    pub seed: Option<u64>,
    #[serde(default)]
    #[cfg_attr(feature = "clap", arg(long))]
    pub dry_run: bool,
}

fn default_tokenizer() -> String {
    "gpt2".into()
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            tokenizer: default_tokenizer(),
            size: ModelSize::default(),
            ttt: TTTConfig::default(),
            train: TrainConfig::default(),
            seed: None,
            dry_run: false,
        }
    }
}

impl TrainParams {
    /// Get model architecture from size preset + vocab_size.
    pub fn arch(&self, vocab_size: usize) -> ModelArch {
        ModelArch::from_size(self.size, vocab_size)
    }

    /// Convert to CLI arguments for subprocess invocation.
    pub fn to_cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "--tokenizer".into(),
            self.tokenizer.clone(),
            "--size".into(),
            self.size.to_string(),
            // TTT config
            "--layer-type".into(),
            self.ttt.layer_type.to_string(),
            "--pos-encoding".into(),
            self.ttt.pos_encoding.to_string(),
            "--base-lr".into(),
            self.ttt.base_lr.to_string(),
            "--mini-batch-size".into(),
            self.ttt.mini_batch_size.to_string(),
            "--max-seq-len".into(),
            self.ttt.max_seq_len.to_string(),
            "--mlp-expansion-factor".into(),
            self.ttt.mlp_expansion_factor.to_string(),
            "--dtype".into(),
            self.ttt.dtype.to_string(),
            "--adam-lr".into(),
            self.ttt.adam_lr.to_string(),
            "--adam-beta1".into(),
            self.ttt.adam_beta1.to_string(),
            "--adam-beta2".into(),
            self.ttt.adam_beta2.to_string(),
            // Train config
            "--batch".into(),
            self.train.batch.to_string(),
            "--epochs".into(),
            self.train.epochs.to_string(),
            "--lr".into(),
            self.train.lr.to_string(),
            "--samples".into(),
            self.train.samples.to_string(),
            "--test-samples".into(),
            self.train.test_samples.to_string(),
            "--grad-accum".into(),
            self.train.grad_accum.to_string(),
            "--workers".into(),
            self.train.workers.to_string(),
            "--warmup-steps".into(),
            self.train.warmup_steps.to_string(),
        ];
        if let Some(threads) = self.ttt.threads {
            args.extend(["--threads".into(), threads.to_string()]);
        }
        if let Some(seed) = self.seed {
            args.extend(["--seed".into(), seed.to_string()]);
        }
        if self.dry_run {
            args.push("--dry-run".into());
        }
        args
    }
}
