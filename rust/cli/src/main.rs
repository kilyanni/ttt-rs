use std::sync::Arc;

use burn::tensor::backend::AutodiffBackend;
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{Shell, generate};
use half::{bf16, f16};
use ttt_cli::{FusedTttBackend, artifact_info, metrics_export};
use ttt_config::{DType, ModelArch, TrainParams};
use ttt_core::{GpuBackend, TrainingBackend, config::ModelConfig};
use ttt_data::{Tokenizer, TokenizerTrait};
use ttt_training::{
    TTTTrainingConfig, eval_pretokenized, generate as inference_generate, interactive,
    train_dataset, train_dataset_pretokenized,
};

/// Load a tokenizer from a HuggingFace model name or local file path.
fn load_tokenizer(identifier: &str) -> Tokenizer {
    Tokenizer::load(identifier, None, None, None)
}

#[derive(Parser)]
#[command(
    name = "ttt",
    about = "TTT text generation model training and inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train on TinyStories dataset
    Train(TrainArgs),
    /// Generate text from prompt
    Generate {
        /// Artifact directory containing the trained model
        artifact_dir: String,
        /// The prompt to generate from
        prompt: String,
        /// Tokenizer: HuggingFace model name (e.g., "gpt2", "EleutherAI/gpt-neox-20b") or local file path
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,
    },
    /// Interactive generation session
    Interactive {
        /// Artifact directory containing the trained model
        artifact_dir: String,
        /// Tokenizer: HuggingFace model name (e.g., "gpt2", "EleutherAI/gpt-neox-20b") or local file path
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,
    },
    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        shell: Shell,
    },
    /// Show information about a training run
    Info {
        /// Artifact directory to inspect
        artifact_dir: String,
        /// Show detailed metrics for each epoch
        #[arg(long, short)]
        verbose: bool,
    },
    /// Evaluate a trained model on validation data
    Eval {
        /// Artifact directory containing the trained model
        artifact_dir: String,
        /// Override max sequence length (default: use training config value).
        /// Use this to test how the model performs at longer contexts.
        #[arg(long)]
        max_seq_len: Option<usize>,
        /// Number of validation samples to evaluate on
        #[arg(long, default_value = "1000")]
        samples: usize,
        /// Batch size for evaluation
        #[arg(long, default_value = "32")]
        batch: usize,
        /// Tokenizer: HuggingFace model name or local file path
        #[arg(long, default_value = "gpt2")]
        tokenizer: String,
        /// Evaluate a specific checkpoint epoch instead of the final model
        #[arg(long)]
        checkpoint: Option<usize>,
    },
    /// Export training metrics to CSV for plotting
    ExportMetrics {
        /// Artifact directories (supports glob patterns like "./runs/*")
        #[arg(required = true)]
        dirs: Vec<String>,

        /// Output CSV base path (creates {name}_train.csv and {name}_valid.csv)
        #[arg(short, long, default_value = "metrics.csv")]
        output: String,

        /// Metrics to export (comma-separated)
        #[arg(short, long, value_delimiter = ',', default_value = "loss,perplexity")]
        metrics: Vec<metrics_export::MetricType>,

        /// Include training metrics
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        train: bool,

        /// Include validation metrics
        #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
        valid: bool,

        /// Downsample to N points per epoch (bucket averaging)
        #[arg(long)]
        target_points: Option<usize>,

        /// Rolling average window size for smoothing
        #[arg(long)]
        window: Option<usize>,
    },
}

#[derive(Parser)]
struct TrainArgs {
    /// Training parameters (model size, batch, lr, etc.)
    #[command(flatten)]
    params: TrainParams,

    /// Output directory
    #[arg(long, default_value = "./artifacts")]
    out: String,

    /// Use pre-tokenized dataset
    #[arg(long, default_value = "true")]
    pretokenized: bool,

    /// Resume training from a checkpoint directory (same as artifact dir)
    #[arg(long)]
    resume: Option<String>,
}

/// Find the latest checkpoint epoch in the given artifact directory.
/// Only returns epochs where all three checkpoint files (model, optim, scheduler) exist.
fn find_latest_checkpoint(artifact_dir: &str) -> Option<usize> {
    let checkpoint_dir = std::path::Path::new(artifact_dir).join("checkpoint");
    std::fs::read_dir(&checkpoint_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.strip_prefix("model-")?
                .strip_suffix(".mpk")?
                .parse::<usize>()
                .ok()
        })
        .filter(|epoch| {
            let has_optim = checkpoint_dir.join(format!("optim-{epoch}.mpk")).exists();
            let has_scheduler = checkpoint_dir
                .join(format!("scheduler-{epoch}.mpk"))
                .exists();
            if !has_optim || !has_scheduler {
                eprintln!(
                    "Warning: checkpoint {epoch} is incomplete (missing optim/scheduler), skipping"
                );
            }
            has_optim && has_scheduler
        })
        .max()
}

impl TrainArgs {
    fn into_config(self, tokenizer: &Tokenizer) -> TTTTrainingConfig {
        let p = self.params;
        let vocab_size = tokenizer.vocab_size();
        let pad_token = tokenizer.pad_token();

        let arch = Arc::new(ModelArch::from_size(p.size, vocab_size));

        let ttt = Arc::new(p.ttt);

        let model_config = ModelConfig::new(arch, ttt);

        TTTTrainingConfig {
            model_config,
            train: p.train,
            pad_token,
            artifact_dir: self.out,
            resume_epoch: None,
            dry_run: p.dry_run,
        }
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train(args) => {
            let pretokenized = args.pretokenized;
            let resume_dir = args.resume.clone();
            let tokenizer_name = args.params.tokenizer.clone();
            let seed = args.params.seed;
            let tokenizer = load_tokenizer(&tokenizer_name);

            // Determine config based on whether we're resuming
            let config = if let Some(ref resume_dir) = resume_dir {
                let config_path = format!("{resume_dir}/config.json");
                let mut config = TTTTrainingConfig::load(&config_path)
                    .unwrap_or_else(|e| panic!("Failed to load config from {config_path}: {e}"));

                // Override epochs if specified
                if args.params.train.epochs != ttt_config::TrainConfig::default().epochs {
                    config.train.epochs = args.params.train.epochs;
                }

                // Override dry_run if specified
                if args.params.dry_run {
                    config.dry_run = true;
                }

                // Find latest checkpoint
                let resume_epoch = find_latest_checkpoint(resume_dir)
                    .unwrap_or_else(|| panic!("No checkpoint found in {resume_dir}/checkpoint/"));

                if resume_epoch >= config.train.epochs {
                    println!(
                        "Training already completed (checkpoint {resume_epoch} >= epochs {}). Nothing to do.",
                        config.train.epochs
                    );
                    println!("To continue training, pass --epochs with a higher value.");
                    return;
                }

                config.resume_epoch = Some(resume_epoch);

                // Keep the original artifact_dir from loaded config
                println!("Resuming training from epoch {resume_epoch}");
                config
            } else {
                args.into_config(&tokenizer)
            };

            let artifact_dir = &config.artifact_dir;

            println!("Training TTT text generation model...");
            println!("Artifacts will be saved to: {artifact_dir}");
            println!(
                "Tokenizer: {tokenizer_name} (vocab_size: {})",
                tokenizer.vocab_size()
            );
            println!("Layer type: {}", config.model_config.ttt.layer_type);
            println!(
                "Model size: {} hidden, {} layers",
                config.model_config.arch.hidden_size, config.model_config.arch.num_hidden_layers
            );
            println!(
                "Batch size: {}, Epochs: {}, LR: {}",
                config.train.batch, config.train.epochs, config.train.lr
            );
            println!(
                "Sequence length: {}, Samples: {}",
                config.model_config.ttt.max_seq_len, config.train.samples
            );
            println!(
                "Pre-tokenized: {} (use --no-pretokenized to disable)",
                pretokenized
            );

            let device = Default::default();

            match config.model_config.ttt.dtype {
                DType::F32 => {
                    println!("Using float32");
                    train::<TrainingBackend<f32>>(
                        pretokenized,
                        tokenizer_name,
                        seed,
                        tokenizer,
                        config,
                        device,
                    );
                }
                DType::F16 => {
                    println!("Using float16");
                    train::<TrainingBackend<f16>>(
                        pretokenized,
                        tokenizer_name,
                        seed,
                        tokenizer,
                        config,
                        device,
                    );
                }
                DType::BF16 => {
                    println!("Using bfloat16");
                    train::<TrainingBackend<bf16>>(
                        pretokenized,
                        tokenizer_name,
                        seed,
                        tokenizer,
                        config,
                        device,
                    );
                }
            }
        }
        Commands::Generate {
            artifact_dir,
            prompt,
            tokenizer,
        } => {
            let device = Default::default();
            let tokenizer = load_tokenizer(&tokenizer);

            match inference_generate::<TrainingBackend>(&artifact_dir, device, &prompt, tokenizer) {
                Ok(generated) => {
                    println!("Prompt: {}", prompt);
                    println!("Generated: {}", generated);
                }
                Err(e) => {
                    eprintln!("Error generating text: {}", e);
                }
            }
        }
        Commands::Interactive {
            artifact_dir,
            tokenizer,
        } => {
            let device = Default::default();
            let tokenizer = load_tokenizer(&tokenizer);

            match interactive::<TrainingBackend>(&artifact_dir, device, tokenizer) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error starting interactive session: {}", e);
                }
            }
        }
        Commands::Eval {
            artifact_dir,
            max_seq_len,
            samples,
            batch,
            tokenizer,
            checkpoint,
        } => {
            let device = Default::default();
            let tokenizer_instance = load_tokenizer(&tokenizer);

            // Load config to determine dtype
            let config = TTTTrainingConfig::load(format!("{artifact_dir}/config.json"))
                .unwrap_or_else(|e| panic!("Failed to load config: {e}"));

            match config.model_config.ttt.dtype {
                DType::F32 => {
                    eval_pretokenized::<GpuBackend<f32>>(
                        &device,
                        &artifact_dir,
                        &tokenizer_instance,
                        &tokenizer,
                        max_seq_len,
                        samples,
                        batch,
                        checkpoint,
                    );
                }
                DType::F16 => {
                    eval_pretokenized::<GpuBackend<f16>>(
                        &device,
                        &artifact_dir,
                        &tokenizer_instance,
                        &tokenizer,
                        max_seq_len,
                        samples,
                        batch,
                        checkpoint,
                    );
                }
                DType::BF16 => {
                    eval_pretokenized::<GpuBackend<bf16>>(
                        &device,
                        &artifact_dir,
                        &tokenizer_instance,
                        &tokenizer,
                        max_seq_len,
                        samples,
                        batch,
                        checkpoint,
                    );
                }
            }
        }
        Commands::Completions { shell } => {
            generate(shell, &mut Cli::command(), "ttt", &mut std::io::stdout());
        }
        Commands::Info {
            artifact_dir,
            verbose,
        } => match artifact_info::ArtifactInfo::load(&artifact_dir) {
            Ok(info) => artifact_info::print_info(&info, verbose),
            Err(e) => {
                eprintln!("Error loading artifact info from {artifact_dir}: {e}");
                std::process::exit(1);
            }
        },
        Commands::ExportMetrics {
            dirs,
            output,
            metrics,
            train,
            valid,
            target_points,
            window,
        } => {
            let config = metrics_export::ExportConfig {
                metrics,
                include_train: train,
                include_valid: valid,
                target_points,
                window,
            };
            if let Err(e) = metrics_export::export_metrics(dirs, &output, config) {
                eprintln!("Error exporting metrics: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn train<B>(
    pretokenized: bool,
    tokenizer_name: String,
    seed: Option<u64>,
    tokenizer: Tokenizer,
    config: TTTTrainingConfig,
    device: B::Device,
) where
    B: FusedTttBackend + AutodiffBackend,
    <B as AutodiffBackend>::InnerBackend: FusedTttBackend,
{
    // Set RNG seed for reproducibility if provided
    if let Some(seed) = seed {
        println!("Using fixed RNG seed: {seed}");
        B::seed(&device, seed);
    }

    if pretokenized {
        train_dataset_pretokenized::<B>(&device, &config, &tokenizer, &tokenizer_name);
    } else {
        train_dataset::<B>(&device, &config);
    }
}
