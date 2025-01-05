use std::sync::Arc;

use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{Dataset, transform::SamplerDataset},
    },
    prelude::*,
    record::{DefaultRecorder, Recorder},
};
use ttt_data::{TokenBatcher, TokenizedItem};
use ttt_fused::FusedTttBackend;

use crate::{
    text_generation::{TTTTextGenerationConfig, TTTTextGenerationModel},
    training::TTTTrainingConfig,
};

/// Results from model evaluation.
pub struct EvalResult {
    pub avg_loss: f64,
    pub perplexity: f64,
    pub num_batches: usize,
    pub total_samples: usize,
}

fn eval_inner<B: FusedTttBackend, D: Dataset<TokenizedItem> + 'static>(
    device: &B::Device,
    model_path: &str,
    config: &TTTTrainingConfig,
    dataset: D,
    max_seq_len: usize,
    samples: usize,
    batch_size: usize,
) -> EvalResult {
    // Override max_seq_len in config for model initialization
    let mut model_config = config.model_config.clone();
    let mut ttt_config = (*model_config.ttt).clone();
    ttt_config.max_seq_len = max_seq_len;
    model_config.ttt = Arc::new(ttt_config);

    let mix = &config.model_config.ttt.layer_type;
    let text_gen_config = TTTTextGenerationConfig::new(model_config, config.pad_token);
    let mut model: TTTTextGenerationModel<B> = text_gen_config.init(mix, device);

    // Load trained weights
    let record = DefaultRecorder::new()
        .load(model_path.into(), device)
        .expect("Failed to load model weights");
    model = model.load_record(record);

    // Create batcher and dataloader
    let batcher = TokenBatcher::new(config.pad_token, max_seq_len + 1);
    let num_samples = samples.min(dataset.len());
    let dataset = SamplerDataset::new(dataset, num_samples);
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(2)
        .build(dataset);

    // Evaluate
    let mut total_loss = 0.0f64;
    let mut num_batches = 0usize;

    println!("Running evaluation...");
    for batch in dataloader.iter() {
        let output = model.forward_training(batch);
        let loss_data = output.loss.into_data();
        let loss_val = f64::from(loss_data.convert::<f32>().as_slice::<f32>().unwrap()[0]);
        total_loss += loss_val;
        num_batches += 1;

        if num_batches.is_multiple_of(10) {
            let running_avg = total_loss / num_batches as f64;
            println!(
                "  Batch {num_batches}: loss = {loss_val:.4}, running avg = {running_avg:.4}, ppl = {:.2}",
                running_avg.exp()
            );
        }
    }

    let avg_loss = if num_batches > 0 {
        total_loss / num_batches as f64
    } else {
        0.0
    };
    let perplexity = avg_loss.exp();

    EvalResult {
        avg_loss,
        perplexity,
        num_batches,
        total_samples: num_samples,
    }
}

/// Evaluate a trained model on the validation set using pre-tokenized data.
///
/// If `max_seq_len` is `None`, uses the training config's value.
/// If `checkpoint` is `Some(epoch)`, evaluates that checkpoint instead of the final model.
pub fn eval_pretokenized<B: FusedTttBackend>(
    device: &B::Device,
    artifact_dir: &str,
    tokenizer: &ttt_data::Tokenizer,
    tokenizer_name: &str,
    max_seq_len: Option<usize>,
    samples: usize,
    batch_size: usize,
    checkpoint: Option<usize>,
) {
    let config = TTTTrainingConfig::load(format!("{artifact_dir}/config.json"))
        .unwrap_or_else(|e| panic!("Failed to load config from {artifact_dir}/config.json: {e}"));

    let train_max_seq_len = config.model_config.ttt.max_seq_len;
    let eval_max_seq_len = max_seq_len.unwrap_or(train_max_seq_len);

    let model_path = match checkpoint {
        Some(epoch) => format!("{artifact_dir}/checkpoint/model-{epoch}"),
        None => format!("{artifact_dir}/model"),
    };

    println!("Model trained with max_seq_len = {train_max_seq_len}");
    if eval_max_seq_len == train_max_seq_len {
        println!("Evaluating with max_seq_len = {eval_max_seq_len}");
    } else {
        println!("Evaluating with max_seq_len = {eval_max_seq_len} (overridden)");
        if config.model_config.ttt.pos_encoding == ttt_config::PosEncoding::Absolute {
            eprintln!(
                "Warning: Absolute position encoding may not support sequence lengths \
                 different from training ({train_max_seq_len})."
            );
        }
    }

    if let Some(epoch) = checkpoint {
        println!("Using checkpoint from epoch {epoch}");
    }

    let dataset = ttt_data::load_or_pretokenize(
        tokenizer,
        tokenizer_name,
        "validation",
        eval_max_seq_len + 1,
    )
    .expect("Failed to load/create pre-tokenized validation dataset");

    println!(
        "Loaded {} validation sequences (using {} samples)",
        dataset.len(),
        samples.min(dataset.len()),
    );

    let mix = &config.model_config.ttt.layer_type;
    println!("Layer type: {mix}");

    let result = eval_inner::<B, _>(
        device,
        &model_path,
        &config,
        dataset,
        eval_max_seq_len,
        samples,
        batch_size,
    );

    println!();
    println!("=== Evaluation Results ===");
    println!("  Max sequence length: {eval_max_seq_len}");
    println!("  Batches evaluated:   {}", result.num_batches);
    println!("  Total samples:       {}", result.total_samples);
    println!("  Average loss:        {:.4}", result.avg_loss);
    println!("  Perplexity:          {:.2}", result.perplexity);
}
