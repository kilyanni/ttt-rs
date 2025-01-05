//! VRAM estimation for training runs.
//!
//! Conservative formula based on model params + activations:
//! - Model params: embedding + per-layer weights
//! - Optimizer state: 2x model size (Adam moments)
//! - Activations: batch * `seq_len` * hidden * layers * dtype * factor
//! - 20% overhead for framework internals

use ttt_config::{InnerModel, ModelArch};

use crate::config::RunConfig;

/// VRAM estimator for training runs.
#[derive(Debug, Clone)]
pub struct VramEstimator {
    /// Bytes per element (2 for bf16, 4 for fp32).
    pub bytes_per_element: usize,
    /// Default vocab size if not specified.
    pub default_vocab_size: usize,
}

impl Default for VramEstimator {
    fn default() -> Self {
        Self {
            bytes_per_element: 2, // bf16
            default_vocab_size: 50257,
        }
    }
}

/// Estimate TTT inner model weights per head.
fn inner_model_weights(head_dim: usize, inner_type: InnerModel, expansion: usize) -> usize {
    match inner_type {
        // Linear: W (head_dim x head_dim) + b (head_dim)
        InnerModel::Linear
        | InnerModel::LinearAdam
        | InnerModel::FusedNaiveLinear
        | InnerModel::FusedNaiveMultiLinear
        | InnerModel::FusedTileLinear
        | InnerModel::FusedTileMultiLinear
        | InnerModel::D2dStreamingLinear
        | InnerModel::PtrStreamingLinear => head_dim * head_dim + head_dim,
        // MLP with N hidden layers: input + N hidden + output
        // Each hidden layer: inner_dim x inner_dim + bias
        // inner_dim = head_dim * expansion
        InnerModel::Mlp => mlp_weights(head_dim, expansion, 1),
        InnerModel::Mlp2 => mlp_weights(head_dim, expansion, 2),
        InnerModel::Mlp3 => mlp_weights(head_dim, expansion, 3),
        InnerModel::Mlp4 => mlp_weights(head_dim, expansion, 4),
    }
}

/// Calculate MLP inner model weights with N hidden layers.
fn mlp_weights(head_dim: usize, expansion: usize, hidden_layers: usize) -> usize {
    let inner_dim = head_dim * expansion;
    // Input projection: head_dim -> inner_dim
    let input_proj = head_dim * inner_dim + inner_dim;
    // Hidden layers: inner_dim -> inner_dim (N-1 of these, since first hidden is input proj)
    let hidden_proj = if hidden_layers > 1 {
        (hidden_layers - 1) * (inner_dim * inner_dim + inner_dim)
    } else {
        0
    };
    // Output projection: inner_dim -> head_dim
    let output_proj = inner_dim * head_dim + head_dim;

    input_proj + hidden_proj + output_proj
}

/// Estimate total model parameters for VRAM calculation.
fn total_params(arch: &ModelArch, inner_type: InnerModel, mlp_expansion: usize) -> usize {
    // Embedding: vocab_size * hidden_size
    let embedding = arch.vocab_size * arch.hidden_size;

    // Per-layer parameters (TTT layer + SwiGLU MLP):
    // TTT: Q, K, V projections + output projection + convolutions
    let qkv_proj = 3 * arch.hidden_size * arch.hidden_size;
    let out_proj = arch.hidden_size * arch.hidden_size;
    let conv = arch.hidden_size * 4 * 2; // kernel_size=4, Q and K conv

    // TTT inner model weights (per head)
    let head_dim = arch.head_dim();
    let inner_weights = arch.num_heads * inner_model_weights(head_dim, inner_type, mlp_expansion);

    // SwiGLU MLP: gate, up, down projections
    let mlp = arch.hidden_size * arch.intermediate_size * 3;

    // Layer norms
    let layer_norms = arch.hidden_size * 4; // 2 per layer (pre-TTT, pre-MLP), each has weight + bias

    let per_layer = qkv_proj + out_proj + conv + inner_weights + mlp + layer_norms;

    // Final layer norm
    let final_ln = arch.hidden_size * 2;

    embedding + (per_layer * arch.num_hidden_layers) + final_ln
}

/// VRAM estimate breakdown.
#[derive(Debug, Clone)]
pub struct VramEstimate {
    /// Model parameters in bytes.
    pub model_bytes: usize,
    /// Optimizer state in bytes (Adam: 2x model for m and v).
    pub optimizer_bytes: usize,
    /// Gradient storage in bytes.
    pub gradient_bytes: usize,
    /// Activation memory in bytes.
    pub activation_bytes: usize,
    /// Framework overhead in bytes.
    pub overhead_bytes: usize,
    /// Total estimated VRAM in bytes.
    pub total_bytes: usize,
}

impl VramEstimate {
    /// Convert total to GB.
    #[must_use]
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Human-readable breakdown.
    #[must_use]
    pub fn breakdown(&self) -> String {
        format!(
            "Model: {:.2} GB, Optimizer: {:.2} GB, Gradients: {:.2} GB, Activations: {:.2} GB, Overhead: {:.2} GB => Total: {:.2} GB",
            self.model_bytes as f64 / 1e9,
            self.optimizer_bytes as f64 / 1e9,
            self.gradient_bytes as f64 / 1e9,
            self.activation_bytes as f64 / 1e9,
            self.overhead_bytes as f64 / 1e9,
            self.total_gb()
        )
    }
}

impl VramEstimator {
    /// Estimate VRAM usage for a run configuration.
    #[must_use]
    pub fn estimate(&self, run: &RunConfig) -> VramEstimate {
        let arch = ModelArch::from_size(run.params.size, self.default_vocab_size);

        let batch_size = run.params.train.batch;
        let seq_len = run.params.ttt.max_seq_len;
        let grad_accum = run.params.train.grad_accum;

        // Effective batch for activation memory
        let effective_batch = batch_size / grad_accum.max(1);

        let params = total_params(
            &arch,
            run.params.ttt.layer_type.get(0),
            run.params.ttt.mlp_expansion_factor,
        );

        // Model parameters
        let model_bytes = params * self.bytes_per_element;

        // Optimizer state: Adam stores m and v, each same size as params
        // Use fp32 for optimizer state
        let optimizer_bytes = params * 4 * 2;

        // Gradients: same size as model
        let gradient_bytes = model_bytes;

        // Activation memory estimate
        // Key factors: batch * seq_len * hidden * layers * multiplier
        // Multiplier accounts for:
        // - Forward activations stored for backward
        // - Attention patterns (Q @ K^T)
        // - TTT inner model states
        // Conservative estimate: 8x per layer
        let activation_factor = 8;
        let activation_bytes = effective_batch
            * seq_len
            * arch.hidden_size
            * arch.num_hidden_layers
            * self.bytes_per_element
            * activation_factor;

        // Sum before overhead
        let subtotal = model_bytes + optimizer_bytes + gradient_bytes + activation_bytes;

        // 33% framework overhead
        let overhead_bytes = subtotal / 3;

        let total_bytes = subtotal + overhead_bytes;

        VramEstimate {
            model_bytes,
            optimizer_bytes,
            gradient_bytes,
            activation_bytes,
            overhead_bytes,
            total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use ttt_config::{ModelSize, TTTConfig, TrainConfig, TrainParams};

    use super::*;

    #[test]
    fn test_model_arch() {
        let arch = ModelArch::from_size(ModelSize::M60, 50257);
        assert_eq!(arch.hidden_size, 512);
        assert_eq!(arch.num_hidden_layers, 6);
    }

    #[test]
    fn test_inner_model_weights_ordering() {
        let head_dim = 64;
        let expansion = 4;

        let linear = inner_model_weights(head_dim, InnerModel::Linear, expansion);
        let mlp1 = inner_model_weights(head_dim, InnerModel::Mlp, expansion);
        let mlp2 = inner_model_weights(head_dim, InnerModel::Mlp2, expansion);
        let mlp3 = inner_model_weights(head_dim, InnerModel::Mlp3, expansion);
        let mlp4 = inner_model_weights(head_dim, InnerModel::Mlp4, expansion);

        assert!(linear < mlp1, "linear ({linear}) should be < mlp ({mlp1})");
        assert!(mlp1 < mlp2, "mlp ({mlp1}) should be < mlp2 ({mlp2})");
        assert!(mlp2 < mlp3, "mlp2 ({mlp2}) should be < mlp3 ({mlp3})");
        assert!(mlp3 < mlp4, "mlp3 ({mlp3}) should be < mlp4 ({mlp4})");
    }

    #[test]
    fn test_estimate_60m() {
        let estimator = VramEstimator::default();
        let run = RunConfig {
            name: "test".to_string(),
            params: TrainParams {
                size: ModelSize::M60,
                train: TrainConfig {
                    batch: 32,
                    ..Default::default()
                },
                ttt: TTTConfig {
                    max_seq_len: 256,
                    ..Default::default()
                },
                ..Default::default()
            },
            out: None,
        };

        let estimate = estimator.estimate(&run);
        // Sanity check: 60m model shouldn't need more than 20GB
        assert!(estimate.total_gb() < 20.0);
        assert!(estimate.total_gb() > 0.5);
    }

    #[test]
    fn test_estimate_1b() {
        let estimator = VramEstimator::default();
        let run = RunConfig {
            name: "test".to_string(),
            params: TrainParams {
                size: ModelSize::B1,
                train: TrainConfig {
                    batch: 8,
                    ..Default::default()
                },
                ttt: TTTConfig {
                    max_seq_len: 512,
                    ..Default::default()
                },
                ..Default::default()
            },
            out: None,
        };

        let estimate = estimator.estimate(&run);
        // 1B model should need significant VRAM
        assert!(estimate.total_gb() > 10.0);
    }
}
