//! Validation tests for full TTT layer, block, and model against PyTorch reference.
//!
//! Run `python3 -m reference.validate` first to generate validation data.
//!
//! Test organization:
//! - Component tests: ln_fwd, ln_fused_l2_bwd, permute_qk
//! - Inner model test: TTTLinear.ttt() (dual-form computation)
//! - Full layer test: TTT.forward() (projections + conv + RoPE + gate + inner)
//! - Block test: Block.forward() (TTT + FFN + residuals)
//! - Model test: TTTForCausalLM.forward() (embedding + blocks + LM head)

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::{module::Param, prelude::*, tensor::Tensor};
use safetensors::{Dtype, SafeTensors};
use test_case::test_case;
use ttt_config::{ModelArch, TTTConfig};
use ttt_core::{
    GpuBackend, Qkv, TEST_VOCAB_SIZE, TTTInnerModel, TTTInputsInner, TTTLinear, TTTLinearConfig,
    TTTLinearState, TTTMLP, TTTMLPConfig, config::ModelConfig, mlp::TTTMLPState,
    util::MultiHeadLayerNorm,
};
use ttt_fused::FusedTttBackend;
use ttt_layer::{
    AnyInner, InnerModel, ModelConfigExt, ModelConfigModelExt, TTT, TTTBlock, TTTBlockConfig,
    TTTModel,
};

/// Helper to build a ModelConfig for testing
fn build_model_config(
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_layers: usize,
    intermediate_size: usize,
    mini_batch_size: usize,
    conv_kernel_size: usize,
    use_gate: bool,
    conv_before_ttt: bool,
    share_qk: bool,
    tie_word_embeddings: bool,
) -> ModelConfig {
    let arch = Arc::new(ModelArch {
        hidden_size,
        num_hidden_layers: num_layers,
        num_heads,
        intermediate_size,
        vocab_size,
    });
    let ttt = Arc::new(TTTConfig {
        mini_batch_size,
        conv_kernel_size,
        use_gate,
        conv_before_ttt,
        share_qk,
        tie_word_embeddings,
        ..TTTConfig::default()
    });
    ModelConfig::new(arch, ttt)
}

/// Trait for inner models that can be loaded from safetensors for testing
trait TestableInnerModel<B: FusedTttBackend>: TTTInnerModel<B> {
    fn load_from_safetensors(
        loader: &SafeTensorLoader,
        config: &ModelConfig,
        device: &B::Device,
        prefix: &str,
    ) -> Self;

    fn layer_type_name() -> &'static str;

    /// The corresponding `InnerModel` enum variant for this type.
    fn inner_model_type() -> InnerModel;

    /// Wrap this concrete inner model into an `AnyInner` enum variant.
    fn to_any_inner(self) -> AnyInner<B>;

    /// Validation data filename for inner model tests
    fn inner_model_validation_file() -> &'static str;

    /// Load initial state and expected final state from safetensors for inner model testing
    fn load_test_state(loader: &SafeTensorLoader) -> (Self::State, ExpectedState<B>);

    /// Compare actual state against expected state, returns true if all pass
    fn compare_state(actual: &Self::State, expected: &ExpectedState<B>, tolerance: f32) -> bool;

    /// Load and configure inner model from inner model test validation data
    fn load_for_inner_model_test(
        loader: &SafeTensorLoader,
        config: &ModelConfig,
        device: &B::Device,
    ) -> Self;

    fn full_model_tolerance() -> f32 {
        1e-2
    }

    fn inner_model_tolerance() -> f32 {
        2e-3
    }
}

/// Expected state values after forward pass
struct ExpectedState<B: burn::tensor::backend::Backend> {
    tensors: Vec<(String, Tensor<B, 4>)>,
    biases: Vec<(String, Tensor<B, 3>)>,
}

impl TestableInnerModel<GpuBackend> for TTTLinear<GpuBackend> {
    fn load_from_safetensors(
        loader: &SafeTensorLoader,
        config: &ModelConfig,
        device: &<GpuBackend as burn::prelude::Backend>::Device,
        prefix: &str,
    ) -> Self {
        let inner_config = Arc::new(TTTLinearConfig::new());
        let mut inner_model = TTTLinear::new(config, &inner_config, device);

        let w1: Tensor<GpuBackend, 3> = loader.get_tensor(&format!("{prefix}W1"));
        let b1: Tensor<GpuBackend, 3> = loader.get_tensor(&format!("{prefix}b1"));
        let ttt_norm_weight: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{prefix}ttt_norm_weight"));
        let ttt_norm_bias: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{prefix}ttt_norm_bias"));

        inner_model.weight_init = Param::from_tensor(w1);
        inner_model.bias_init = Param::from_tensor(b1.squeeze_dim::<2>(1));
        inner_model.layer_norm = MultiHeadLayerNorm {
            weight: Param::from_tensor(ttt_norm_weight),
            bias: Param::from_tensor(ttt_norm_bias),
            epsilon: 1e-6,
        };

        inner_model
    }

    fn layer_type_name() -> &'static str {
        "linear"
    }

    fn inner_model_type() -> InnerModel {
        InnerModel::Linear
    }

    fn to_any_inner(self) -> AnyInner<GpuBackend> {
        AnyInner::Linear(self)
    }

    fn inner_model_validation_file() -> &'static str {
        "ttt_linear.safetensors"
    }

    fn load_test_state(loader: &SafeTensorLoader) -> (Self::State, ExpectedState<GpuBackend>) {
        let w1_init: Tensor<GpuBackend, 4> = loader.get_tensor("W1_init");
        let b1_init: Tensor<GpuBackend, 3> = loader.get_tensor("b1_init");
        let w1_last_expected: Tensor<GpuBackend, 4> = loader.get_tensor("W1_last_expected");
        let b1_last_expected: Tensor<GpuBackend, 3> = loader.get_tensor("b1_last_expected");

        let state = TTTLinearState {
            weight: w1_init,
            bias: b1_init,
        };

        let expected = ExpectedState {
            tensors: vec![("W1_last".to_string(), w1_last_expected)],
            biases: vec![("b1_last".to_string(), b1_last_expected)],
        };

        (state, expected)
    }

    fn compare_state(
        actual: &Self::State,
        expected: &ExpectedState<GpuBackend>,
        tolerance: f32,
    ) -> bool {
        let w1_passed = compare_tensors(
            &actual.weight,
            &expected.tensors[0].1,
            &expected.tensors[0].0,
            tolerance,
        );
        let b1_passed = compare_tensors_3d(
            &actual.bias,
            &expected.biases[0].1,
            &expected.biases[0].0,
            tolerance,
        );
        w1_passed && b1_passed
    }

    fn load_for_inner_model_test(
        loader: &SafeTensorLoader,
        config: &ModelConfig,
        device: &<GpuBackend as burn::prelude::Backend>::Device,
    ) -> Self {
        let inner_config = Arc::new(TTTLinearConfig::new());
        let mut inner_model = TTTLinear::new(config, &inner_config, device);

        let w1_init: Tensor<GpuBackend, 4> = loader.get_tensor("W1_init");
        let b1_init: Tensor<GpuBackend, 3> = loader.get_tensor("b1_init");
        let ln_weight: Tensor<GpuBackend, 2> = loader.get_tensor("ln_weight");
        let ln_bias: Tensor<GpuBackend, 2> = loader.get_tensor("ln_bias");

        // W1 init is stored as [batch_size, num_heads, head_dim, head_dim], take first batch
        inner_model.weight_init =
            Param::from_tensor(w1_init.slice(s![0..1, .., .., ..]).squeeze_dim::<3>(0));
        inner_model.bias_init =
            Param::from_tensor(b1_init.slice(s![0..1, .., ..]).squeeze_dim::<2>(0));
        inner_model.layer_norm = MultiHeadLayerNorm {
            weight: Param::from_tensor(ln_weight),
            bias: Param::from_tensor(ln_bias),
            epsilon: 1e-6,
        };

        inner_model
    }
}

impl TestableInnerModel<GpuBackend> for TTTMLP<GpuBackend> {
    fn load_from_safetensors(
        loader: &SafeTensorLoader,
        config: &ModelConfig,
        device: &<GpuBackend as burn::prelude::Backend>::Device,
        prefix: &str,
    ) -> Self {
        let inner_config = Arc::new(TTTMLPConfig::new());
        let mut inner_model = TTTMLP::new(config, &inner_config, device);

        let w1: Tensor<GpuBackend, 3> = loader.get_tensor(&format!("{prefix}W1"));
        let b1: Tensor<GpuBackend, 3> = loader.get_tensor(&format!("{prefix}b1"));
        let w2: Tensor<GpuBackend, 3> = loader.get_tensor(&format!("{prefix}W2"));
        let b2: Tensor<GpuBackend, 3> = loader.get_tensor(&format!("{prefix}b2"));
        let ttt_norm_weight: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{prefix}ttt_norm_weight"));
        let ttt_norm_bias: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{prefix}ttt_norm_bias"));

        inner_model.w1_init = Param::from_tensor(w1);
        inner_model.b1_init = Param::from_tensor(b1.squeeze_dim::<2>(1));
        inner_model.w2_init = Param::from_tensor(w2);
        inner_model.b2_init = Param::from_tensor(b2.squeeze_dim::<2>(1));
        inner_model.layer_norm = MultiHeadLayerNorm {
            weight: Param::from_tensor(ttt_norm_weight),
            bias: Param::from_tensor(ttt_norm_bias),
            epsilon: 1e-6,
        };

        inner_model
    }

    fn layer_type_name() -> &'static str {
        "mlp"
    }

    fn inner_model_type() -> InnerModel {
        InnerModel::Mlp
    }

    fn to_any_inner(self) -> AnyInner<GpuBackend> {
        AnyInner::Mlp(self)
    }

    fn inner_model_validation_file() -> &'static str {
        "ttt_mlp.safetensors"
    }

    fn load_test_state(loader: &SafeTensorLoader) -> (Self::State, ExpectedState<GpuBackend>) {
        let w1_init: Tensor<GpuBackend, 4> = loader.get_tensor("W1_init");
        let b1_init: Tensor<GpuBackend, 3> = loader.get_tensor("b1_init");
        let w2_init: Tensor<GpuBackend, 4> = loader.get_tensor("W2_init");
        let b2_init: Tensor<GpuBackend, 3> = loader.get_tensor("b2_init");
        let w1_last_expected: Tensor<GpuBackend, 4> = loader.get_tensor("W1_last_expected");
        let b1_last_expected: Tensor<GpuBackend, 3> = loader.get_tensor("b1_last_expected");
        let w2_last_expected: Tensor<GpuBackend, 4> = loader.get_tensor("W2_last_expected");
        let b2_last_expected: Tensor<GpuBackend, 3> = loader.get_tensor("b2_last_expected");

        let state = TTTMLPState {
            w1: w1_init,
            b1: b1_init,
            w2: w2_init,
            b2: b2_init,
        };

        let expected = ExpectedState {
            tensors: vec![
                ("W1_last".to_string(), w1_last_expected),
                ("W2_last".to_string(), w2_last_expected),
            ],
            biases: vec![
                ("b1_last".to_string(), b1_last_expected),
                ("b2_last".to_string(), b2_last_expected),
            ],
        };

        (state, expected)
    }

    fn compare_state(
        actual: &Self::State,
        expected: &ExpectedState<GpuBackend>,
        tolerance: f32,
    ) -> bool {
        let w1_passed = compare_tensors(
            &actual.w1,
            &expected.tensors[0].1,
            &expected.tensors[0].0,
            tolerance,
        );
        let b1_passed = compare_tensors_3d(
            &actual.b1,
            &expected.biases[0].1,
            &expected.biases[0].0,
            tolerance,
        );
        let w2_passed = compare_tensors(
            &actual.w2,
            &expected.tensors[1].1,
            &expected.tensors[1].0,
            tolerance,
        );
        let b2_passed = compare_tensors_3d(
            &actual.b2,
            &expected.biases[1].1,
            &expected.biases[1].0,
            tolerance,
        );
        w1_passed && b1_passed && w2_passed && b2_passed
    }

    fn load_for_inner_model_test(
        loader: &SafeTensorLoader,
        config: &ModelConfig,
        device: &<GpuBackend as burn::prelude::Backend>::Device,
    ) -> Self {
        let inner_config = Arc::new(TTTMLPConfig::new());
        let mut inner_model = TTTMLP::new(config, &inner_config, device);

        let w1_init: Tensor<GpuBackend, 4> = loader.get_tensor("W1_init");
        let b1_init: Tensor<GpuBackend, 3> = loader.get_tensor("b1_init");
        let w2_init: Tensor<GpuBackend, 4> = loader.get_tensor("W2_init");
        let b2_init: Tensor<GpuBackend, 3> = loader.get_tensor("b2_init");
        let ln_weight: Tensor<GpuBackend, 2> = loader.get_tensor("ln_weight");
        let ln_bias: Tensor<GpuBackend, 2> = loader.get_tensor("ln_bias");

        // W1 init is stored as [batch_size, num_heads, head_dim, 4*head_dim], take first batch
        inner_model.w1_init =
            Param::from_tensor(w1_init.slice(s![0..1, .., .., ..]).squeeze_dim::<3>(0));
        inner_model.b1_init =
            Param::from_tensor(b1_init.slice(s![0..1, .., ..]).squeeze_dim::<2>(0));
        inner_model.w2_init =
            Param::from_tensor(w2_init.slice(s![0..1, .., .., ..]).squeeze_dim::<3>(0));
        inner_model.b2_init =
            Param::from_tensor(b2_init.slice(s![0..1, .., ..]).squeeze_dim::<2>(0));
        inner_model.layer_norm = MultiHeadLayerNorm {
            weight: Param::from_tensor(ln_weight),
            bias: Param::from_tensor(ln_bias),
            epsilon: 1e-6,
        };

        inner_model
    }

    fn full_model_tolerance() -> f32 {
        0.015 // MLP needs slightly higher tolerance than Linear due to more numerical complexity
    }
}

/// A helper struct to load tensors from a safetensors file.
/// Parses the file once on load. Uses Box::leak for simplicity in tests.
struct SafeTensorLoader {
    tensors: SafeTensors<'static>,
}

impl SafeTensorLoader {
    fn load(path: &Path) -> Self {
        let data =
            fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
        // This is only used in tests, so let's not bother with lifetime management.
        let data: &'static [u8] = Box::leak(data.into_boxed_slice());
        let tensors = SafeTensors::deserialize(data).expect("Failed to parse safetensors");
        Self { tensors }
    }

    fn get_tensor_raw<const D: usize, T, F>(
        &self,
        name: &str,
        expected_dtype: Dtype,
        elem_size: usize,
        convert: F,
    ) -> (Vec<T>, [usize; D])
    where
        F: Fn(&[u8]) -> T,
    {
        let tensor_view = self
            .tensors
            .tensor(name)
            .unwrap_or_else(|_| panic!("Tensor '{}' not found", name));

        assert_eq!(
            tensor_view.dtype(),
            expected_dtype,
            "Tensor '{}' has dtype {:?}, expected {:?}",
            name,
            tensor_view.dtype(),
            expected_dtype
        );

        let shape: Vec<usize> = tensor_view.shape().to_vec();
        assert_eq!(
            shape.len(),
            D,
            "Tensor '{}': expected {} dimensions, got {} (shape: {:?})",
            name,
            D,
            shape.len(),
            shape
        );

        let expected_elements: usize = shape.iter().product();
        let actual_elements = tensor_view.data().len() / elem_size;
        assert_eq!(
            actual_elements, expected_elements,
            "Tensor '{}': element count {} doesn't match shape {:?} (expected {})",
            name, actual_elements, shape, expected_elements
        );

        let data: Vec<T> = tensor_view
            .data()
            .chunks_exact(elem_size)
            .map(convert)
            .collect();

        let shape: [usize; D] = shape
            .try_into()
            .expect("Shape dimension mismatch in safetensor");

        (data, shape)
    }

    fn get_tensor<const D: usize>(&self, name: &str) -> Tensor<GpuBackend, D> {
        let (data, shape) = self.get_tensor_raw(name, Dtype::F32, 4, |chunk| {
            f32::from_le_bytes(chunk.try_into().unwrap())
        });
        let device = Default::default();
        Tensor::<GpuBackend, 1>::from_floats(data.as_slice(), &device).reshape(shape)
    }

    fn get_int_tensor<const D: usize>(&self, name: &str) -> Tensor<GpuBackend, D, Int> {
        let (data, shape) = self.get_tensor_raw(name, Dtype::I64, 8, |chunk| {
            i64::from_le_bytes(chunk.try_into().unwrap()) as i32
        });
        let device = Default::default();
        Tensor::<GpuBackend, 1, Int>::from_ints(data.as_slice(), &device).reshape(shape)
    }

    /// Read a scalar i64 config value stored as a 1D tensor
    fn get_config_i64(&self, name: &str) -> i64 {
        let (data, _shape) = self.get_tensor_raw::<1, _, _>(name, Dtype::I64, 8, |chunk| {
            i64::from_le_bytes(chunk.try_into().unwrap())
        });
        data[0]
    }
}

fn compare_tensors<const D: usize>(
    actual: &Tensor<GpuBackend, D>,
    expected: &Tensor<GpuBackend, D>,
    name: &str,
    tolerance: f32,
) -> bool {
    let actual_data = actual.clone().to_data();
    let expected_data = expected.clone().to_data();

    let actual_slice = actual_data.as_slice::<f32>().unwrap();
    let expected_slice = expected_data.as_slice::<f32>().unwrap();

    if actual_slice.len() != expected_slice.len() {
        println!(
            "{}: SIZE MISMATCH {} vs {}",
            name,
            actual_slice.len(),
            expected_slice.len()
        );
        return false;
    }

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;

    for (a, e) in actual_slice.iter().zip(expected_slice.iter()) {
        let diff = (a - e).abs();
        sum_diff += diff;
        max_diff = max_diff.max(diff);
    }

    let mean_diff = sum_diff / actual_slice.len() as f32;
    let passed = max_diff < tolerance;

    println!(
        "  {}: max_diff={:.6}, mean_diff={:.6} {}",
        name,
        max_diff,
        mean_diff,
        if passed { "PASS" } else { "FAIL" }
    );

    if !passed {
        // Print first few mismatches
        for (i, (a, e)) in actual_slice
            .iter()
            .zip(expected_slice.iter())
            .enumerate()
            .take(5)
        {
            let diff = (a - e).abs();
            if diff > tolerance * 0.1 {
                println!(
                    "    [{}]: expected={:.6} got={:.6} diff={:.6}",
                    i, e, a, diff
                );
            }
        }
    }

    passed
}

fn compare_tensors_3d(
    actual: &Tensor<GpuBackend, 3>,
    expected: &Tensor<GpuBackend, 3>,
    name: &str,
    tolerance: f32,
) -> bool {
    compare_tensors(actual, expected, name, tolerance)
}

fn print_header(title: &str) {
    println!("\n========================================");
    println!("{}", title);
    println!("========================================");
}

fn load_validation_data(filename: &str, dir: Option<PathBuf>) -> SafeTensorLoader {
    let data_dir = dir.unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation_data_reference")
    });
    let path = data_dir.join(filename);
    if !path.exists() {
        panic!(
            "Validation data not found at {}. Run `python3 -m reference.validate` first!",
            path.display()
        );
    }
    SafeTensorLoader::load(&path)
}

// Component Tests

/// Test layer norm forward pass (ln_fwd)
#[test_case(None)]
fn test_ln_fwd(dir: Option<PathBuf>) {
    print_header("Layer Norm Forward (ln_fwd) Validation");
    let loader = load_validation_data("ln_fwd.safetensors", dir);

    let x: Tensor<GpuBackend, 4> = loader.get_tensor("ln_fwd_x");
    let gamma: Tensor<GpuBackend, 3> = loader.get_tensor("ln_fwd_gamma");
    let beta: Tensor<GpuBackend, 3> = loader.get_tensor("ln_fwd_beta");
    let result_expected: Tensor<GpuBackend, 4> = loader.get_tensor("ln_fwd_result");

    println!("  x shape: {:?}", x.dims());
    println!("  gamma shape: {:?}", gamma.dims());
    println!("  expected result shape: {:?}", result_expected.dims());

    let [_batch_size, _num_heads, _seq_len, _head_dim] = x.dims();

    let ln = MultiHeadLayerNorm {
        weight: Param::from_tensor(gamma.squeeze_dim::<2>(1)),
        bias: Param::from_tensor(beta.squeeze_dim::<2>(1)),
        epsilon: 1e-6,
    };

    let result_actual = ln.forward(x);

    println!("\n  Comparison:");
    let tolerance = 1e-4;
    let passed = compare_tensors(&result_actual, &result_expected, "ln_fwd", tolerance);

    assert!(passed, "Layer norm forward validation failed!");
}

/// Test fused layer norm + L2 loss backward (ln_fused_l2_bwd)
#[test_case(None)]
fn test_ln_fused_l2_bwd(dir: Option<PathBuf>) {
    print_header("Fused LayerNorm + L2 Backward Validation");
    let loader = load_validation_data("ln_fused.safetensors", dir);

    let x: Tensor<GpuBackend, 4> = loader.get_tensor("ln_fused_x");
    let target: Tensor<GpuBackend, 4> = loader.get_tensor("ln_fused_target");
    let gamma: Tensor<GpuBackend, 3> = loader.get_tensor("ln_fused_gamma");
    let beta: Tensor<GpuBackend, 3> = loader.get_tensor("ln_fused_beta");
    let result_expected: Tensor<GpuBackend, 4> = loader.get_tensor("ln_fused_result");

    println!("  x shape: {:?}", x.dims());
    println!("  target shape: {:?}", target.dims());
    println!("  expected result shape: {:?}", result_expected.dims());

    let [_batch_size, _num_heads, _seq_len, _head_dim] = x.dims();

    let ln = MultiHeadLayerNorm {
        weight: Param::from_tensor(gamma.squeeze_dim::<2>(1)),
        bias: Param::from_tensor(beta.squeeze_dim::<2>(1)),
        epsilon: 1e-6,
    };

    let (_ln_out, grad) = ln.forward_and_l2_grad(x, target);

    println!("\n  Comparison:");
    let tolerance = 1e-4;
    let passed = compare_tensors(&grad, &result_expected, "ln_fused_l2_bwd", tolerance);

    assert!(passed, "Fused LN + L2 backward validation failed!");
}

/// Test Q/K permutation for RoPE alignment
#[test_case(None)]
fn test_permute_qk(dir: Option<PathBuf>) {
    print_header("Permute QK Validation");
    let loader = load_validation_data("permute_qk.safetensors", dir);

    let q_in: Tensor<GpuBackend, 4> = loader.get_tensor("permute_q_in");
    let k_in: Tensor<GpuBackend, 4> = loader.get_tensor("permute_k_in");
    let q_out_expected: Tensor<GpuBackend, 4> = loader.get_tensor("permute_q_out");
    let k_out_expected: Tensor<GpuBackend, 4> = loader.get_tensor("permute_k_out");

    println!("  q_in shape: {:?}", q_in.dims());
    println!("  k_in shape: {:?}", k_in.dims());

    // Apply permute_qk (same function as in layer.rs)
    fn permute_qk<B: burn::tensor::backend::Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, head_dim] = x.shape().dims();
        x.reshape([batch_size, num_heads, seq_len, head_dim / 2, 2])
            .permute([0, 1, 2, 4, 3])
            .reshape([batch_size, num_heads, seq_len, head_dim])
    }

    fn undo_permute_qk<B: burn::tensor::backend::Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, head_dim] = x.shape().dims();
        x.reshape([batch_size, num_heads, seq_len, 2, head_dim / 2])
            .permute([0, 1, 2, 4, 3])
            .reshape([batch_size, num_heads, seq_len, head_dim])
    }

    let q_out_actual = permute_qk(q_in.clone());
    let k_out_actual = permute_qk(k_in.clone());

    println!("\n  Comparison:");
    let tolerance = 1e-6;
    let q_passed = compare_tensors(&q_out_actual, &q_out_expected, "permute_q", tolerance);
    let k_passed = compare_tensors(&k_out_actual, &k_out_expected, "permute_k", tolerance);

    let q_roundtrip = undo_permute_qk(q_out_actual);
    let k_roundtrip = undo_permute_qk(k_out_actual);
    let q_rt_passed = compare_tensors(&q_roundtrip, &q_in, "q_roundtrip", tolerance);
    let k_rt_passed = compare_tensors(&k_roundtrip, &k_in, "k_roundtrip", tolerance);

    let passed = q_passed && k_passed && q_rt_passed && k_rt_passed;
    assert!(passed, "Permute QK validation failed!");
}

/// TTT Inner Model Test - Generic implementation
fn test_inner_model_impl<Inner: TestableInnerModel<GpuBackend>>(dir: Option<PathBuf>) {
    let layer_type = Inner::layer_type_name();
    print_header(&format!("TTT Inner Model ({layer_type}) Validation"));
    let device = Default::default();
    let loader = load_validation_data(Inner::inner_model_validation_file(), dir);

    // Load common inputs
    let xq: Tensor<GpuBackend, 4> = loader.get_tensor("XQ");
    let xk: Tensor<GpuBackend, 4> = loader.get_tensor("XK");
    let xv: Tensor<GpuBackend, 4> = loader.get_tensor("XV");
    let token_eta: Tensor<GpuBackend, 1> = loader.get_tensor("token_eta");
    let ttt_lr_eta: Tensor<GpuBackend, 3> = loader.get_tensor("ttt_lr_eta");
    let output_expected: Tensor<GpuBackend, 4> = loader.get_tensor("output_expected");

    let [_batch_size, num_heads, seq_len, head_dim] = xq.dims();

    println!("  XQ shape: {:?}", xq.dims());
    println!("  token_eta shape: {:?}", token_eta.dims());
    println!("  ttt_lr_eta shape: {:?}", ttt_lr_eta.dims());

    let hidden_size = num_heads * head_dim;
    let config = build_model_config(
        TEST_VOCAB_SIZE,
        hidden_size,
        num_heads,
        1,               // num_layers
        hidden_size * 4, // intermediate_size
        seq_len,         // mini_batch_size
        4,               // conv_kernel_size
        false,           // use_gate
        false,           // conv_before_ttt
        false,           // share_qk
        true,            // tie_word_embeddings
    );

    let inner_model = Inner::load_for_inner_model_test(&loader, &config, &device);
    let (mut state, expected_state) = Inner::load_test_state(&loader);

    let inputs = TTTInputsInner {
        qkv: Qkv { xq, xk, xv },
        token_eta,
        ttt_lr_eta,
        start_idx: 0,
    };

    println!("\n  Running Burn inner model forward...");
    let output_actual = inner_model.forward_mini_batch(&mut state, &inputs, 0..seq_len);

    println!("  Output shape: {:?}", output_actual.dims());

    println!("\n  Comparison:");
    let tolerance = Inner::inner_model_tolerance();

    let out_passed = compare_tensors(&output_actual, &output_expected, "output", tolerance);
    let state_passed = Inner::compare_state(&state, &expected_state, tolerance);

    let passed = out_passed && state_passed;

    if passed {
        println!(
            "\n  TTT {} INNER MODEL VALIDATION PASSED!",
            layer_type.to_uppercase()
        );
    } else {
        println!(
            "\n  TTT {} INNER MODEL VALIDATION FAILED!",
            layer_type.to_uppercase()
        );
    }

    assert!(passed, "TTT {} inner model validation failed!", layer_type);
}

#[test_case(None)]
fn test_ttt_inner_model_linear(dir: Option<PathBuf>) {
    test_inner_model_impl::<TTTLinear<GpuBackend>>(dir);
}

#[test_case(None)]
fn test_ttt_inner_model_mlp(dir: Option<PathBuf>) {
    test_inner_model_impl::<TTTMLP<GpuBackend>>(dir);
}

/// Full Block Test - Generic implementation
fn test_ttt_block_forward_impl<Inner: TestableInnerModel<GpuBackend>>(dir: Option<PathBuf>) {
    let layer_type = Inner::layer_type_name();
    print_header(&format!("Block Forward Validation ({layer_type})"));
    let device = Default::default();
    let loader = load_validation_data(&format!("block_{layer_type}.safetensors"), dir);

    let input: Tensor<GpuBackend, 3> = loader.get_tensor("input");
    let output_expected: Tensor<GpuBackend, 3> = loader.get_tensor("output_expected");

    // Read config from safetensors
    let mini_batch_size = loader.get_config_i64("config_mini_batch_size") as usize;
    let pre_conv = loader.get_config_i64("config_pre_conv") != 0;
    let share_qk = loader.get_config_i64("config_share_qk") != 0;
    let use_gate = loader.get_config_i64("config_use_gate") != 0;

    // Infer dimensions from tensor shapes
    let [batch_size, _seq_len, hidden_size] = input.dims();
    let w1: Tensor<GpuBackend, 3> = loader.get_tensor("W1");
    // With MLP the third dim may be different from head_dim
    // with linear it'd be square
    let [num_heads, head_dim, _] = w1.dims();
    // up_proj_weight is [intermediate_size, hidden_size]
    let up_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("up_proj_weight");
    let [intermediate_size, up_hidden] = up_proj_weight.dims();

    // Verify dimension consistency
    assert_eq!(
        num_heads * head_dim,
        hidden_size,
        "hidden_size ({}) != num_heads ({}) * head_dim ({})",
        hidden_size,
        num_heads,
        head_dim
    );
    assert_eq!(
        up_hidden, hidden_size,
        "up_proj_weight hidden_size ({}) != input hidden_size ({})",
        up_hidden, hidden_size
    );

    println!("  Input shape: {:?}", input.dims());
    println!("  Expected output shape: {:?}", output_expected.dims());
    println!(
        "  Inferred: num_heads={}, head_dim={}, intermediate_size={}, mini_batch_size={}, pre_conv={}, share_qk={}, use_gate={}",
        num_heads, head_dim, intermediate_size, mini_batch_size, pre_conv, share_qk, use_gate
    );

    let seq_norm_weight: Tensor<GpuBackend, 1> = loader.get_tensor("seq_norm_weight");
    let ffn_norm_weight: Tensor<GpuBackend, 1> = loader.get_tensor("ffn_norm_weight");

    let q_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("q_proj_weight");
    let v_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("v_proj_weight");
    let o_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("o_proj_weight");

    let post_norm_weight: Tensor<GpuBackend, 1> = loader.get_tensor("post_norm_weight");
    let post_norm_bias: Tensor<GpuBackend, 1> = loader.get_tensor("post_norm_bias");

    let lr_weight: Tensor<GpuBackend, 3> = loader.get_tensor("lr_weight");
    let lr_bias: Tensor<GpuBackend, 2> = loader.get_tensor("lr_bias");

    let token_idx: Tensor<GpuBackend, 1> = loader.get_tensor("token_idx");
    let learnable_token_idx: Tensor<GpuBackend, 1> = loader.get_tensor("learnable_token_idx");

    // up_proj_weight already loaded above for dimension inference
    let gate_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("gate_proj_weight");
    let down_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("down_proj_weight");

    println!("\n  Parameters loaded:");
    println!("    seq_norm_weight: {:?}", seq_norm_weight.dims());
    println!("    q_proj_weight: {:?}", q_proj_weight.dims());
    println!("    up_proj_weight: {:?}", up_proj_weight.dims());
    println!("    down_proj_weight: {:?}", down_proj_weight.dims());

    // Infer conv_kernel_size from pre_conv weight if present
    let conv_kernel_size = if pre_conv {
        let pre_conv_weight: Tensor<GpuBackend, 3> = loader.get_tensor("pre_conv_weight");
        pre_conv_weight.dims()[2] // [hidden_size, 1, kernel_size]
    } else {
        4
    };

    let config = build_model_config(
        TEST_VOCAB_SIZE,
        hidden_size,
        num_heads,
        1, // num_layers
        intermediate_size,
        mini_batch_size,
        conv_kernel_size,
        use_gate,
        pre_conv, // conv_before_ttt
        share_qk,
        true, // tie_word_embeddings
    );

    let mut block: TTTBlock<GpuBackend> = TTTBlockConfig::new(config.clone(), 0).init(&device);
    let inner_model: AnyInner<GpuBackend> =
        Inner::load_from_safetensors(&loader, &config, &device, "").to_any_inner();

    // Set all weights from loaded data

    // RMS norms
    block.seq_norm.gamma = Param::from_tensor(seq_norm_weight);
    block.ffn_norm.gamma = Param::from_tensor(ffn_norm_weight);

    // TTT layer weights
    block.ttt.q_proj.weight = Param::from_tensor(q_proj_weight.transpose());
    if share_qk {
        // Convolutions only exist when share_qk=true
        let conv_q_weight: Tensor<GpuBackend, 3> = loader.get_tensor("conv_q_weight");
        let conv_q_bias: Tensor<GpuBackend, 1> = loader.get_tensor("conv_q_bias");
        let conv_k_weight: Tensor<GpuBackend, 3> = loader.get_tensor("conv_k_weight");
        let conv_k_bias: Tensor<GpuBackend, 1> = loader.get_tensor("conv_k_bias");
        block.ttt.q_conv.weight = Param::from_tensor(conv_q_weight);
        block.ttt.q_conv.bias = Some(Param::from_tensor(conv_q_bias));
        block.ttt.k_conv.weight = Param::from_tensor(conv_k_weight);
        block.ttt.k_conv.bias = Some(Param::from_tensor(conv_k_bias));
    } else {
        let k_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("k_proj_weight");
        if let Some(ref mut k_proj) = block.ttt.k_proj {
            k_proj.weight = Param::from_tensor(k_proj_weight.transpose());
        }
    }
    block.ttt.v_proj.weight = Param::from_tensor(v_proj_weight.transpose());
    block.ttt.o_proj.weight = Param::from_tensor(o_proj_weight.transpose());
    if use_gate {
        let g_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("g_proj_weight");
        if let Some(ref mut g_proj) = block.ttt.g_proj {
            g_proj.weight = Param::from_tensor(g_proj_weight.transpose());
        }
    }

    // Learning rate parameters
    block.ttt.learnable_ttt_lr_weight = lr_weight.permute([0, 2, 1]);
    block.ttt.learnable_ttt_lr_bias = lr_bias;
    block.ttt.token_idx = token_idx;
    block.ttt.learnable_token_idx = learnable_token_idx;

    // Post norm
    block.ttt.post_norm.gamma = Param::from_tensor(post_norm_weight);
    block.ttt.post_norm.beta = Some(Param::from_tensor(post_norm_bias));

    // PyTorch: gate_proj [intermediate_size, hidden_size], up_proj [intermediate_size, hidden_size]
    // Burn: up_gate_proj [hidden_size, 2*intermediate_size]
    // The forward splits on last dim: [gate, up] = output.split(intermediate_size, 2)
    // So we concatenate gate.T and up.T along dim 1
    let gate_t = gate_proj_weight.transpose(); // [hidden_size, intermediate_size]
    let up_t = up_proj_weight.transpose(); // [hidden_size, intermediate_size]
    let up_gate_combined = Tensor::cat(vec![gate_t, up_t], 1); // [hidden_size, 2*intermediate_size]
    block.swi_glu_mlp.up_gate_proj.weight = Param::from_tensor(up_gate_combined);

    // down_proj
    block.swi_glu_mlp.down_proj.weight = Param::from_tensor(down_proj_weight.transpose());

    // Pre-conv weights if enabled
    if pre_conv && let Some((ref mut conv, ref mut conv_norm)) = block.conv {
        let pre_conv_weight: Tensor<GpuBackend, 3> = loader.get_tensor("pre_conv_weight");
        let pre_conv_bias: Tensor<GpuBackend, 1> = loader.get_tensor("pre_conv_bias");
        let pre_conv_norm_weight: Tensor<GpuBackend, 1> = loader.get_tensor("pre_conv_norm_weight");

        conv.weight = Param::from_tensor(pre_conv_weight);
        conv.bias = Param::from_tensor(pre_conv_bias);
        conv_norm.gamma = Param::from_tensor(pre_conv_norm_weight);
    }

    let mut state = inner_model.init_state(batch_size);

    println!("\n  Running Burn forward pass...");
    let output_actual = block.forward(input, &mut state, &inner_model, 0);

    println!("  Output shape: {:?}", output_actual.dims());

    println!("\n  Comparison:");
    let tolerance = 1e-2;

    let passed = compare_tensors(&output_actual, &output_expected, "output", tolerance);

    if passed {
        println!("\n  BLOCK VALIDATION PASSED!");
    } else {
        println!("\n  BLOCK VALIDATION FAILED!");

        // Print some debug stats
        let actual_data = output_actual.clone().to_data();
        let expected_data = output_expected.clone().to_data();
        let actual_slice = actual_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();

        println!(
            "  Actual stats: mean={:.6}, std={:.6}",
            actual_slice.iter().sum::<f32>() / actual_slice.len() as f32,
            {
                let mean = actual_slice.iter().sum::<f32>() / actual_slice.len() as f32;
                (actual_slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / actual_slice.len() as f32)
                    .sqrt()
            }
        );
        println!(
            "  Expected stats: mean={:.6}, std={:.6}",
            expected_slice.iter().sum::<f32>() / expected_slice.len() as f32,
            {
                let mean = expected_slice.iter().sum::<f32>() / expected_slice.len() as f32;
                (expected_slice
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>()
                    / expected_slice.len() as f32)
                    .sqrt()
            }
        );
    }

    assert!(passed, "Block validation failed!");
}

#[test_case(None)]
fn test_ttt_block_forward_linear(dir: Option<PathBuf>) {
    test_ttt_block_forward_impl::<TTTLinear<GpuBackend>>(dir);
}

#[test_case(None)]
fn test_ttt_block_forward_mlp(dir: Option<PathBuf>) {
    test_ttt_block_forward_impl::<TTTMLP<GpuBackend>>(dir);
}

/// Full TTT Layer Test - Generic implementation
fn test_ttt_layer_forward_impl<Inner: TestableInnerModel<GpuBackend>>(dir: Option<PathBuf>) {
    let layer_type = Inner::layer_type_name();
    print_header(&format!("TTT Layer Forward Validation ({layer_type})"));
    let device = Default::default();
    let loader = load_validation_data(&format!("ttt_layer_{layer_type}.safetensors"), dir);

    let input: Tensor<GpuBackend, 3> = loader.get_tensor("input");
    let output_expected: Tensor<GpuBackend, 3> = loader.get_tensor("output_expected");

    // Read config from safetensors
    let mini_batch_size = loader.get_config_i64("config_mini_batch_size") as usize;
    let share_qk = loader.get_config_i64("config_share_qk") != 0;
    let use_gate = loader.get_config_i64("config_use_gate") != 0;

    // Infer dimensions from tensor shapes
    let [batch_size, _seq_len, hidden_size] = input.dims();
    let w1: Tensor<GpuBackend, 3> = loader.get_tensor("W1");
    let [num_heads, head_dim, _] = w1.dims();

    println!("  Input shape: {:?}", input.dims());
    println!("  Expected output shape: {:?}", output_expected.dims());
    println!(
        "  Inferred: num_heads={}, head_dim={}, mini_batch_size={}, share_qk={}, use_gate={}",
        num_heads, head_dim, mini_batch_size, share_qk, use_gate
    );

    let q_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("q_proj_weight");
    let v_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("v_proj_weight");
    let o_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("o_proj_weight");

    let post_norm_weight: Tensor<GpuBackend, 1> = loader.get_tensor("post_norm_weight");
    let post_norm_bias: Tensor<GpuBackend, 1> = loader.get_tensor("post_norm_bias");

    let lr_weight: Tensor<GpuBackend, 3> = loader.get_tensor("lr_weight");
    let lr_bias: Tensor<GpuBackend, 2> = loader.get_tensor("lr_bias");

    let token_idx: Tensor<GpuBackend, 1> = loader.get_tensor("token_idx");
    let learnable_token_idx: Tensor<GpuBackend, 1> = loader.get_tensor("learnable_token_idx");

    let config = build_model_config(
        TEST_VOCAB_SIZE,
        hidden_size,
        num_heads,
        1,               // num_layers
        hidden_size * 4, // intermediate_size
        mini_batch_size,
        4, // conv_kernel_size
        use_gate,
        false, // conv_before_ttt
        share_qk,
        true, // tie_word_embeddings
    );

    let mut ttt_layer: TTT<GpuBackend> = config.init_ttt_seq(&device);
    let inner_model: AnyInner<GpuBackend> =
        Inner::load_from_safetensors(&loader, &config, &device, "").to_any_inner();

    // Set TTT layer weights
    ttt_layer.q_proj.weight = Param::from_tensor(q_proj_weight.transpose());
    if share_qk {
        // Convolutions only exist when share_qk=true
        let conv_q_weight: Tensor<GpuBackend, 3> = loader.get_tensor("conv_q_weight");
        let conv_q_bias: Tensor<GpuBackend, 1> = loader.get_tensor("conv_q_bias");
        let conv_k_weight: Tensor<GpuBackend, 3> = loader.get_tensor("conv_k_weight");
        let conv_k_bias: Tensor<GpuBackend, 1> = loader.get_tensor("conv_k_bias");
        ttt_layer.q_conv.weight = Param::from_tensor(conv_q_weight);
        ttt_layer.q_conv.bias = Some(Param::from_tensor(conv_q_bias));
        ttt_layer.k_conv.weight = Param::from_tensor(conv_k_weight);
        ttt_layer.k_conv.bias = Some(Param::from_tensor(conv_k_bias));
    } else {
        let k_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("k_proj_weight");
        if let Some(ref mut k_proj) = ttt_layer.k_proj {
            k_proj.weight = Param::from_tensor(k_proj_weight.transpose());
        }
    }
    ttt_layer.v_proj.weight = Param::from_tensor(v_proj_weight.transpose());
    ttt_layer.o_proj.weight = Param::from_tensor(o_proj_weight.transpose());
    if use_gate {
        let g_proj_weight: Tensor<GpuBackend, 2> = loader.get_tensor("g_proj_weight");
        if let Some(ref mut g_proj) = ttt_layer.g_proj {
            g_proj.weight = Param::from_tensor(g_proj_weight.transpose());
        }
    }

    ttt_layer.learnable_ttt_lr_weight = lr_weight.permute([0, 2, 1]);
    ttt_layer.learnable_ttt_lr_bias = lr_bias;
    ttt_layer.token_idx = token_idx;
    ttt_layer.learnable_token_idx = learnable_token_idx;

    ttt_layer.post_norm.gamma = Param::from_tensor(post_norm_weight);
    ttt_layer.post_norm.beta = Some(Param::from_tensor(post_norm_bias));

    let mut state = inner_model.init_state(batch_size);

    println!("\n  Running Burn forward pass...");
    let output_actual = ttt_layer.forward(input, &inner_model, &mut state, 0);

    println!("  Output shape: {:?}", output_actual.dims());

    println!("\n  Comparison:");
    let tolerance = 1e-2;

    let passed = compare_tensors(&output_actual, &output_expected, "output", tolerance);

    if passed {
        println!("\n  TTT LAYER VALIDATION PASSED!");
    } else {
        println!("\n  TTT LAYER VALIDATION FAILED!");
    }

    assert!(passed, "TTT layer validation failed!");
}

#[test_case(None)]
fn test_ttt_layer_forward_linear(dir: Option<PathBuf>) {
    test_ttt_layer_forward_impl::<TTTLinear<GpuBackend>>(dir);
}

#[test_case(None)]
fn test_ttt_layer_forward_mlp(dir: Option<PathBuf>) {
    test_ttt_layer_forward_impl::<TTTMLP<GpuBackend>>(dir);
}

/// Full Model Test - Generic implementation
fn test_full_model_forward_impl<Inner: TestableInnerModel<GpuBackend>>(dir: Option<PathBuf>) {
    use burn::tensor::Int;

    let layer_type = Inner::layer_type_name();
    print_header(&format!("Full Model Forward Validation ({layer_type})"));
    let device = Default::default();
    let loader = load_validation_data(&format!("full_model_{layer_type}.safetensors"), dir);

    let input_ids: Tensor<GpuBackend, 2, Int> = loader.get_int_tensor("input_ids");
    let logits_expected: Tensor<GpuBackend, 3> = loader.get_tensor("logits_expected");

    // Read config from safetensors
    let mini_batch_size = loader.get_config_i64("config_mini_batch_size") as usize;
    let num_layers = loader.get_config_i64("config_num_layers") as usize;
    let pre_conv = loader.get_config_i64("config_pre_conv") != 0;
    let share_qk = loader.get_config_i64("config_share_qk") != 0;
    let tie_word_embeddings = loader.get_config_i64("config_tie_word_embeddings") != 0;
    let use_gate = loader.get_config_i64("config_use_gate") != 0;

    // Infer dimensions from tensor shapes
    let embed_weight: Tensor<GpuBackend, 2> = loader.get_tensor("embed_weight");
    let [vocab_size, hidden_size] = embed_weight.dims();
    // layer_0_W1 is [num_heads, head_dim, head_dim]
    let layer_0_w1: Tensor<GpuBackend, 3> = loader.get_tensor("layer_0_W1");
    let [num_heads, head_dim, _] = layer_0_w1.dims();
    // layer_0_up_proj_weight is [intermediate_size, hidden_size]
    let layer_0_up_proj: Tensor<GpuBackend, 2> = loader.get_tensor("layer_0_up_proj_weight");
    let [intermediate_size, up_hidden] = layer_0_up_proj.dims();

    // Verify dimension consistency
    assert_eq!(
        num_heads * head_dim,
        hidden_size,
        "hidden_size ({}) != num_heads ({}) * head_dim ({})",
        hidden_size,
        num_heads,
        head_dim
    );
    assert_eq!(
        up_hidden, hidden_size,
        "up_proj_weight hidden_size ({}) != embed hidden_size ({})",
        up_hidden, hidden_size
    );

    println!("  Input IDs shape: {:?}", input_ids.dims());
    println!("  Expected logits shape: {:?}", logits_expected.dims());
    println!(
        "  Inferred: vocab_size={}, hidden_size={}, num_heads={}, head_dim={}, intermediate_size={}, num_layers={}, mini_batch_size={}, pre_conv={}, share_qk={}, tie_embed={}, use_gate={}",
        vocab_size,
        hidden_size,
        num_heads,
        head_dim,
        intermediate_size,
        num_layers,
        mini_batch_size,
        pre_conv,
        share_qk,
        tie_word_embeddings,
        use_gate
    );

    // embed_weight already loaded above for dimension inference
    let final_norm_weight: Tensor<GpuBackend, 1> = loader.get_tensor("final_norm_weight");

    // Infer conv_kernel_size from pre_conv weight if present
    let conv_kernel_size = if pre_conv {
        let pre_conv_weight: Tensor<GpuBackend, 3> = loader.get_tensor("layer_0_pre_conv_weight");
        pre_conv_weight.dims()[2] // [hidden_size, 1, kernel_size]
    } else {
        4
    };

    println!("\n  Model parameters:");
    println!("    embed_weight: {:?}", embed_weight.dims());
    println!("    final_norm_weight: {:?}", final_norm_weight.dims());

    let config = build_model_config(
        vocab_size,
        hidden_size,
        num_heads,
        num_layers,
        intermediate_size,
        mini_batch_size,
        conv_kernel_size,
        use_gate,
        pre_conv, // conv_before_ttt
        share_qk,
        tie_word_embeddings,
    );

    let mut model: TTTModel<GpuBackend> = config.init_uniform(Inner::inner_model_type(), &device);

    model.embedding.weight = Param::from_tensor(embed_weight);

    model.norm.gamma = Param::from_tensor(final_norm_weight);

    // Load lm_head if not tied
    if !tie_word_embeddings {
        let lm_head_weight: Tensor<GpuBackend, 2> = loader.get_tensor("lm_head_weight");
        if let Some(ref mut lm_head) = model.lm_head {
            lm_head.weight = Param::from_tensor(lm_head_weight.transpose());
        }
    }

    // Load and set parameters for each layer
    for layer_idx in 0..num_layers {
        let prefix = format!("layer_{}_", layer_idx);

        let seq_norm_weight: Tensor<GpuBackend, 1> =
            loader.get_tensor(&format!("{}seq_norm_weight", prefix));
        let ffn_norm_weight: Tensor<GpuBackend, 1> =
            loader.get_tensor(&format!("{}ffn_norm_weight", prefix));

        let q_proj_weight: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{}q_proj_weight", prefix));
        let v_proj_weight: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{}v_proj_weight", prefix));
        let o_proj_weight: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{}o_proj_weight", prefix));

        let post_norm_weight: Tensor<GpuBackend, 1> =
            loader.get_tensor(&format!("{}post_norm_weight", prefix));
        let post_norm_bias: Tensor<GpuBackend, 1> =
            loader.get_tensor(&format!("{}post_norm_bias", prefix));

        let lr_weight: Tensor<GpuBackend, 3> = loader.get_tensor(&format!("{}lr_weight", prefix));
        let lr_bias: Tensor<GpuBackend, 2> = loader.get_tensor(&format!("{}lr_bias", prefix));

        let token_idx: Tensor<GpuBackend, 1> = loader.get_tensor(&format!("{}token_idx", prefix));
        let learnable_token_idx: Tensor<GpuBackend, 1> =
            loader.get_tensor(&format!("{}learnable_token_idx", prefix));

        // Load MLP parameters
        let up_proj_weight: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{}up_proj_weight", prefix));
        let gate_proj_weight: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{}gate_proj_weight", prefix));
        let down_proj_weight: Tensor<GpuBackend, 2> =
            loader.get_tensor(&format!("{}down_proj_weight", prefix));

        // Set block parameters
        let layer = &mut model.layers[layer_idx];
        let block = &mut layer.block;

        // RMS norms
        block.seq_norm.gamma = Param::from_tensor(seq_norm_weight);
        block.ffn_norm.gamma = Param::from_tensor(ffn_norm_weight);

        // TTT layer weights
        block.ttt.q_proj.weight = Param::from_tensor(q_proj_weight.transpose());
        if share_qk {
            // Convolutions only exist when share_qk=true
            let conv_q_weight: Tensor<GpuBackend, 3> =
                loader.get_tensor(&format!("{}conv_q_weight", prefix));
            let conv_q_bias: Tensor<GpuBackend, 1> =
                loader.get_tensor(&format!("{}conv_q_bias", prefix));
            let conv_k_weight: Tensor<GpuBackend, 3> =
                loader.get_tensor(&format!("{}conv_k_weight", prefix));
            let conv_k_bias: Tensor<GpuBackend, 1> =
                loader.get_tensor(&format!("{}conv_k_bias", prefix));
            block.ttt.q_conv.weight = Param::from_tensor(conv_q_weight);
            block.ttt.q_conv.bias = Some(Param::from_tensor(conv_q_bias));
            block.ttt.k_conv.weight = Param::from_tensor(conv_k_weight);
            block.ttt.k_conv.bias = Some(Param::from_tensor(conv_k_bias));
        } else {
            let k_proj_weight: Tensor<GpuBackend, 2> =
                loader.get_tensor(&format!("{}k_proj_weight", prefix));
            if let Some(ref mut k_proj) = block.ttt.k_proj {
                k_proj.weight = Param::from_tensor(k_proj_weight.transpose());
            }
        }
        block.ttt.v_proj.weight = Param::from_tensor(v_proj_weight.transpose());
        block.ttt.o_proj.weight = Param::from_tensor(o_proj_weight.transpose());
        if use_gate {
            let g_proj_weight: Tensor<GpuBackend, 2> =
                loader.get_tensor(&format!("{}g_proj_weight", prefix));
            if let Some(ref mut g_proj) = block.ttt.g_proj {
                g_proj.weight = Param::from_tensor(g_proj_weight.transpose());
            }
        }

        // Learning rate parameters
        block.ttt.learnable_ttt_lr_weight = lr_weight.permute([0, 2, 1]);
        block.ttt.learnable_ttt_lr_bias = lr_bias;
        block.ttt.token_idx = token_idx;
        block.ttt.learnable_token_idx = learnable_token_idx;

        // Post norm
        block.ttt.post_norm.gamma = Param::from_tensor(post_norm_weight);
        block.ttt.post_norm.beta = Some(Param::from_tensor(post_norm_bias));

        // Inner model parameters - load via trait, wrap in AnyInner
        layer.inner =
            Inner::load_from_safetensors(&loader, &config, &device, &prefix).to_any_inner();

        // MLP weights
        let gate_t = gate_proj_weight.transpose();
        let up_t = up_proj_weight.transpose();
        let up_gate_combined = Tensor::cat(vec![gate_t, up_t], 1);
        block.swi_glu_mlp.up_gate_proj.weight = Param::from_tensor(up_gate_combined);
        block.swi_glu_mlp.down_proj.weight = Param::from_tensor(down_proj_weight.transpose());

        // Pre-conv weights if enabled
        if pre_conv && let Some((ref mut conv, ref mut conv_norm)) = block.conv {
            let pre_conv_weight: Tensor<GpuBackend, 3> =
                loader.get_tensor(&format!("{}pre_conv_weight", prefix));
            let pre_conv_bias: Tensor<GpuBackend, 1> =
                loader.get_tensor(&format!("{}pre_conv_bias", prefix));
            let pre_conv_norm_weight: Tensor<GpuBackend, 1> =
                loader.get_tensor(&format!("{}pre_conv_norm_weight", prefix));

            conv.weight = Param::from_tensor(pre_conv_weight);
            conv.bias = Param::from_tensor(pre_conv_bias);
            conv_norm.gamma = Param::from_tensor(pre_conv_norm_weight);
        }
    }

    println!("\n  Running Burn forward pass...");
    let logits_actual = model.forward(input_ids, 0);

    println!("  Output logits shape: {:?}", logits_actual.dims());

    println!("\n  Comparison:");
    let tolerance = Inner::full_model_tolerance();

    let passed = compare_tensors(&logits_actual, &logits_expected, "logits", tolerance);

    if passed {
        println!("\n  FULL MODEL VALIDATION PASSED!");
    } else {
        println!("\n  FULL MODEL VALIDATION FAILED!");

        // Print some debug stats
        let actual_data = logits_actual.clone().to_data();
        let expected_data = logits_expected.clone().to_data();
        let actual_slice = actual_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();

        println!(
            "  Actual stats: mean={:.6}, std={:.6}",
            actual_slice.iter().sum::<f32>() / actual_slice.len() as f32,
            {
                let mean = actual_slice.iter().sum::<f32>() / actual_slice.len() as f32;
                (actual_slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / actual_slice.len() as f32)
                    .sqrt()
            }
        );
        println!(
            "  Expected stats: mean={:.6}, std={:.6}",
            expected_slice.iter().sum::<f32>() / expected_slice.len() as f32,
            {
                let mean = expected_slice.iter().sum::<f32>() / expected_slice.len() as f32;
                (expected_slice
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>()
                    / expected_slice.len() as f32)
                    .sqrt()
            }
        );
    }

    assert!(passed, "Full model validation failed!");
}

#[test_case(None)]
fn test_full_model_forward_linear(dir: Option<PathBuf>) {
    test_full_model_forward_impl::<TTTLinear<GpuBackend>>(dir);
}

#[test_case(None)]
fn test_full_model_forward_mlp(dir: Option<PathBuf>) {
    test_full_model_forward_impl::<TTTMLP<GpuBackend>>(dir);
}

#[allow(dead_code)]
pub fn test_all(dir: Option<PathBuf>) {
    fn header(s: &str) {
        let len = 70 - s.len();
        println!(
            "{} {} {}",
            s,
            "=".repeat(len.div_ceil(2)),
            "=".repeat(len / 2)
        );
    }

    header("Full Model Forward Linear");
    test_full_model_forward_linear(dir.clone());

    header("Full Model Forward MLP");
    test_full_model_forward_mlp(dir.clone());

    header("LayerNorm Fused L2 Backward");
    test_ln_fused_l2_bwd(dir.clone());

    header("LayerNorm Forward");
    test_ln_fwd(dir.clone());

    header("Permute QK");
    test_permute_qk(dir.clone());

    header("TTT Block Forward Linear");
    test_ttt_block_forward_linear(dir.clone());

    header("TTT Block Forward MLP");
    test_ttt_block_forward_mlp(dir.clone());

    header("TTT Inner Model Linear");
    test_ttt_inner_model_linear(dir.clone());

    header("TTT Inner Model MLP");
    test_ttt_inner_model_mlp(dir.clone());

    header("TTT Layer Forward Linear");
    test_ttt_layer_forward_linear(dir.clone());

    header("TTT Layer Forward MLP");
    test_ttt_layer_forward_mlp(dir);
}
