#![cfg(feature = "sweep-tests")]
//! Parameter sweep validation tests.
//!
//! These tests run the Python reference validation with various parameter combinations,
//! then run the Rust validation to verify correctness across different configurations.
//!
//! Tests are parameterized using test_case/test_matrix macros and will fail if validation fails.
#![allow(clippy::too_many_arguments)]
use std::{fs, path::PathBuf, process::Command};

use test_case::{test_case, test_matrix};

mod validate_full;

/// Run Python validation data generation with given config, returns success status.
fn generate_validation_data(
    output_dir: String,
    b: usize,
    l: usize,
    h: usize,
    d: usize,
    mini_batch_size: usize,
    seed: usize,
    use_gate: bool,
    conv_kernel: usize,
    pre_conv: bool,
    share_qk: bool,
    tie_word_embeddings: bool,
) -> Result<(), String> {
    fn bool_flag(flag: bool, name: &str) -> String {
        if flag {
            format!("--{}", name)
        } else {
            format!("--no-{}", name)
        }
    }

    let args = vec![
        "-m".to_string(),
        "reference.validate".to_string(),
        "--output_dir".to_string(),
        output_dir.to_string(),
        "--B".to_string(),
        b.to_string(),
        "--L".to_string(),
        l.to_string(),
        "--H".to_string(),
        h.to_string(),
        "--D".to_string(),
        d.to_string(),
        "--mini_batch_size".to_string(),
        mini_batch_size.to_string(),
        "--seed".to_string(),
        seed.to_string(),
        "--conv_kernel".to_string(),
        conv_kernel.to_string(),
        bool_flag(use_gate, "use_gate"),
        bool_flag(pre_conv, "pre_conv"),
        bool_flag(share_qk, "share_qk"),
        bool_flag(tie_word_embeddings, "tie_word_embeddings"),
    ];

    let output = Command::new("python3")
        .args(&args)
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .output()
        .map_err(|e| format!("Failed to run Python: {e}"))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Python validation generation failed: {stderr}"))
    }
}

/// Run Rust validation tests, returns success status with error details on failure.
fn run_rust_validation(dir: String) {
    validate_full::test_all(Some(PathBuf::from(dir)));
}

/// Run a single validation sweep with the given parameters.
fn run_sweep(
    b: usize,
    l: usize,
    h: usize,
    d: usize,
    mini_batch_size: usize,
    seed: usize,
    use_gate: bool,
    conv_kernel: usize,
    pre_conv: bool,
    share_qk: bool,
    tie_word_embeddings: bool,
) {
    let project_root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let rel_dir = format!(
        "validation_data/batch_{b}/layer_{l}/head_{h}/dim_{d}/mini_batch_{mini_batch_size}/seed_{seed}/conv_{conv_kernel}/gate_{use_gate}_pre_conv_{pre_conv}_share_qk_{share_qk}_tie_word_embeddings_{tie_word_embeddings}"
    );
    let abs_dir = project_root.join(&rel_dir);

    fs::create_dir_all(&abs_dir).expect("Failed to create directory");

    generate_validation_data(
        rel_dir.clone(),
        b,
        l,
        h,
        d,
        mini_batch_size,
        seed,
        use_gate,
        conv_kernel,
        pre_conv,
        share_qk,
        tie_word_embeddings,
    )
    .expect("Failed to generate validation data");

    run_rust_validation(abs_dir.to_string_lossy().to_string());
}

#[test_matrix(
    [1, 2, 3, 4],            // batch sizes
    [4],                     // heads (baseline)
    [16],                    // head_dim (baseline)
    [16],                    // seq_len (baseline)
    [16]                     // mini_batch (baseline)
)]
fn sweep_batch_size(b: usize, h: usize, d: usize, l: usize, mini_batch_size: usize) {
    run_sweep(b, l, h, d, mini_batch_size, 42, true, 4, true, false, false);
}

#[test_matrix(
    [2],                     // batch (baseline)
    [2, 3, 4, 5, 8],         // heads
    [16],                    // head_dim (baseline)
    [16],                    // seq_len (baseline)
    [16]                     // mini_batch (baseline)
)]
fn sweep_num_heads(b: usize, h: usize, d: usize, l: usize, mini_batch_size: usize) {
    run_sweep(b, l, h, d, mini_batch_size, 42, true, 4, true, false, false);
}

#[test_matrix(
    [2],                     // batch (baseline)
    [4],                     // heads (baseline)
    [8, 12, 16, 20, 32],     // head_dim
    [16],                    // seq_len (baseline)
    [16]                     // mini_batch (baseline)
)]
fn sweep_head_dim(b: usize, h: usize, d: usize, l: usize, mini_batch_size: usize) {
    run_sweep(b, l, h, d, mini_batch_size, 42, true, 4, true, false, false);
}

// Sequence length with matching mini_batch
#[test_matrix(
    [8, 16, 24, 32]          // seq_len = mini_batch
)]
fn sweep_seq_len(l: usize) {
    run_sweep(2, l, 4, 16, l, 42, true, 4, true, false, false);
}

// Multiple mini-batches (seq_len > mini_batch)
#[test_matrix(
    [32, 48, 64]             // seq_len with mini_batch=16
)]
fn sweep_multi_mini_batch(l: usize) {
    run_sweep(2, l, 4, 16, 16, 42, true, 4, true, false, false);
}

// Feature flag sweep tests

#[test_matrix(
    [true, false],          // use_gate
    [2, 4, 8],              // conv_kernel
    [true, false]           // pre_conv
)]
fn sweep_features(use_gate: bool, conv_kernel: usize, pre_conv: bool) {
    run_sweep(
        2,
        16,
        4,
        16,
        16,
        42,
        use_gate,
        conv_kernel,
        pre_conv,
        false,
        false,
    );
}

#[test_matrix(
    [true, false],          // share_qk
    [true, false]           // tie_word_embeddings
)]
fn sweep_weight_sharing(share_qk: bool, tie_word_embeddings: bool) {
    run_sweep(
        2,
        16,
        4,
        16,
        16,
        42,
        true,
        4,
        true,
        share_qk,
        tie_word_embeddings,
    );
}

#[test_case(2, 16, 5, 8, 16, 42; "h5_d8")]
#[test_case(3, 20, 5, 8, 10, 42; "complex_1")]
#[test_case(2, 24, 6, 12, 12, 42; "complex_2")]
#[test_case(2, 32, 4, 16, 16, 42; "multi_batch_seq32")]
#[test_case(2, 16, 6, 12, 16, 42; "varied_dims")]
fn sweep_combined(b: usize, l: usize, h: usize, d: usize, mini_batch_size: usize, seed: usize) {
    run_sweep(
        b,
        l,
        h,
        d,
        mini_batch_size,
        seed,
        true,
        4,
        true,
        false,
        false,
    );
}

// Weight sharing with varied dimensions (share_qk=false)
#[test_matrix(
    [1, 2, 3],              // batch
    [16, 32],               // seq_len
    [4, 8],                 // heads
    [16, 32],               // head_dim
    [true, false]           // tie_word_embeddings
)]
fn sweep_sharing_with_dimensions(
    b: usize,
    l: usize,
    h: usize,
    d: usize,
    tie_word_embeddings: bool,
) {
    let mini_batch_size = l.min(16);
    run_sweep(
        b,
        l,
        h,
        d,
        mini_batch_size,
        42,
        true,
        4,
        true,
        false,
        tie_word_embeddings,
    );
}

// Weight sharing with varied dimensions (share_qk=true, h=8 d=32 excluded)
#[test_matrix(
    [1, 2, 3],              // batch
    [16, 32],               // seq_len
    [4, 8],                 // heads
    [16],                   // head_dim
    [true, false]           // tie_word_embeddings
)]
fn sweep_sharing_with_qk_d16(
    b: usize,
    l: usize,
    h: usize,
    d: usize,
    tie_word_embeddings: bool,
) {
    let mini_batch_size = l.min(16);
    run_sweep(
        b,
        l,
        h,
        d,
        mini_batch_size,
        42,
        true,
        4,
        true,
        true,
        tie_word_embeddings,
    );
}

#[test_matrix(
    [1, 2, 3],              // batch
    [16, 32],               // seq_len
    [16, 32],               // head_dim
    [true, false]           // tie_word_embeddings
)]
fn sweep_sharing_with_qk_h4(
    b: usize,
    l: usize,
    d: usize,
    tie_word_embeddings: bool,
) {
    let mini_batch_size = l.min(16);
    run_sweep(
        b,
        l,
        4,
        d,
        mini_batch_size,
        42,
        true,
        4,
        true,
        true,
        tie_word_embeddings,
    );
}

// Full feature matrix tests

#[test_matrix(
    [true, false],          // use_gate
    [2, 4],                 // conv_kernel (reduced from [2,4,8] to limit explosion)
    [true, false],          // pre_conv
    [true, false],          // share_qk
    [true, false]           // tie_word_embeddings
)]
fn sweep_full_feature_matrix(
    use_gate: bool,
    conv_kernel: usize,
    pre_conv: bool,
    share_qk: bool,
    tie_word_embeddings: bool,
) {
    run_sweep(
        2,
        16,
        4,
        16,
        16,
        42,
        use_gate,
        conv_kernel,
        pre_conv,
        share_qk,
        tie_word_embeddings,
    );
}
