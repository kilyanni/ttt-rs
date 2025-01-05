//! Tests for the tiled fused TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuAutodiffBackend, GpuBackend, TTTInnerModel, TTTLinearState,
    test_utils::{
        TestDims, generate_test_inputs, test_backward_fmb, test_backward_fwd, test_fmb, test_fwd,
    },
};

use crate::{FusedTile, FusedTileMulti};

// Tolerance constants for this kernel
// Tiled kernel has slightly higher tolerance due to accumulation differences
const RTOL: f32 = 1e-2;
const ATOL: f32 = 1e-3;
const BACKWARD_RTOL: f32 = 5e-2;
const BACKWARD_ATOL: f32 = 1e-3;
// Multi-stage backward accumulates errors across stages
const BACKWARD_MULTI_RTOL: f32 = 1e-1;
const BACKWARD_MULTI_ATOL: f32 = 5e-3;

#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
#[test_case(1, 4, 32, 16 ; "batch1_heads4_dim32_seq16")]
#[test_case(2, 2, 64, 8 ; "batch2_heads2_dim64_seq8")]
fn test_fused_tile_forward_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<GpuBackend, FusedTile<GpuBackend>, TTTLinearState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTile",
    );
}

#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
fn test_fused_tile_backward_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_backward_fmb::<GpuAutodiffBackend, FusedTile<GpuAutodiffBackend>, _>(
        dims,
        |m| m.into(),
        BACKWARD_RTOL,
        BACKWARD_ATOL,
        "FusedTile",
    );
}

#[test_case(2, 2, 32, 8, 4 ; "batch2_heads2_dim32_mini8_stages4")]
#[test_case(1, 4, 32, 8, 2 ; "batch1_heads4_dim32_mini8_stages2")]
fn test_fused_tile_multi_forward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
) {
    let dims = TestDims::multi_stage(batch, heads, dim, mini_batch, stages);
    test_fwd::<GpuBackend, FusedTileMulti<GpuBackend>, TTTLinearState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedTileMulti",
    );
}

#[test_case(2, 2, 32, 8, 1, 1 ; "batch2_heads2_dim32_mini8_stages1_ckpt1")]
#[test_case(2, 2, 32, 8, 2, 1 ; "batch2_heads2_dim32_mini8_stages2_ckpt1")]
#[test_case(2, 2, 32, 8, 4, 2 ; "batch2_heads2_dim32_mini8_stages4_ckpt2")]
#[test_case(2, 2, 32, 8, 4, 4 ; "batch2_heads2_dim32_mini8_stages4_ckpt4")]
fn test_fused_tile_multi_backward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
    ckpt_interval: usize,
) {
    let dims = TestDims::multi_stage(batch, heads, dim, mini_batch, stages)
        .with_checkpoint_interval(ckpt_interval);
    test_backward_fwd::<GpuAutodiffBackend, FusedTileMulti<GpuAutodiffBackend>, _>(
        dims,
        |m| m.into(),
        BACKWARD_MULTI_RTOL,
        BACKWARD_MULTI_ATOL,
        "FusedTileMulti",
    );
}

// =============================================================================
// Compute benchmark (fused-tile-multi, non-streaming baseline)
// =============================================================================

#[test_case(1, 1, 64, 16, 1 ; "1head_seq16_ckpt1")]
#[test_case(1, 12, 64, 16, 1 ; "125m_seq16_ckpt1")]
#[test_case(1, 12, 64, 128, 1 ; "125m_seq128_ckpt1")]
#[test_case(1, 12, 64, 1024, 1 ; "125m_seq1024_ckpt1")]
#[test_case(1, 12, 64, 2048, 1 ; "125m_seq2048_ckpt1")]
#[test_case(1, 12, 64, 128, 4 ; "125m_seq128_ckpt4")]
#[test_case(1, 12, 64, 1024, 8 ; "125m_seq1024_ckpt8")]
#[test_case(1, 12, 64, 2048, 16 ; "125m_seq2048_ckpt16")]
#[ignore]
fn bench_fused_tile_multi(
    batch: usize,
    heads: usize,
    dim: usize,
    seq: usize,
    ckpt_interval: usize,
) {
    use burn::prelude::Backend;

    let device: <GpuBackend as Backend>::Device = Default::default();
    // Use multi_stage with mini_batch_size = seq for single-stage comparison,
    // or a fixed mini_batch for multi-stage.
    // For 125m: 12 heads, 64 dim. Mini-batch = 16 is the standard.
    let mini_batch = 16;
    let stages = seq / mini_batch;
    let dims = TestDims::multi_stage(batch, heads, dim, mini_batch, stages)
        .with_checkpoint_interval(ckpt_interval);
    let config = ttt_core::test_utils::default_test_config(dims);
    let ref_model: ttt_core::TTTLinear<GpuBackend> =
        ttt_core::test_utils::create_test_model(&config, &device);
    let model: FusedTileMulti<GpuBackend> = ref_model.into();

    let warmup = 3;
    let iterations = 20;

    // Pre-generate inputs and state once (clone is cheap - just refcount bump)
    let inputs = generate_test_inputs(dims, &device);
    let base_state = model.init_state(batch);

    // Warmup (includes kernel compilation)
    for _ in 0..warmup {
        let mut state = base_state.clone();
        let _output = model.forward(&mut state, inputs.clone());
        let _ = <GpuBackend as Backend>::sync(&device);
    }

    // Timed iterations: forward + GPU sync (no data transfer, no tensor allocation)
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let mut state = base_state.clone();
        let _output = model.forward(&mut state, inputs.clone());
        let _ = <GpuBackend as Backend>::sync(&device);
    }
    let elapsed = start.elapsed();

    let per_iter_us = elapsed.as_micros() as f64 / iterations as f64;
    let per_stage_us = per_iter_us / stages as f64;
    eprintln!(
        "\n[BENCH fused-tile-multi] batch={}, heads={}, dim={}, seq={} ({}×mini_batch={}), ckpt_interval={}",
        batch, heads, dim, seq, stages, mini_batch, ckpt_interval,
    );
    eprintln!(
        "[BENCH] {} iters, {:.1} us/iter ({:.3} ms/iter), {:.1} us/stage ({:.3} ms/stage), total {:.1} ms",
        iterations,
        per_iter_us,
        per_iter_us / 1000.0,
        per_stage_us,
        per_stage_us / 1000.0,
        elapsed.as_secs_f64() * 1000.0,
    );
}
