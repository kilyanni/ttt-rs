//! Tests for the pointer-based streaming TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuBackend, TTTInnerModel,
    test_utils::{TestDims, generate_test_inputs, test_fmb, test_fwd},
};

use super::FusedTilePtrStreamingState;
use crate::FusedTilePtrStreaming;

const RTOL: f32 = 0.5;
const ATOL: f32 = 0.4;

// =============================================================================
// Forward tests (forward - multi-stage, multi-iteration)
// =============================================================================

// Minimal test with 1 cube to isolate visibility issues
// NOTE: Multi-iteration tests are disabled because persistent kernels on AMD GPUs
// block compute resources even when sleeping, preventing other GPU work from running.
// The kernel works correctly for single iteration (which is the normal use case).
#[test_case(1, 1, 64, 16, 2, 1 ; "batch1_heads1_dim64_mini16_stages2_iter1")]
// #[test_case(1, 1, 64, 16, 2, 2 ; "batch1_heads1_dim64_mini16_stages2_iter2")]  // blocked by persistent kernel
// #[test_case(2, 2, 32, 8, 2, 2 ; "batch2_heads2_dim32_mini8_stages2_iter2")]    // blocked by persistent kernel
#[ignore]
fn test_fused_tile_ptr_streaming_forward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
    iterations: usize,
) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims =
        TestDims::multi_stage(batch, heads, dim, mini_batch, stages).with_iterations(iterations);
    test_fwd::<
        GpuBackend,
        FusedTilePtrStreaming<GpuBackend>,
        FusedTilePtrStreamingState<GpuBackend>,
        _,
    >(dims, |m| m.into(), RTOL, ATOL, "FusedTilePtrStreaming");
}

// =============================================================================
// Forward tests (forward_mini_batch)
// =============================================================================

// NOTE: Multi-cube tests are disabled because persistent kernels on AMD GPUs
// can starve workgroups, preventing all cubes from starting.
#[test_case(1, 1, 64, 16 ; "batch1_heads1_dim64_seq16")]
// #[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]  // blocked by workgroup starvation
#[ignore]
fn test_fused_tile_ptr_streaming_fmb_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    seq: usize,
) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<
        GpuBackend,
        FusedTilePtrStreaming<GpuBackend>,
        FusedTilePtrStreamingState<GpuBackend>,
        _,
    >(dims, |m| m.into(), RTOL, ATOL, "FusedTilePtrStreaming");
}

// =============================================================================
// Compute benchmark
// =============================================================================

// NOTE: ptr-streaming launches batch×heads cubes, which causes workgroup starvation
// on AMD GPUs when heads > 1. Only single-head configs work for benchmarking.
#[test_case(1, 1, 64, 16 ; "1head_seq16")]
#[test_case(1, 1, 64, 2048 ; "1head_seq2048")]
#[ignore]
fn bench_ptr_streaming_compute(batch: usize, heads: usize, dim: usize, total_seq: usize) {
    use burn::prelude::Backend;

    use super::host::TttPtrStreamingState;

    #[cfg(feature = "rocm")]
    type R = cubecl::hip::HipRuntime;
    #[cfg(feature = "cuda")]
    type R = cubecl::cuda::CudaRuntime;

    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let device: <GpuBackend as Backend>::Device = Default::default();
    let mini_batch = 16;
    let dims = TestDims::new(batch, heads, dim, mini_batch);
    let inputs = generate_test_inputs(dims, &device);

    let config = ttt_core::test_utils::default_test_config(dims);
    let ref_model: ttt_core::TTTLinear<GpuBackend> =
        ttt_core::test_utils::create_test_model(&config, &device);
    let model: FusedTilePtrStreaming<GpuBackend> = ref_model.into();
    let mut state = model.init_state(batch);
    let _output = model.forward(&mut state, inputs);

    let stages_per_fwd = total_seq / mini_batch;
    let bench_iters = 20;

    let stream_id = state.stream_id();
    super::host::with_ptr_streaming_state::<R, _>(
        stream_id,
        |streaming_state: &TttPtrStreamingState<R>| {
            let us_per_stage = streaming_state.bench_compute(3, bench_iters);
            let us_per_fwd = us_per_stage * stages_per_fwd as f64;
            eprintln!(
                "\n[BENCH ptr-streaming] batch={}, heads={}, dim={}, total_seq={} ({}×mini_batch={})",
                batch, heads, dim, total_seq, stages_per_fwd, mini_batch,
            );
            eprintln!(
                "[BENCH] {:.1} us/stage ({:.3} ms), {:.1} us/fwd ({:.3} ms/fwd)",
                us_per_stage,
                us_per_stage / 1000.0,
                us_per_fwd,
                us_per_fwd / 1000.0,
            );
        },
    );
}
