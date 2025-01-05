//! Tests for the D2D streaming TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuBackend, TTTInnerModel,
    test_utils::{TestDims, generate_test_inputs, test_fmb, test_fwd},
};

use super::FusedTileD2dStreamingState;
use crate::FusedTileD2dStreaming;

const RTOL: f32 = 1e-2;
const ATOL: f32 = 1e-2;

// =============================================================================
// Forward tests (forward - multi-iteration)
// =============================================================================

#[test_case(2, 2, 32, 8, 2 ; "batch2_heads2_dim32_seq8_iter2")]
#[ignore]
fn test_fused_tile_streaming_forward_vs_reference(
    batch: usize,
    heads: usize,
    dim: usize,
    seq: usize,
    iterations: usize,
) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims = TestDims::new(batch, heads, dim, seq).with_iterations(iterations);
    test_fwd::<
        GpuBackend,
        FusedTileD2dStreaming<GpuBackend>,
        FusedTileD2dStreamingState<GpuBackend>,
        _,
    >(dims, |m| m.into(), RTOL, ATOL, "FusedTileD2dStreaming");
}

// =============================================================================
// Forward tests (forward_mini_batch)
// =============================================================================

#[test_case(1, 1, 64, 16 ; "batch1_heads1_dim64_seq16")]
#[test_case(1, 2, 32, 8 ; "batch1_heads2_dim32_seq8")]
#[test_case(2, 1, 32, 8 ; "batch2_heads1_dim32_seq8")]
#[test_case(2, 2, 32, 8 ; "batch2_heads2_dim32_seq8")]
#[ignore]
fn test_fused_tile_streaming_fmb_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<
        GpuBackend,
        FusedTileD2dStreaming<GpuBackend>,
        FusedTileD2dStreamingState<GpuBackend>,
        _,
    >(dims, |m| m.into(), RTOL, ATOL, "FusedTileD2dStreaming");
}

// =============================================================================
// Compute benchmark
// =============================================================================

// bench_d2d_streaming_compute: measures pure kernel compute for equivalent of total_seq tokens.
// Kernel processes mini_batch=16 tokens per READY→DONE cycle.
// Runs (total_seq / 16) iterations per "forward equivalent".
// Repeats 20 times and averages.
#[test_case(1, 1, 64, 16 ; "1head_seq16")]
#[test_case(1, 12, 64, 16 ; "125m_seq16")]
#[test_case(1, 12, 64, 128 ; "125m_seq128")]
#[test_case(1, 12, 64, 1024 ; "125m_seq1024")]
#[test_case(1, 12, 64, 2048 ; "125m_seq2048")]
#[ignore]
fn bench_d2d_streaming_compute(batch: usize, heads: usize, dim: usize, total_seq: usize) {
    use burn::prelude::Backend;

    use super::host::TttD2dStreamingState;

    #[cfg(feature = "rocm")]
    type R = cubecl::hip::HipRuntime;
    #[cfg(feature = "cuda")]
    type R = cubecl::cuda::CudaRuntime;

    let _guard = crate::linear_fused_tile::STREAMING_TEST_MUTEX
        .lock()
        .unwrap();

    let device: <GpuBackend as Backend>::Device = Default::default();
    let mini_batch = 16;
    // Create with mini_batch=16, do one forward to populate kernel buffers
    let dims = TestDims::new(batch, heads, dim, mini_batch);
    let inputs = generate_test_inputs(dims, &device);

    let config = ttt_core::test_utils::default_test_config(dims);
    let ref_model: ttt_core::TTTLinear<GpuBackend> =
        ttt_core::test_utils::create_test_model(&config, &device);
    let model: FusedTileD2dStreaming<GpuBackend> = ref_model.into();
    let mut state = model.init_state(batch);
    let _output = model.forward(&mut state, inputs);

    // Measure per-stage time with a fixed small iteration count to avoid weight explosion.
    // (Each iteration accumulates weight updates on the same data; too many → NaN → hang.)
    // Then extrapolate: per-forward = per_stage × stages_per_fwd.
    let stages_per_fwd = total_seq / mini_batch;
    let bench_iters = 20;

    let stream_id = state.stream_id();
    super::host::with_d2d_streaming_state::<R, _>(
        stream_id,
        |streaming_state: &TttD2dStreamingState<R>| {
            let us_per_stage = streaming_state.bench_compute(3, bench_iters);
            let us_per_fwd = us_per_stage * stages_per_fwd as f64;
            eprintln!(
                "\n[BENCH d2d-streaming] batch={}, heads={}, dim={}, total_seq={} ({}×mini_batch={})",
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
