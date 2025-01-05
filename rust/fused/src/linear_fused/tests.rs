//! Tests for the naive fused TTT-Linear kernel.

use test_case::test_case;
use ttt_core::{
    GpuAutodiffBackend, GpuBackend, TTTLinearState,
    test_utils::{TestDims, test_backward_fmb, test_backward_fwd, test_fmb, test_fwd},
};

use crate::{FusedNaive, FusedNaiveMulti};

// Tolerance constants for this kernel
const RTOL: f32 = 1e-3;
const ATOL: f32 = 1e-4;
const BACKWARD_RTOL: f32 = 2e-2;
const BACKWARD_ATOL: f32 = 1e-3;

// =============================================================================
// Forward tests (forward_mini_batch)
// =============================================================================

#[test_case(2, 4, 16, 8 ; "batch2_heads4_dim16_seq8")]
#[test_case(1, 2, 8, 4 ; "batch1_heads2_dim8_seq4")]
#[test_case(4, 8, 32, 16 ; "batch4_heads8_dim32_seq16")]
fn test_fused_linear_forward_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_fmb::<GpuBackend, FusedNaive<GpuBackend>, TTTLinearState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedNaive",
    );
}

// =============================================================================
// Backward tests
// =============================================================================

#[test_case(2, 2, 8, 4 ; "batch2_heads2_dim8_seq4")]
fn test_fused_linear_backward_vs_reference(batch: usize, heads: usize, dim: usize, seq: usize) {
    let dims = TestDims::new(batch, heads, dim, seq);
    test_backward_fmb::<GpuAutodiffBackend, FusedNaive<GpuAutodiffBackend>, _>(
        dims,
        |m| m.into(),
        BACKWARD_RTOL,
        BACKWARD_ATOL,
        "FusedNaive",
    );
}

// =============================================================================
// Multi-stage forward tests (using forward() which dispatches to multi kernel)
// =============================================================================

#[test_case(2, 2, 8, 4, 2 ; "batch2_heads2_dim8_mini4_stages2")]
#[test_case(1, 2, 8, 4, 4 ; "batch1_heads2_dim8_mini4_stages4")]
fn test_fused_linear_multi_forward(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
) {
    let dims = TestDims::multi_stage(batch, heads, dim, mini_batch, stages);
    test_fwd::<GpuBackend, FusedNaiveMulti<GpuBackend>, TTTLinearState<GpuBackend>, _>(
        dims,
        |m| m.into(),
        RTOL,
        ATOL,
        "FusedNaiveMulti",
    );
}

// =============================================================================
// Multi-stage backward tests
// =============================================================================

#[test_case(2, 2, 8, 4, 1, 1 ; "batch2_heads2_dim8_mini4_stages1_ckpt1")]
#[test_case(2, 2, 8, 4, 2, 1 ; "batch2_heads2_dim8_mini4_stages2_ckpt1")]
#[test_case(2, 2, 8, 4, 2, 2 ; "batch2_heads2_dim8_mini4_stages2_ckpt2")]
fn test_fused_linear_multi_backward(
    batch: usize,
    heads: usize,
    dim: usize,
    mini_batch: usize,
    stages: usize,
    ckpt_interval: usize,
) {
    let dims = TestDims::multi_stage(batch, heads, dim, mini_batch, stages)
        .with_checkpoint_interval(ckpt_interval);
    test_backward_fwd::<GpuAutodiffBackend, FusedNaiveMulti<GpuAutodiffBackend>, _>(
        dims,
        |m| m.into(),
        BACKWARD_RTOL,
        BACKWARD_ATOL,
        "FusedNaiveMulti",
    );
}
