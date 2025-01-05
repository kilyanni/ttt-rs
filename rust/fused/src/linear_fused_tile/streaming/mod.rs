//! Streaming TTT kernels for persistent GPU execution.
//!
//! This module contains two streaming implementations:
//! - `d2d`: Device-to-device memory copy based streaming
//! - `ptr`: Raw GPU pointer based streaming (zero-copy)

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique stream IDs for streaming state registries.
/// Shared between D2D and PTR to avoid any collision risk.
static STREAM_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique stream ID for a new streaming session.
pub fn next_stream_id() -> u64 {
    STREAM_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub mod d2d;
pub mod ptr;

// Re-export commonly used items
pub use d2d::{
    CTRL_ARRAY_SIZE, CTRL_STATUS, D2dStreamingConfig, D2dStreamingKernelConfig,
    FusedTileD2dStreamingState, STATUS_DONE, STATUS_IDLE, STATUS_READY, STATUS_SHUTDOWN,
    TttD2dStreamingKernel, TttD2dStreamingState, fused_ttt_d2d_streaming_kernel,
    get_or_create_d2d_streaming_state,
};
pub use ptr::{
    CTRL_ARRAY_SIZE as PTR_CTRL_ARRAY_SIZE, FusedTilePtrStreamingState, PTR_TABLE_SIZE,
    PtrStreamingConfig, PtrStreamingKernelConfig, PtrStreamingTensors, TttPtrStreamingKernel,
    TttPtrStreamingState, fused_ttt_streaming_ptr_kernel,
};
