//! Pointer-based streaming TTT kernel.
//!
//! Uses raw GPU pointers for zero-copy tensor access between host and persistent kernel.

mod host;
mod kernel;
mod wrapper;

#[cfg(test)]
mod tests;

pub use host::{
    PtrStreamingConfig, PtrStreamingTensors, TttPtrStreamingState,
    get_or_create_ptr_streaming_state, remove_ptr_streaming_state,
    remove_ptr_streaming_state_by_id, shutdown_ptr_streaming_state,
};
pub use kernel::{
    CTRL_ARRAY_SIZE, CTRL_STATUS, PTR_OUTPUT, PTR_TABLE_SIZE, PTR_TTT_LR_ETA, PTR_XK, PTR_XQ,
    PTR_XV, STATUS_DONE, STATUS_IDLE, STATUS_READY, STATUS_SHUTDOWN,
    fused_ttt_streaming_ptr_kernel,
};
pub use wrapper::{
    FusedTilePtrStreamingState, PtrStreamHandle, PtrStreamingKernelConfig, TttPtrStreamingKernel,
    fused_ttt_ptr_streaming_forward,
};
