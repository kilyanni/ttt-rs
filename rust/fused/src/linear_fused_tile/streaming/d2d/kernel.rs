//! Streaming TTT-Linear kernel for persistent GPU execution.
//!
//! This kernel runs persistently on the GPU, receiving mini-batches incrementally
//! via async memory transfers. Weight and bias are kept in shared memory between
//! stages to avoid global memory round-trips.
//!
//! ## Control Protocol
//!
//! Communication between host and kernel uses a single status per cube:
//! - IDLE (0): Kernel is waiting for work
//! - READY (1): Host has provided input, kernel should process
//! - DONE (2): Kernel has finished, output is ready
//! - SHUTDOWN (3): Host signals kernel to exit
//!
//! ## Batch Iteration
//!
//! To avoid workgroup starvation on AMD GPUs, we launch fewer cubes (num_heads)
//! and each cube iterates through all batches for its assigned head.

#![allow(non_camel_case_types, non_snake_case)]

use cubecl::prelude::*;
use thundercube::{prelude::*, util::index_2d};

use super::super::super::{
    forward::{Inputs, Outputs, fused_ttt_forward_stage},
    helpers::ParamsTrait,
};
use crate::FusedTttConfig;

/// Inject HIP code for system-level memory fence.
/// This ensures all preceding memory operations are visible to other streams and the host.
#[cube]
fn memory_fence_system() {
    use cubecl::intrinsic;
    intrinsic!(|scope| {
        scope.register(cubecl::ir::NonSemantic::Comment {
            content: r#"*/
__threadfence_system();
/*"#
            .to_string(),
        });
    });
}

/// Configuration for streaming kernel with debug flag.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct D2dStreamingKernelConfig {
    pub fused: FusedTttConfig,
    pub debug: bool,
}

impl D2dStreamingKernelConfig {
    pub fn new(fused: FusedTttConfig, debug: bool) -> Self {
        Self { fused, debug }
    }
}

/// Control flags - single status per cube (matches PTR protocol)
pub const CTRL_STATUS: usize = 0;
pub const CTRL_ARRAY_SIZE: usize = 1;

pub const STATUS_IDLE: u32 = 0;
pub const STATUS_READY: u32 = 1;
pub const STATUS_DONE: u32 = 2;
pub const STATUS_SHUTDOWN: u32 = 3;

/// Streaming input/output buffers for a single mini-batch.
///
/// These buffers are sized for one mini-batch `[batch, heads, mini_batch_len, head_dim]`
/// and are reused across stages.
#[derive(CubeType, CubeLaunch)]
pub struct D2dStreamingBuffers<F: Float> {
    /// Query input [batch, heads, mini_batch_len, head_dim]
    pub xq: Tensor<Line<F>>,
    /// Key input [batch, heads, mini_batch_len, head_dim]
    pub xk: Tensor<Line<F>>,
    /// Value input [batch, heads, mini_batch_len, head_dim]
    pub xv: Tensor<Line<F>>,
    /// TTT learning rate eta [batch, heads, mini_batch_len]
    pub ttt_lr_eta: Tensor<Line<F>>,
    /// Output [batch, heads, mini_batch_len, head_dim]
    pub output: Tensor<Line<F>>,
    /// Control array for host-kernel communication [batch * heads * CTRL_ARRAY_SIZE]
    pub control: Tensor<Atomic<u32>>,
}

/// Streaming kernel that processes mini-batches incrementally.
///
/// The kernel runs in a persistent loop:
/// 1. Wait for CTRL_READY flag from host
/// 2. Iterate through all batches for this head, processing each mini-batch
/// 3. Set CTRL_DONE flag for host
/// 4. Repeat until CTRL_SHUTDOWN is received
///
/// Weight and bias are kept in shared memory between stages.
/// Each cube handles one head and iterates through all batches internally.
#[cube(launch, launch_unchecked)]
pub fn fused_ttt_d2d_streaming_kernel<P: ParamsTrait>(
    // Input/output streaming buffers
    inputs: &Inputs<P::EVal>,
    outputs: &mut Outputs<P::EVal>,
    // Control array for synchronization - one status per cube (head)
    control: &mut Tensor<Atomic<u32>>,
    #[comptime] config: D2dStreamingKernelConfig,
) {
    // Single cube iterates through all (batch, head) pairs to avoid workgroup starvation
    let batch_count = inputs.xq.shape(0); // [batch, heads, seq, dim]
    let num_heads = inputs.xq.shape(1);
    let epsilon = comptime!(config.fused.epsilon());
    let debug = comptime!(config.debug);

    // Single cube = single control index (CUBE_POS_X is always 0)
    let ctrl_idx = CUBE_POS_X as usize;

    if comptime!(debug) {
        if UNIT_POS == 0 {
            debug_print!("STREAM: kernel start cubes=%u\n", CUBE_COUNT_X);
        }
    }

    // Main processing loop
    loop {
        // Poll for status change (only thread 0)
        // Wait for READY (1) or SHUTDOWN (3), skip IDLE (0) and DONE (2)
        if UNIT_POS == 0 {
            loop {
                // System fence to invalidate cache and see fresh values from host
                memory_fence_system();
                let s = Atomic::load(&control[ctrl_idx]);
                // Only break on READY (1) or SHUTDOWN (3)
                if s == STATUS_READY || s == STATUS_SHUTDOWN {
                    break;
                }
                gpu_sleep(50u32);
            }
        }

        // Broadcast status to all threads
        // All threads need the fence to see host writes, not just thread 0
        sync_cube();
        memory_fence_system();
        let status = Atomic::load(&control[ctrl_idx]);

        if status == STATUS_SHUTDOWN {
            if comptime!(debug) {
                if UNIT_POS == 0 {
                    debug_print!("STREAM: shutdown received ctrl=%u\n", ctrl_idx);
                }
            }
            break;
        }

        // Clear status to IDLE immediately to prevent host from seeing stale DONE
        if UNIT_POS == 0 {
            Atomic::store(&control[ctrl_idx], STATUS_IDLE);
        }
        memory_fence_system();

        // Iterate through all (batch, head) pairs
        for batch_idx in 0..batch_count {
            for head_idx in 0..num_heads {
                // Ensure all threads start iteration together
                sync_cube();

                // Compute base offsets (index_2d returns scalar offsets)
                let base_qkv = index_2d(&inputs.xq, batch_idx, head_idx);
                let base_weight = index_2d(&inputs.weight, batch_idx, head_idx);
                let base_bias = index_2d(&inputs.bias, batch_idx, head_idx);
                let base_eta = index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx);
                let base_ln = index_2d(&inputs.ln_weight, head_idx, 0);

                // Initialize weight in shared memory from inputs.weight for this batch
                let mut weight_smem = P::st_ff();
                cube::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

                sync_cube();

                // Initialize bias in register vector
                let mut bias_rv = P::rvb_f_v();
                cube::broadcast::load_rv_direct(&inputs.bias, &mut bias_rv, base_bias);

                // Load layer norm params for this head
                let mut ln_weight_rv = P::rvb_f_v();
                let mut ln_bias_rv = P::rvb_f_v();
                cube::broadcast::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
                cube::broadcast::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

                // Process the mini-batch
                fused_ttt_forward_stage::<P>(
                    inputs,
                    outputs,
                    &mut weight_smem,
                    &mut bias_rv,
                    &ln_weight_rv,
                    &ln_bias_rv,
                    base_qkv,
                    base_eta,
                    epsilon,
                );

                sync_cube();

                // Store updated weight and bias back to global memory
                let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
                let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);
                cube::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);
                cube::broadcast::store_rv_direct(&bias_rv, &mut outputs.bias_out, base_bias_out);

                // Ensure stores are visible before next iteration
                sync_cube();
                memory_fence_system();
            }
        }

        // Mark as done after processing all (batch, head) pairs
        if UNIT_POS == 0 {
            Atomic::store(&control[ctrl_idx], STATUS_DONE);
        }

        // System fence to ensure DONE is visible to host
        memory_fence_system();

        sync_cube();
    }

    if comptime!(debug) {
        if UNIT_POS == 0 {
            debug_print!("STREAM: kernel exit cubes=%u\n", CUBE_COUNT_X);
        }
    }
}
