//! Streaming TTT-Linear kernel with pointer indirection for zero-copy input.
//!
//! This kernel uses a pointer table to access input tensors directly without
//! any memory copies. The host writes tensor addresses to the pointer table,
//! and the kernel dereferences them directly.
//!
//! ## Pointer Table Layout
//! - Slot 0: xq address
//! - Slot 1: xk address
//! - Slot 2: xv address
//! - Slot 3: ttt_lr_eta address
//! - Slot 4: output address (for writing)
//!
//! ## Buffer Indices (kernel parameters)
//! - buffer_0: ptr_table (u64 addresses)
//! - buffer_1: control (atomic u32)
//! - buffer_2: weight
//! - buffer_3: bias
//! - buffer_4: token_eta
//! - buffer_5: ln_weight
//! - buffer_6: ln_bias
//! - buffer_7: weight_out
//! - buffer_8: bias_out
//! - buffer_9: xq_buf (Array parameter)
//! - buffer_10: xk_buf (Array parameter)
//! - buffer_11: xv_buf (Array parameter)
//! - buffer_12: eta_buf (Array parameter)

#![allow(non_camel_case_types, non_snake_case)]

use cubecl::prelude::*;
use thundercube::{prelude::*, util::index_2d};

use super::super::super::{
    forward::{Inputs, Outputs, fused_ttt_forward_stage},
    helpers::ParamsTrait,
};
use crate::FusedTttConfig;

/// Pointer table slot indices
pub const PTR_XQ: usize = 0;
pub const PTR_XK: usize = 1;
pub const PTR_XV: usize = 2;
pub const PTR_TTT_LR_ETA: usize = 3;
pub const PTR_OUTPUT: usize = 4;
pub const PTR_TABLE_SIZE: usize = 5;

/// Control flags
pub const CTRL_STATUS: usize = 0;
pub const CTRL_ARRAY_SIZE: usize = 1;

pub const STATUS_IDLE: u32 = 0;
pub const STATUS_READY: u32 = 1;
pub const STATUS_DONE: u32 = 2;
pub const STATUS_SHUTDOWN: u32 = 3;

/// Buffer indices for injected HIP code (must match kernel parameter order)
pub const BUF_PTR_TABLE: usize = 0;
pub const _BUF_CONTROL: usize = 1;
pub const BUF_XQ: usize = 2;
pub const BUF_XK: usize = 3;
pub const BUF_XV: usize = 4;
pub const BUF_ETA: usize = 5;

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

/// Inject HIP code to load from pointer table into Array parameters.
#[cube]
#[allow(unused_variables)]
fn load_from_pointers<P: ParamsTrait>(
    xq_buf: &mut Array<Line<P::EVal>>,
    xk_buf: &mut Array<Line<P::EVal>>,
    xv_buf: &mut Array<Line<P::EVal>>,
    eta_buf: &mut Array<Line<P::EVal>>,
    #[comptime] qkv_count: usize,
    #[comptime] eta_count: usize,
) {
    use cubecl::intrinsic;
    intrinsic!(|scope| {
        scope.register(cubecl::ir::NonSemantic::Comment {
            content: format!(
                r#"*/
// Per-cube offset for Array buffers (each cube has its own region)
const uint32 cube_idx = blockIdx.x * gridDim.y + blockIdx.y;
const uint32 qkv_off = cube_idx * {qkv}u;
const uint32 eta_off = cube_idx * {eta}u;
// Load xq from pointer table slot 0 into buffer_9
{{
    const float_4* xq_src = (const float_4*)((const uint64*)buffer_{buf_ptr})[{ptr_xq}];
    for (uint32 i = 0; i < {qkv}u; i++) {{ buffer_{buf_xq}[qkv_off + i] = xq_src[qkv_off + i]; }}
}}
// Load xk from pointer table slot 1 into buffer_10
{{
    const float_4* xk_src = (const float_4*)((const uint64*)buffer_{buf_ptr})[{ptr_xk}];
    for (uint32 i = 0; i < {qkv}u; i++) {{ buffer_{buf_xk}[qkv_off + i] = xk_src[qkv_off + i]; }}
}}
// Load xv from pointer table slot 2 into buffer_11
{{
    const float_4* xv_src = (const float_4*)((const uint64*)buffer_{buf_ptr})[{ptr_xv}];
    for (uint32 i = 0; i < {qkv}u; i++) {{ buffer_{buf_xv}[qkv_off + i] = xv_src[qkv_off + i]; }}
}}
// Load ttt_lr_eta from pointer table slot 3 into buffer_12
{{
    const float_4* eta_src = (const float_4*)((const uint64*)buffer_{buf_ptr})[{ptr_eta}];
    for (uint32 i = 0; i < {eta}u; i++) {{ buffer_{buf_eta}[eta_off + i] = eta_src[eta_off + i]; }}
}}
/*"#,
                buf_ptr = BUF_PTR_TABLE,
                buf_xq = BUF_XQ,
                buf_xk = BUF_XK,
                buf_xv = BUF_XV,
                buf_eta = BUF_ETA,
                ptr_xq = PTR_XQ,
                ptr_xk = PTR_XK,
                ptr_xv = PTR_XV,
                ptr_eta = PTR_TTT_LR_ETA,
                qkv = qkv_count,
                eta = eta_count,
            ),
        });
    });
}

/// Inject HIP code to store from xq_buf to output pointer.
#[cube]
#[allow(unused_variables)]
fn store_to_output<P: ParamsTrait>(xq_buf: &Array<Line<P::EVal>>, #[comptime] count: usize) {
    use cubecl::intrinsic;
    intrinsic!(|scope| {
        scope.register(cubecl::ir::NonSemantic::Comment {
            content: format!(
                r#"*/
// Store from buffer_9 to output via pointer table slot 4 (per-cube offsets)
{{
    float_4* out_dst = (float_4*)((const uint64*)buffer_{buf_ptr})[{ptr_out}];
    const uint32 off = (blockIdx.x * gridDim.y + blockIdx.y) * {count}u;
    for (uint32 i = 0; i < {count}u; i++) {{ out_dst[off + i] = buffer_{buf_xq}[off + i]; }}
}}
/*"#,
                buf_ptr = BUF_PTR_TABLE,
                buf_xq = BUF_XQ,
                ptr_out = PTR_OUTPUT,
                count = count,
            ),
        });
    });
}

/// Cooperatively copy from Array to Tensor.
/// All threads participate in the copy.
#[cube]
fn copy_array_to_tensor<F: Float>(
    src: &Array<Line<F>>,
    src_offset: usize,
    dst: &mut Tensor<Line<F>>,
    dst_offset: usize,
    #[comptime] count: usize,
) {
    for i in range_stepped(UNIT_POS as usize, count, CUBE_DIM as usize) {
        dst[dst_offset + i] = src[src_offset + i];
    }
}

/// Cooperatively copy from Tensor to Array.
/// All threads participate in the copy.
#[cube]
fn copy_tensor_to_array<F: Float>(
    src: &Tensor<Line<F>>,
    src_offset: usize,
    dst: &mut Array<Line<F>>,
    dst_offset: usize,
    #[comptime] count: usize,
) {
    for i in range_stepped(UNIT_POS as usize, count, CUBE_DIM as usize) {
        dst[dst_offset + i] = src[src_offset + i];
    }
}

/// Streaming kernel with pointer indirection.
///
/// The kernel receives data via pointer table,
/// copies to scratch Tensors, then calls the standard TTT forward stage.
///
/// Arrays (buffer_9-12) receive data via injected HIP code that dereferences
/// addresses from ptr_table. After sync, we copy Array -> scratch Tensor
/// so the standard forward stage can operate on Tensors.
/// Essentially we're just hoping that hipcc will be smart enough
/// to optimize away the Array -> Tensor copies,
/// as we can't easily trick CubeCL any other way.
#[cube(launch, launch_unchecked)]
#[allow(unused_assignments, reason = "False positive on `status`")]
pub fn fused_ttt_streaming_ptr_kernel<P: ParamsTrait>(
    // Pointer table: addresses of input tensors [PTR_TABLE_SIZE]
    // not read by CubeCL, but our injected HIP code reads it.
    // rustc can't see that we're using it, so we give it the underscore prefix.
    _ptr_table: &Tensor<u64>,
    // Control array [batch * heads]
    control: &mut Tensor<Atomic<u32>>,
    // Array buffers for pointer-based loading (get predictable buffer_N names)
    xq_buf: &mut Array<Line<P::EVal>>,
    xk_buf: &mut Array<Line<P::EVal>>,
    xv_buf: &mut Array<Line<P::EVal>>,
    eta_buf: &mut Array<Line<P::EVal>>,
    // Inputs struct (scratch tensors for xq/xk/xv/ttt_lr_eta, constants for others)
    inputs: &mut Inputs<P::EVal>,
    // Outputs struct (output tensor, weight_out, bias_out)
    outputs: &mut Outputs<P::EVal>,
    #[comptime] config: FusedTttConfig,
    #[comptime] debug: bool,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let num_heads = CUBE_COUNT_Y as usize;
    let epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(config.mini_batch_len);
    let head_dim = comptime!(config.head_dim);

    // Control index for this cube
    let ctrl_idx = batch_idx * num_heads + head_idx;

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!(
                "PTR_STREAM: kernel start ctrl_idx=%u cubes=(%u,%u)\n",
                ctrl_idx,
                CUBE_COUNT_X,
                CUBE_COUNT_Y
            );
        }
        // Print startup for all cubes to see which are running
        if UNIT_POS == 0 {
            debug_print!(
                "PTR_STREAM: cube start b=%u h=%u\n",
                batch_idx as u32,
                head_idx as u32
            );
        }
    }

    // Sizes in Lines (float_4 units)
    let qkv_lines = comptime!(mini_batch_len * head_dim / LINE_SIZE);
    let eta_lines = comptime!(mini_batch_len / LINE_SIZE);

    // Initialize weight in shared memory from inputs.weight
    let mut weight_smem = P::st_ff();
    let weight_offset = (batch_idx * num_heads + head_idx) * head_dim * head_dim / LINE_SIZE;
    cube::load_st_direct(&inputs.weight, &mut weight_smem, weight_offset, 0, 0);

    sync_cube();

    // Initialize bias in registers from inputs.bias
    let mut bias_rv = P::rvb_f_v();
    let bias_offset = (batch_idx * num_heads + head_idx) * head_dim / LINE_SIZE;
    cube::broadcast::load_rv_direct(&inputs.bias, &mut bias_rv, bias_offset);

    // Load layer norm params
    let ln_offset = head_idx * head_dim / LINE_SIZE;
    let mut ln_weight_rv = P::rvb_f_v();
    let mut ln_bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, ln_offset);
    cube::broadcast::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, ln_offset);

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!(
                "PTR_STREAM: init done, entering main loop ctrl=%u\n",
                ctrl_idx
            );
        }
    }

    // Main processing loop
    loop {
        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: loop iteration ctrl=%u\n", ctrl_idx);
            }
        }

        // Poll for status change (only thread 0)
        // Wait for READY (1) or SHUTDOWN (3), skip IDLE (0) and DONE (2)
        let mut status: u32 = 0u32;
        if UNIT_POS == 0 {
            loop {
                // System fence to invalidate cache and see fresh values from host
                memory_fence_system();
                status = Atomic::load(&control[ctrl_idx]);
                if comptime!(debug) {
                    if batch_idx == 0 && head_idx == 0 {
                        debug_print!("PTR_STREAM: poll status=%u\n", status);
                    }
                }
                // Only break on READY (1) or SHUTDOWN (3)
                if status == 1u32 || status == 3u32 {
                    break;
                }
                // Small sleep to reduce memory bus contention
                gpu_sleep(1000u32);
            }
        }

        // Broadcast status to all threads
        sync_cube();
        status = Atomic::load(&control[ctrl_idx]);

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: after sync status=%u\n", status);
            }
        }

        if status == 3u32 {
            // SHUTDOWN
            if comptime!(debug) {
                if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                    debug_print!("PTR_STREAM: shutdown received ctrl=%u\n", ctrl_idx);
                }
            }
            break;
        }

        // Clear status to IDLE immediately to prevent host from seeing stale DONE
        // This ensures the host's poll for DONE won't see values from previous iterations
        if UNIT_POS == 0 {
            Atomic::store(&control[ctrl_idx], 0u32); // IDLE
        }
        memory_fence_system();

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: loading from pointers ctrl=%u\n", ctrl_idx);
            }
        }

        // Load input data from pointers via injected HIP code
        load_from_pointers::<P>(xq_buf, xk_buf, xv_buf, eta_buf, qkv_lines, eta_lines);

        sync_cube();

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: copying to scratch tensors ctrl=%u\n", ctrl_idx);
            }
        }

        // Use index_2d for proper stride-aware offset calculation (returns scalar offset)
        // This matches how the non-streaming forward and d2d streaming compute offsets
        let base_qkv = index_2d(&inputs.xq, batch_idx, head_idx);
        let base_eta = index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx);

        // Convert to Line offsets for Array/direct-Tensor indexing
        let qkv_offset_lines = base_qkv / LINE_SIZE;
        let eta_offset_lines = base_eta / LINE_SIZE;

        if comptime!(debug) {
            if UNIT_POS == 0 {
                debug_print!(
                    "PTR_STREAM: cube b=%u h=%u qkv=%u eta=%u\n",
                    batch_idx as u32,
                    head_idx as u32,
                    base_qkv as u32,
                    base_eta as u32
                );
            }
        }

        // Copy from Arrays to scratch Tensors in Inputs struct.
        // Array and direct Tensor indexing use Line offsets
        copy_array_to_tensor(
            xq_buf,
            qkv_offset_lines,
            &mut inputs.xq,
            qkv_offset_lines,
            qkv_lines,
        );
        copy_array_to_tensor(
            xk_buf,
            qkv_offset_lines,
            &mut inputs.xk,
            qkv_offset_lines,
            qkv_lines,
        );
        copy_array_to_tensor(
            xv_buf,
            qkv_offset_lines,
            &mut inputs.xv,
            qkv_offset_lines,
            qkv_lines,
        );
        copy_array_to_tensor(
            eta_buf,
            eta_offset_lines,
            &mut inputs.ttt_lr_eta,
            eta_offset_lines,
            eta_lines,
        );

        sync_cube();

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: running forward stage ctrl=%u\n", ctrl_idx);
            }
        }

        // Run TTT forward stage
        // base_qkv and base_eta are scalar offsets from index_2d
        fused_ttt_forward_stage::<P>(
            inputs,
            outputs,
            &mut weight_smem,
            &mut bias_rv,
            &ln_weight_rv,
            &ln_bias_rv,
            base_qkv, // stage_offset - scalar offset for load_st_transpose
            base_eta, // ttt_lr_eta_idx - scalar offset for ttt_lr_eta
            epsilon,
        );

        sync_cube();

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: storing output ctrl=%u\n", ctrl_idx);
            }
        }

        // Copy output back to pointer destination using Line offset
        copy_tensor_to_array(
            &outputs.output,
            qkv_offset_lines,
            xq_buf,
            qkv_offset_lines,
            qkv_lines,
        );
        store_to_output::<P>(xq_buf, qkv_lines);

        sync_cube();

        // Mark as done
        if UNIT_POS == 0 {
            Atomic::store(&control[ctrl_idx], 2u32); // DONE
        }

        // System fence to ensure DONE is visible to host before we loop back to polling.
        // This also invalidates our cache so we can see the host's next READY write.
        memory_fence_system();

        if comptime!(debug) {
            if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
                debug_print!("PTR_STREAM: marked DONE ctrl=%u\n", ctrl_idx);
            }
        }

        sync_cube();
    }

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!("PTR_STREAM: storing final weight/bias ctrl=%u\n", ctrl_idx);
        }
    }

    // Shutdown: store final weight and bias to outputs
    let weight_out_offset = (batch_idx * num_heads + head_idx) * head_dim * head_dim / LINE_SIZE;
    let bias_out_offset = (batch_idx * num_heads + head_idx) * head_dim / LINE_SIZE;

    cube::store_st_direct(
        &weight_smem,
        &mut outputs.weight_out,
        weight_out_offset,
        0,
        0,
    );
    cube::broadcast::store_rv_direct(&bias_rv, &mut outputs.bias_out, bias_out_offset);

    // Ensure all stores are complete before kernel exits
    sync_cube();

    if comptime!(debug) {
        if UNIT_POS == 0 && batch_idx == 0 && head_idx == 0 {
            debug_print!("PTR_STREAM: kernel exit ctrl=%u\n", ctrl_idx);
        }
    }
}
