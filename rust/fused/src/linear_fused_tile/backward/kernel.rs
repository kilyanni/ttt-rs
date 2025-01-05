use cubecl::prelude::*;
use thundercube::{cube::ReduceBuf, prelude::*, reduction_ops::SumOp, util::index_2d};

use super::{
    super::{
        helpers::{ParamsTrait, RvbFV, StCsF, StFCs, StFF},
        layer_norm::layer_norm_l2_grad,
    },
    stage::fused_ttt_backward_stage,
    types::{GradOutputs, RecomputationInputs, SavedTensors, atomic_add_rv},
};
use crate::FusedTttConfig;

// =============================================================================
// Kernel entry points
// =============================================================================

/// Fused TTT-Linear backward pass kernel (single mini-batch).
#[cube(launch, launch_unchecked)]
pub fn fused_ttt_backward_kernel<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    grad_output: &Tensor<Line<P::EVal>>,
    grads: &mut GradOutputs<P::EVal>,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let epsilon = comptime!(config.epsilon());

    let base_qkv = index_2d(&saved.xq, batch_idx, head_idx);
    let ttt_lr_eta_idx = index_2d(&saved.ttt_lr_eta, batch_idx, head_idx);

    // For single-stage, weight_stage = weight_init, bias_stage = bias_init
    let base_weight = index_2d(&saved.weight_init, batch_idx, head_idx);
    let base_bias = index_2d(&recomp.bias, batch_idx, head_idx);

    // Weight gradient accumulator uses global memory (no StFF allocation for grad_L_W_last)
    let grad_weight_base = index_2d(&grads.grad_weight, batch_idx, head_idx);
    let grad_bias_base = index_2d(&grads.grad_bias, batch_idx, head_idx);

    let mut weight_stage = P::st_ff();
    // Zero the weight gradient in global memory using weight_stage as a shuttle
    weight_stage.fill(P::EVal::new(0.0));

    sync_cube();

    cube::store_st_direct(
        &weight_stage,
        &mut grads.grad_weight,
        grad_weight_base,
        0,
        0,
    );

    // Now load the actual weight_init
    cube::load_st_direct(&saved.weight_init, &mut weight_stage, base_weight, 0, 0);

    sync_cube();

    let mut bias_stage = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&recomp.bias, &mut bias_stage, base_bias);

    // REMOVED: grad_L_W_last StFF allocation (now uses global memory)

    let mut grad_L_b_last = P::rvb_f_a();
    grad_L_b_last.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_weight_acc = P::rvb_f_a();
    grad_L_ln_weight_acc.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_bias_acc = P::rvb_f_a();
    grad_L_ln_bias_acc.fill(P::EAcc::new(0.0));

    // Shared scratch tiles
    let mut scratch1 = P::st_cs_f();
    let mut scratch2 = P::st_cs_f();
    let mut ext_k_smem = P::st_f_cs();
    let mut tile_b = P::st_cs_f();
    let mut tile_c = P::st_cs_f();
    let mut ext_buf = ReduceBuf::<P::EAcc>::new();

    sync_cube();

    let base_weight_init = index_2d(&saved.weight_init, batch_idx, head_idx);

    fused_ttt_backward_stage::<P>(
        saved,
        recomp,
        grad_output,
        &mut weight_stage,
        &bias_stage,
        &mut grad_L_b_last,
        &mut grad_L_ln_weight_acc,
        &mut grad_L_ln_bias_acc,
        grads,
        base_qkv,
        ttt_lr_eta_idx,
        0, // token_eta_base: single-stage, offset = 0
        &saved.weight_init,
        base_weight_init,
        grad_weight_base,
        &mut scratch1,
        &mut scratch2,
        &mut ext_k_smem,
        &mut tile_b,
        &mut tile_c,
        &mut ext_buf,
        epsilon,
    );

    sync_cube();

    // Weight gradient already accumulated to global memory by backward_stage.
    // Store accumulated bias gradient.
    let grad_L_b_last_val = grad_L_b_last.cast::<P::EVal>();
    cube::broadcast::store_rv_direct(&grad_L_b_last_val, &mut grads.grad_bias, grad_bias_base);

    // Atomically add LN gradients (unbatched tensors shared across batch dimension)
    // LN tensors have shape [num_heads, head_dim], indexed by head_idx only
    let base_ln = head_idx * P::F::VALUE;
    atomic_add_rv::<_, P::F>(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, base_ln);
    atomic_add_rv::<_, P::F>(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, base_ln);
}

/// Simulate one forward stage to update weight and bias in place.
/// This recomputes the weight/bias evolution without producing output.
///
/// weight_out = weight - last_eta * XK^T @ grad_l
/// bias_out = bias - last_eta @ grad_l
///
/// Uses external scratch tiles (shared with backward stage) to avoid
/// exceeding shared memory limits in the multi-stage kernel.
#[cube]
#[allow(clippy::too_many_arguments)]
fn forward_simulate_weight_update<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    weight_smem: &mut StFF<P>,
    bias_rv: &mut RvbFV<P>,
    ln_weight_rv: &RvbFV<P>,
    ln_bias_rv: &RvbFV<P>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    // External tiles (shared with backward stage to avoid duplicate smem allocs)
    k_smem: &mut StFCs<P>,
    scratch_a: &mut StCsF<P>,
    scratch_b: &mut StCsF<P>,
    scratch_c: &mut StCsF<P>,
    scratch_d: &mut StCsF<P>,
    buf: &mut ReduceBuf<P::EAcc>,
    #[comptime] epsilon: f32,
) {
    // scratch_a = xk_smem, scratch_b = v_direct_smem, scratch_c = z1_smem, scratch_d = temp_smem

    // Load XK transposed and direct
    cube::load_st_transpose(&saved.xk, k_smem, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xk, scratch_a, stage_offset, 0, 0);
    cube::load_st_direct(&recomp.xv, scratch_b, stage_offset, 0, 0);

    sync_cube();

    // z1 = xk @ W + b
    let mut z1_reg = P::rt_cs_f();
    z1_reg.zero();
    cube::mma_AtB(&mut z1_reg, k_smem, weight_smem);

    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let mut bias_reg = P::rv_f();
    #[unroll]
    for i in 0..P::F_Reg::LINES {
        let src_idx = thread_n * P::F_Reg::LINES + i;
        bias_reg.data[i] = thundercube::util::cast_line(bias_rv.data[src_idx]);
    }
    z1_reg.add_row(&bias_reg);

    cube::store_rt_to_st(&z1_reg, scratch_c);

    // target = xv - xk
    scratch_b.sub(scratch_a);

    sync_cube();

    // grad_l = layer_norm_l2_grad(z1, target)
    layer_norm_l2_grad::<P::EVal, P::EAcc, P::CS, P::F>(
        scratch_c,
        scratch_b,
        ln_weight_rv,
        ln_bias_rv,
        scratch_d,
        buf,
        epsilon,
    );

    let grad_l = scratch_c;

    // Weight update: W -= last_eta * XK^T @ grad_l
    let last_token_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_idx % comptime!(LINE_SIZE);
    let token_eta_line = saved.token_eta[last_line_idx];
    let last_token_eta_scalar = token_eta_line[last_elem_in_line];

    let mut last_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta_scalar);

    // Reload k transposed (reuse k_smem)
    cube::load_st_transpose(&saved.xk, k_smem, stage_offset, 0, 0);

    sync_cube();

    // Scale by last_eta
    k_smem.mul_row(&last_eta_rv);

    sync_cube();

    // weight_update = scaled_xk^T @ grad_l
    let mut weight_update_reg = P::rt_ff();
    weight_update_reg.zero();
    cube::mma_AB(&mut weight_update_reg, k_smem, grad_l);

    // W -= weight_update
    let mut weight_reg = P::rt_ff();
    cube::load_rt_from_st(weight_smem, &mut weight_reg);
    weight_reg.sub(&weight_update_reg);
    cube::store_rt_to_st(&weight_reg, weight_smem);

    sync_cube();

    // Bias update: b -= last_eta @ grad_l
    scratch_d.copy_from(grad_l);

    sync_cube();

    scratch_d.mul_col(&last_eta_rv);

    sync_cube();

    let mut bias_update_rv = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(scratch_d, &mut bias_update_rv, buf);

    let bias_update_val = bias_update_rv.cast::<P::EVal>();
    bias_rv.sub(&bias_update_val);
}

/// Fused TTT-Linear backward pass kernel (multi-stage).
///
/// For multi-stage backward, we process stages in reverse order.
/// Per-stage weight/bias checkpoints are loaded from tensors saved during forward,
/// eliminating the O(N^2) forward re-simulation.
#[cube(launch, launch_unchecked)]
pub fn fused_ttt_backward_kernel_multi<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    weight_checkpoints: &Tensor<Line<P::EVal>>,
    bias_checkpoints: &Tensor<Line<P::EVal>>,
    // Per-(batch,head) scratch buffer for storing reconstructed W[stage_idx]
    // before backward_stage overwrites it. Shape: [batch, heads, head_dim, head_dim].
    weight_stage_buf: &mut Tensor<Line<P::EVal>>,
    grad_output: &Tensor<Line<P::EVal>>,
    grads: &mut GradOutputs<P::EVal>,
    num_stages: u32,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let num_heads = saved.xq.shape(1);
    let epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(P::CS::VALUE);
    let head_dim = comptime!(P::F::VALUE);
    let checkpoint_interval = comptime!(config.checkpoint_interval);
    let stage_stride = mini_batch_len * head_dim;
    let num_stages = num_stages as usize;

    let base_qkv = index_2d(&saved.xq, batch_idx, head_idx);
    let base_ttt_lr_eta = index_2d(&saved.ttt_lr_eta, batch_idx, head_idx);

    // Checkpoint layout matches forward: ceil(num_stages / checkpoint_interval) per batch/head
    let num_checkpoints = num_stages.div_ceil(checkpoint_interval);
    let ckpt_bh = (batch_idx * num_heads + head_idx) * num_checkpoints;

    // Per-(batch,head) offset into weight_stage_buf [batch, heads, F, F]
    let bh_buf_offset = index_2d(weight_stage_buf, batch_idx, head_idx);

    // Weight gradient accumulator uses global memory (no StFF allocation for grad_L_W_last)
    let grad_weight_base = index_2d(&grads.grad_weight, batch_idx, head_idx);
    let grad_bias_base = index_2d(&grads.grad_bias, batch_idx, head_idx);

    // REMOVED: grad_L_W_last StFF allocation (now uses global memory)

    let mut grad_L_b_last = P::rvb_f_a();
    grad_L_b_last.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_weight_acc = P::rvb_f_a();
    grad_L_ln_weight_acc.fill(P::EAcc::new(0.0));

    let mut grad_L_ln_bias_acc = P::rvb_f_a();
    grad_L_ln_bias_acc.fill(P::EAcc::new(0.0));

    // Shared scratch tiles (reused between forward_simulate and backward_stage)
    let mut scratch1 = P::st_cs_f();
    let mut scratch2 = P::st_cs_f();
    let mut ext_k_smem = P::st_f_cs();
    let mut tile_b = P::st_cs_f();
    let mut tile_c = P::st_cs_f();
    let mut ext_buf = ReduceBuf::<P::EAcc>::new();
    let mut weight_stage = P::st_ff();

    sync_cube();

    // Zero the weight gradient in global memory using weight_stage as a shuttle
    weight_stage.fill(P::EVal::new(0.0));

    sync_cube();

    cube::store_st_direct(
        &weight_stage,
        &mut grads.grad_weight,
        grad_weight_base,
        0,
        0,
    );

    // Load LN params once (needed for forward simulation)
    let base_ln = index_2d(&saved.ln_weight, head_idx, 0);
    let mut ln_weight_rv = P::rvb_f_v();
    let mut ln_bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&saved.ln_weight, &mut ln_weight_rv, base_ln);
    cube::broadcast::load_rv_direct(&recomp.ln_bias, &mut ln_bias_rv, base_ln);

    // Process stages in reverse order (backward through time).
    for stage in 0..num_stages {
        let stage_idx = num_stages - 1 - stage;
        let stage_offset = base_qkv + stage_idx * stage_stride;
        let ttt_lr_eta_idx = base_ttt_lr_eta + stage_idx * mini_batch_len;

        // === Reconstruct W[stage_idx] from nearest earlier checkpoint ===
        let ckpt_stage = (stage_idx / checkpoint_interval) * checkpoint_interval;
        let ckpt_idx = ckpt_stage / checkpoint_interval;

        let ckpt_weight_offset = (ckpt_bh + ckpt_idx) * head_dim * head_dim;
        cube::load_st_direct(
            weight_checkpoints,
            &mut weight_stage,
            ckpt_weight_offset,
            0,
            0,
        );

        sync_cube();

        let mut bias_stage = P::rvb_f_v();
        let ckpt_bias_offset = (ckpt_bh + ckpt_idx) * head_dim;
        cube::broadcast::load_rv_direct(bias_checkpoints, &mut bias_stage, ckpt_bias_offset);

        // Forward-simulate from checkpoint to stage_idx (0 iterations if checkpoint_interval=1)
        for fwd in ckpt_stage..stage_idx {
            let fwd_offset = base_qkv + fwd * stage_stride;
            let fwd_ttt_lr = base_ttt_lr_eta + fwd * mini_batch_len;

            forward_simulate_weight_update::<P>(
                saved,
                recomp,
                &mut weight_stage,
                &mut bias_stage,
                &ln_weight_rv,
                &ln_bias_rv,
                fwd_offset,
                fwd_ttt_lr,
                &mut ext_k_smem,
                &mut scratch1,
                &mut scratch2,
                &mut tile_b,
                &mut tile_c,
                &mut ext_buf,
                epsilon,
            );

            sync_cube();
        }

        // Store reconstructed W[stage_idx] to global buffer before backward_stage
        // overwrites weight_stage (reused as temp_f_f). The backward stage will
        // reload from this buffer when it needs W[stage_idx] again.
        cube::store_st_direct(&weight_stage, weight_stage_buf, bh_buf_offset, 0, 0);

        // Each stage writes its token_eta gradient to its own offset within [seq_len]
        let token_eta_base = stage_idx * mini_batch_len;
        fused_ttt_backward_stage::<P>(
            saved,
            recomp,
            grad_output,
            &mut weight_stage,
            &bias_stage,
            &mut grad_L_b_last,
            &mut grad_L_ln_weight_acc,
            &mut grad_L_ln_bias_acc,
            grads,
            stage_offset,
            ttt_lr_eta_idx,
            token_eta_base,
            weight_stage_buf,
            bh_buf_offset,
            grad_weight_base,
            &mut scratch1,
            &mut scratch2,
            &mut ext_k_smem,
            &mut tile_b,
            &mut tile_c,
            &mut ext_buf,
            epsilon,
        );

        sync_cube();
    }

    // Weight gradient already accumulated to global memory by backward_stage.
    // Store accumulated bias gradient.
    let grad_L_b_last_val = grad_L_b_last.cast::<P::EVal>();
    cube::broadcast::store_rv_direct(&grad_L_b_last_val, &mut grads.grad_bias, grad_bias_base);

    // Atomically add LN gradients (unbatched tensors shared across batch dimension)
    // LN tensors have shape [num_heads, head_dim], indexed by head_idx only
    let ln_base = head_idx * P::F::VALUE;
    atomic_add_rv::<_, P::F>(&grad_L_ln_weight_acc, &mut grads.grad_ln_weight, ln_base);
    atomic_add_rv::<_, P::F>(&grad_L_ln_bias_acc, &mut grads.grad_ln_bias, ln_base);
}
