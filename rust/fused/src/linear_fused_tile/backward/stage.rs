use cubecl::prelude::*;
use thundercube::{cube::ReduceBuf, prelude::*, reduction_ops::SumOp, util::index_2d};

use super::{
    super::{
        helpers::{
            ParamsTrait, RvbFA, RvbFV, StCsF, StFCs, StFF, build_attn_matrix, build_eta_attn_fused,
            build_eta_matrix,
        },
        layer_norm::{compute_grad_x_from_grad_x_hat, normalize_to_x_hat},
    },
    types::{GradOutputs, RecomputationInputs, SavedTensors, atomic_add_rv},
};
// =============================================================================
// Main backward stage function
// =============================================================================

/// Memory layout (optimized for shared memory reduction):
/// For CS=mini_batch_size, F=head_dim:
///
/// Internal tiles:
/// - 5 CS*F tiles: tile_grad_z1_bar, tile_grad_xk_combined, tile_e, grad_l_smem (+1 q_smem as F*CS)
/// - 2 CS*CS tiles: cs_cs_a, cs_cs_b
/// - 1 F*CS tile: q_smem
///
/// External tiles (from caller):
/// - 4 CS*F tiles: scratch1, scratch2, tile_b, tile_c
/// - 1 F*CS tile: k_smem
/// - 1 F*F tile: weight_stage
/// - 1 ReduceBuf
///
/// Weight gradient (grad_L_W_last) uses global memory instead of a 2nd F*F tile.
/// Stage 2 uses dual-purpose tiles instead of separate grad_Z1/grad_target tiles.
/// Fused LN intermediates (x_hat, grad_output, grad_x_hat) are recomputed before
/// stage 2 instead of saved, eliminating 3 CS*F tiles.
///
/// Example sizes (at f32):
/// - 16x32:  5*1KB + 2*0.5KB + 1*1KB + 4*1KB + 1*1KB + 1*4KB + 1KB = 18KB
/// - 16x64:  5*4KB + 2*1KB + 1*4KB + 4*4KB + 1*4KB + 1*16KB + 1KB = 59KB (fits 64KB LDS)
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_backward_stage<P: ParamsTrait>(
    saved: &SavedTensors<P::EVal>,
    recomp: &RecomputationInputs<P::EVal>,
    grad_L_XQW: &Tensor<Line<P::EVal>>,
    // Weight at this stage. Reused as scratch F*F tile after recomputation.
    weight_stage: &mut StFF<P>,
    bias_stage: &RvbFV<P>,
    // Weight gradient accumulated via global memory (grad_L_W_last eliminated from smem)
    grad_L_b_last: &mut RvbFA<P>,
    grad_L_ln_weight_acc: &mut RvbFA<P>,
    grad_L_ln_bias_acc: &mut RvbFA<P>,
    grads: &mut GradOutputs<P::EVal>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    token_eta_base: usize,
    // Tensor + base offset for reloading the stage weight after it's been overwritten.
    // For single-stage: saved.weight_init with batch/head offset.
    // For multi-stage: weight_stage_buf with batch/head offset.
    weight_z1bar_tensor: &Tensor<Line<P::EVal>>,
    weight_z1bar_base: usize,
    // Global memory offset for weight gradient accumulation
    grad_weight_base: usize,
    // External tiles (allocated by caller, shared with forward_simulate in multi-stage)
    scratch1: &mut StCsF<P>,
    scratch2: &mut StCsF<P>,
    k_smem: &mut StFCs<P>,
    tile_b: &mut StCsF<P>,
    tile_c: &mut StCsF<P>,
    buf: &mut ReduceBuf<P::EAcc>,
    #[comptime] epsilon: f32,
) {
    let mut cs_cs_a = P::st_cs_cs();
    let mut cs_cs_b = P::st_cs_cs();
    let mut tile_grad_z1_bar = P::st_cs_f();
    let mut tile_grad_xk_combined = P::st_cs_f();
    let mut tile_e = P::st_cs_f();
    // REMOVED: tile_grad_Z1, tile_grad_target (folded into stage2 dual-purpose tiles)
    // REMOVED: xk_smem (use tile_b temporarily)
    // REMOVED: x_hat_fused_smem, grad_output_fused_smem, grad_x_hat_fused_smem (recomputed before stage 2)

    // =========================================================================
    // Load persistent data
    // =========================================================================

    let mut q_smem = P::st_f_cs();
    // k_smem passed from caller (shared with forward_simulate)
    cube::load_st_transpose(&saved.xq, &mut q_smem, stage_offset, 0, 0);
    cube::load_st_transpose(&saved.xk, k_smem, stage_offset, 0, 0);

    // LN weight and bias
    let base_ln = index_2d(&saved.ln_weight, CUBE_POS_Y as usize, 0);
    let mut ln_weight_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&saved.ln_weight, &mut ln_weight_rv, base_ln);
    let mut ln_bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&recomp.ln_bias, &mut ln_bias_rv, base_ln);

    // Last eta computation
    let last_token_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_idx % comptime!(LINE_SIZE);
    let token_eta_line = saved.token_eta[last_line_idx];
    let last_token_eta = token_eta_line[last_elem_in_line];

    let mut last_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta);

    // Bias as register tile row (needed for z1 and z1_bar computations)
    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let mut bias_reg = P::rv_f();
    #[unroll]
    for i in 0..P::F_Reg::LINES {
        let src_idx = thread_n * P::F_Reg::LINES + i;
        bias_reg.data[i] = thundercube::util::cast_line(bias_stage.data[src_idx]);
    }

    // =========================================================================
    // Recompute grad_l (without saving fused intermediates)
    // =========================================================================

    // Load xk direct into tile_b (instead of separate xk_smem allocation)
    cube::load_st_direct(&saved.xk, tile_b, stage_offset, 0, 0);

    sync_cube();

    // z1 = xk @ W_stage
    let mut z1_reg = P::rt_cs_f();
    z1_reg.zero();
    cube::mma_AtB(&mut z1_reg, k_smem, weight_stage);

    // z1 += bias_stage
    z1_reg.add_row(&bias_reg);

    // Store z1 to grad_l_smem (will be overwritten with grad_l)
    let mut grad_l_smem = P::st_cs_f();
    cube::store_rt_to_st(&z1_reg, &mut grad_l_smem);

    // Compute target = xv - xk (using tile_b for xk, tile_e as temp for xv)
    cube::load_st_direct(&recomp.xv, &mut tile_e, stage_offset, 0, 0);

    sync_cube();

    tile_e.sub(tile_b); // tile_e = target = xv - xk

    // Normalize z1 -> x_hat (in place in grad_l_smem), get std
    let _std_fused_initial =
        normalize_to_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(&mut grad_l_smem, buf, epsilon);
    let x_hat_smem = grad_l_smem;

    // Compute y = ln_weight * x_hat + ln_bias, then grad_output = y - target
    scratch1.copy_from(&x_hat_smem);
    scratch1.mul_row(&ln_weight_rv);
    scratch1.add_row(&ln_bias_rv);
    scratch1.sub(&tile_e);

    sync_cube();

    // DON'T save grad_output -- will recompute before stage 2.
    // grad_x_hat = grad_output * ln_weight
    scratch1.mul_row(&ln_weight_rv);

    sync_cube();

    // DON'T save grad_x_hat -- will recompute before stage 2.
    // Compute grad_x from grad_x_hat (overwrites x_hat with grad_l)
    let mut grad_l_smem = x_hat_smem;
    compute_grad_x_from_grad_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(
        scratch1,
        &mut grad_l_smem,
        &_std_fused_initial,
        scratch2,
        buf,
    );

    sync_cube();

    // =========================================================================
    // Recompute z1_bar and its layer norm intermediates
    // =========================================================================

    // z1_bar = xq @ W_stage + b_stage - eta @ grad_l - (eta * attn) @ grad_l

    // Step 1: eta_matrix = outer(token_eta, ttt_lr_eta).tril()
    build_eta_matrix::<P>(
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &mut cs_cs_a,
        ttt_lr_eta_idx,
        false,
    );

    // Step 2: eta @ grad_l
    let mut eta_grad_reg = P::rt_cs_f();
    eta_grad_reg.zero();
    cube::mma_AB(&mut eta_grad_reg, &cs_cs_a, &grad_l_smem);

    // ILP: z1_bar = xq @ W_stage (independent of eta_attn, computed early to overlap with step 3)
    let mut z1_bar_reg = P::rt_cs_f();
    z1_bar_reg.zero();
    cube::mma_AtB(&mut z1_bar_reg, &q_smem, weight_stage);

    // Step 3: Build (eta * attn) fused
    build_eta_attn_fused::<P>(
        &q_smem,
        k_smem,
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &mut cs_cs_a,
        ttt_lr_eta_idx,
    );

    // (eta * attn) @ grad_l
    let mut eta_attn_grad_reg = P::rt_cs_f();
    eta_attn_grad_reg.zero();
    cube::mma_AB(&mut eta_attn_grad_reg, &cs_cs_a, &grad_l_smem);

    // z1_bar += bias_stage
    z1_bar_reg.add_row(&bias_reg);
    // z1_bar -= eta @ grad_l
    z1_bar_reg.sub(&eta_grad_reg);
    // z1_bar -= (eta * attn) @ grad_l
    z1_bar_reg.sub(&eta_attn_grad_reg);

    // Store z1_bar into scratch1 for layer norm
    cube::store_rt_to_st(&z1_bar_reg, scratch1);

    sync_cube();

    // Normalize z1_bar to get x_hat_ln and std_ln
    let std_ln = normalize_to_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(scratch1, buf, epsilon);

    let x_hat_ln = scratch1;

    // =========================================================================
    // Stage 4: LN backward (inlined from backward_stage4_ln)
    // =========================================================================

    // Load upstream gradient
    cube::load_st_direct(grad_L_XQW, tile_c, stage_offset, 0, 0);

    sync_cube();

    let f_f = P::EVal::cast_from(P::F::VALUE as f32);
    let f_inv = P::EVal::cast_from(1.0f32 / (P::F::VALUE as f32));

    // grad_ln_weight_s4 = sum(grad_output * x_hat_ln)
    tile_b.copy_from(tile_c);
    tile_b.mul(x_hat_ln);

    sync_cube();

    let mut grad_ln_weight_s4 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(tile_b, &mut grad_ln_weight_s4, buf);

    // grad_ln_bias_s4 = sum(grad_output)
    let mut grad_ln_bias_s4 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(tile_c, &mut grad_ln_bias_s4, buf);

    // grad_x_hat = grad_output * ln_weight
    tile_b.copy_from(tile_c);
    tile_b.mul_row(&ln_weight_rv);

    sync_cube();

    // sum(grad_x_hat) per row
    let mut sum_gxh_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum_gxh_acc, buf);
    let sum_gxh = sum_gxh_acc.cast::<P::EVal>();

    // sum(grad_x_hat * x_hat) per row
    scratch2.copy_from(tile_b);
    scratch2.mul(x_hat_ln);

    sync_cube();

    let mut sum_gxh_xh_s4_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut sum_gxh_xh_s4_acc, buf);
    let sum_gxh_xh_s4 = sum_gxh_xh_s4_acc.cast::<P::EVal>();

    // grad_z1_bar = (grad_x_hat * F - sum_gxh - x_hat * sum_gxh_xh) / (std * F)
    tile_grad_z1_bar.copy_from(tile_b);
    tile_grad_z1_bar.mul_scalar(f_f);
    tile_grad_z1_bar.sub_col(&sum_gxh);

    sync_cube();

    scratch2.copy_from(x_hat_ln);
    scratch2.mul_col(&sum_gxh_xh_s4);

    sync_cube();

    tile_grad_z1_bar.sub(scratch2);
    tile_grad_z1_bar.div_col(&std_ln);
    tile_grad_z1_bar.mul_scalar(f_inv);

    sync_cube();

    // grad_W_z1bar = XQ^T @ grad_z1_bar
    let mut dW_reg = P::rt_ff();
    dW_reg.zero();
    cube::mma_AB(&mut dW_reg, &q_smem, &tile_grad_z1_bar);

    cube::store_rt_to_st(&dW_reg, weight_stage);

    sync_cube();

    // grad_b_z1bar = sum(grad_z1_bar)
    let mut grad_b_z1bar = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        &tile_grad_z1_bar,
        &mut grad_b_z1bar,
        buf,
    );

    // =========================================================================
    // Global memory accumulation: save dW_z1bar, load accumulated grad_W
    // =========================================================================

    // Save dW_z1bar from weight_stage to register before overwriting
    let mut dw_z1bar_rt = P::rt_ff();
    cube::load_rt_from_st(weight_stage, &mut dw_z1bar_rt);

    sync_cube();

    // Load current accumulated weight gradient from global memory
    cube::load_st_direct(&grads.grad_weight, weight_stage, grad_weight_base, 0, 0);

    sync_cube();

    let grad_W_last = weight_stage;

    // =========================================================================
    // Stage 3 Part 1: Update backward (inlined from backward_stage3_part1)
    // =========================================================================

    let scratch1 = x_hat_ln;
    cube::load_st_direct(&saved.xk, scratch1, stage_offset, 0, 0);
    cube::load_st_direct(&saved.xq, tile_c, stage_offset, 0, 0);

    sync_cube();

    let xk_smem = scratch1;
    let xq_smem = tile_c;

    // --- Compute grad_grad_l from three sources ---

    // Build eta^T (upper triangular) into cs_cs_a
    build_eta_matrix::<P>(
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &mut cs_cs_a,
        ttt_lr_eta_idx,
        true,
    );

    // Build attn^T (upper triangular) into cs_cs_b
    build_attn_matrix::<P>(&q_smem, k_smem, &mut cs_cs_b, true);

    // Term A: eta^T @ grad_z1_bar
    let mut term_a_reg = P::rt_cs_f();
    term_a_reg.zero();
    cube::mma_AB(&mut term_a_reg, &cs_cs_a, &tile_grad_z1_bar);

    // Term B: (eta*attn)^T @ grad_z1_bar = (eta^T * attn^T) @ grad_z1_bar
    // MMA only reads cs_cs_a; mul reads+writes cs_cs_b -- no conflict
    cs_cs_b.mul(&cs_cs_a);

    sync_cube();

    let mut term_b_reg = P::rt_cs_f();
    term_b_reg.zero();
    cube::mma_AB(&mut term_b_reg, &cs_cs_b, &tile_grad_z1_bar);

    // Combine in registers: grad_grad_l = -(term_a + term_b)
    term_a_reg.add(&term_b_reg);
    term_a_reg.neg();
    cube::store_rt_to_st(&term_a_reg, &mut tile_e);

    sync_cube();

    // Sources 1 & 2: -(last_eta * XK) @ grad_W_last - last_eta * grad_b_last
    tile_b.copy_from(xk_smem);
    tile_b.mul_col(&last_eta_rv);

    sync_cube();

    let mut src12_reg = P::rt_cs_f();
    src12_reg.zero();
    cube::mma_AB(&mut src12_reg, tile_b, grad_W_last);

    // Accumulate into tile_e via registers (avoids tile_b round-trip)
    let mut tile_e_reg = P::rt_cs_f();
    cube::load_rt_from_st(&tile_e, &mut tile_e_reg);
    tile_e_reg.sub(&src12_reg);
    cube::store_rt_to_st(&tile_e_reg, &mut tile_e);

    // Sync ensures: (1) MMA tile_b reads complete, (2) tile_e store visible
    sync_cube();

    let grad_b_last_val = grad_L_b_last.cast::<P::EVal>();
    tile_b.set_row(&grad_b_last_val);
    tile_b.mul_col(&last_eta_rv);

    sync_cube();

    tile_e.sub(tile_b);

    sync_cube();

    let grad_grad_l = tile_e;

    // --- d_xk (from attn) = d_attn^T @ XQ ---
    // Compute d_attn^T directly: grad_l @ grad_z1_bar^T * eta^T
    let mut d_attn_t_reg = P::rt_cs_cs();
    d_attn_t_reg.zero();
    cube::mma_ABt(&mut d_attn_t_reg, &grad_l_smem, &tile_grad_z1_bar);

    cube::store_rt_to_st(&d_attn_t_reg, &mut cs_cs_b);

    sync_cube();

    // cs_cs_a still has eta^T (preserved by swapping mul operands in Term B)
    cs_cs_b.mul(&cs_cs_a);
    cs_cs_b.neg();
    cs_cs_b.triu();

    sync_cube();

    let mut d_xk_attn_reg = P::rt_cs_f();
    d_xk_attn_reg.zero();
    cube::mma_AB(&mut d_xk_attn_reg, &cs_cs_b, xq_smem);

    // Store first d_xk_attn contribution to combined output
    cube::store_rt_to_st(&d_xk_attn_reg, &mut tile_grad_xk_combined);

    sync_cube();

    // --- grad_xk_mini from weight update term ---
    // grad_l_Last = grad_l @ grad_W_last^T
    let mut grad_l_last_reg = P::rt_cs_f();
    grad_l_last_reg.zero();
    cube::mma_ABt(&mut grad_l_last_reg, &grad_l_smem, grad_W_last);

    cube::store_rt_to_st(&grad_l_last_reg, tile_b);

    sync_cube();

    // --- grad_ttt_lr_eta from weight/bias update (first part) ---
    // grad_last_eta = sum(-(grad_l_last * XK) - (grad_b_last * grad_l))
    let grad_l_last_tile = tile_b;
    scratch2.copy_from(grad_l_last_tile);
    scratch2.mul(xk_smem);

    sync_cube();

    let mut grad_eta_term1 = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut grad_eta_term1, buf);

    scratch2.set_row(&grad_b_last_val);
    scratch2.mul(&grad_l_smem);

    sync_cube();

    let mut grad_eta_term2 = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(scratch2, &mut grad_eta_term2, buf);

    grad_eta_term1.add(&grad_eta_term2);
    grad_eta_term1.neg();

    // d_last_eta = -(sum_rows(grad_l_last * XK) + sum_rows(grad_b_last * grad_l))
    // Save before scaling by last_token_eta (needed for grad_token_eta[CS-1])
    let mut d_last_eta = P::rvb_cs_a();
    d_last_eta.copy_from(&grad_eta_term1);

    // Scale by last_token_eta for grad_ttt_lr_eta contribution
    grad_eta_term1.mul_scalar(P::EAcc::cast_from(last_token_eta));

    // grad_xk_mini = -grad_l_last * last_eta
    let tile_b = grad_l_last_tile;
    tile_b.mul_col(&last_eta_rv);
    tile_b.neg();

    sync_cube();

    // Add grad_xk_mini to the combined output (second contribution)
    tile_grad_xk_combined.add(tile_b);

    sync_cube();

    // =========================================================================
    // Accumulate dW_z1bar into the global weight gradient
    // =========================================================================

    sync_cube();

    let mut acc_rt = P::rt_ff();
    cube::load_rt_from_st(grad_W_last, &mut acc_rt);
    acc_rt.add(&dw_z1bar_rt);
    let weight_stage = grad_W_last;
    cube::store_rt_to_st(&acc_rt, weight_stage);

    sync_cube();

    cube::store_st_direct(weight_stage, &mut grads.grad_weight, grad_weight_base, 0, 0);

    // =========================================================================
    // Reload W_stage for stage 3 part 2 and subsequent recomputation
    // =========================================================================

    cube::load_st_direct(weight_z1bar_tensor, weight_stage, weight_z1bar_base, 0, 0);

    sync_cube();

    // =========================================================================
    // Stage 3 Part 2: Update backward (inlined from backward_stage3_part2)
    // =========================================================================

    // --- grad_xq_mini = grad_z1_bar @ W_init^T ---
    let mut grad_xq_reg = P::rt_cs_f();
    grad_xq_reg.zero();
    cube::mma_ABt(&mut grad_xq_reg, &tile_grad_z1_bar, weight_stage);

    cube::store_rt_to_st(&grad_xq_reg, xq_smem);

    sync_cube();

    // --- Gradient through attn = XQ @ XK^T ---
    // d_attn = -grad_z1_bar @ grad_l^T * eta (element-wise, lower triangular)

    // Build eta (lower triangular) into cs_cs_a
    build_eta_matrix::<P>(
        &saved.token_eta,
        &saved.ttt_lr_eta,
        &mut cs_cs_a,
        ttt_lr_eta_idx,
        false,
    );

    // d_attn_base = grad_z1_bar @ grad_l^T into cs_cs_b
    let mut d_attn_reg = P::rt_cs_cs();
    d_attn_reg.zero();
    cube::mma_ABt(&mut d_attn_reg, &tile_grad_z1_bar, &grad_l_smem);

    cube::store_rt_to_st(&d_attn_reg, &mut cs_cs_b);

    sync_cube();

    cs_cs_b.mul(&cs_cs_a);
    cs_cs_b.neg();
    cs_cs_b.tril();

    sync_cube();

    // d_xq (from attn) = d_attn @ XK
    let mut d_xq_attn_reg = P::rt_cs_f();
    d_xq_attn_reg.zero();
    cube::mma_AB(&mut d_xq_attn_reg, &cs_cs_b, xk_smem);

    // Accumulate into xq_smem via registers (avoids tile_b intermediate)
    let mut xq_reg = P::rt_cs_f();
    cube::load_rt_from_st(xq_smem, &mut xq_reg);
    xq_reg.add(&d_xq_attn_reg);
    cube::store_rt_to_st(&xq_reg, xq_smem);

    sync_cube();

    // --- grad_ttt_lr_eta from eta terms in z1_bar ---
    build_attn_matrix::<P>(&q_smem, k_smem, &mut cs_cs_a, false);

    // d_eta_base = -grad_z1_bar @ grad_l^T (lower tri)
    // Reuse d_attn_reg -- same MMA as line 664
    cube::store_rt_to_st(&d_attn_reg, &mut cs_cs_b);

    sync_cube();

    cs_cs_b.neg();
    cs_cs_b.tril();

    // d_eta = d_eta_base + d_eta_base * attn
    cs_cs_a.mul(&cs_cs_b);

    sync_cube();

    cs_cs_b.add(&cs_cs_a);

    sync_cube();

    // --- grad_token_eta from eta terms in z1_bar + weight/bias update ---
    // grad_token_eta[i] = sum_j(d_eta[i,j] * ttt_lr_eta[j])
    // Plus: grad_token_eta[CS-1] += sum_j(d_last_eta[j] * ttt_lr_eta[j])
    let d_eta = cs_cs_b;
    cs_cs_a.copy_from(&d_eta);

    sync_cube();

    // Load ttt_lr_eta for row-wise multiplication
    let mut ttt_lr_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.ttt_lr_eta, &mut ttt_lr_eta_rv, ttt_lr_eta_idx);

    // Multiply each column j by ttt_lr_eta[j]
    cs_cs_a.mul_row(&ttt_lr_eta_rv);

    sync_cube();

    // Sum rows: grad_token_eta[i] = sum_j(d_eta[i,j] * ttt_lr_eta[j])
    let mut grad_token_eta = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::CS>(&cs_cs_a, &mut grad_token_eta, buf);

    // Add weight/bias update contribution to last element:
    // grad_token_eta[CS-1] += dot(d_last_eta, ttt_lr_eta)
    let ttt_lr_eta_acc = ttt_lr_eta_rv.cast::<P::EAcc>();
    let mut d_last_eta_scaled = P::rvb_cs_a();
    d_last_eta_scaled.copy_from(&d_last_eta);
    d_last_eta_scaled.mul(&ttt_lr_eta_acc);
    // Sum all elements of d_last_eta_scaled to get the scalar dot product
    let mut dot_sum = P::EAcc::new(0.0);
    #[unroll]
    for line_idx in 0..P::CS::LINES {
        let line = d_last_eta_scaled.data[line_idx];
        #[unroll]
        for elem_idx in 0..LINE_SIZE {
            dot_sum += line[elem_idx];
        }
    }
    // Add to last element of grad_token_eta
    let gte_last_line = comptime!((P::CS::VALUE - 1) / LINE_SIZE);
    let gte_last_elem = comptime!((P::CS::VALUE - 1) % LINE_SIZE);
    let mut gte_line = grad_token_eta.data[gte_last_line];
    gte_line[gte_last_elem] += dot_sum;
    grad_token_eta.data[gte_last_line] = gte_line;

    // --- grad_ttt_lr_eta from eta terms in z1_bar ---
    // grad_ttt_lr_eta += sum_cols(d_eta * token_eta)
    let mut cs_cs_b = d_eta;
    let mut token_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&saved.token_eta, &mut token_eta_rv, 0);

    cs_cs_b.mul_col(&token_eta_rv);

    sync_cube();

    // Sum columns to get grad_ttt_lr_eta contribution
    let mut grad_ttt_lr_eta = P::rvb_cs_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::CS, SumOp>(&cs_cs_b, &mut grad_ttt_lr_eta, buf);

    grad_ttt_lr_eta.add(&grad_eta_term1);

    // --- End stage 3 ---

    let grad_xq_mini = xq_smem;
    // Rename back: xk_smem is no longer needed, restore scratch1 identity
    let scratch1 = xk_smem;

    // =========================================================================
    // Stage 2: Recompute fused LN intermediates, then backward
    // =========================================================================
    // weight_stage still has W_stage (read-only in stage3_part2). Reuse for z1 computation.

    // Store grad_xk_combined to global memory (frees tile for use in stage 2).
    // Will be reloaded before stage 1.
    cube::store_st_direct(
        &tile_grad_xk_combined,
        &mut grads.grad_xk,
        stage_offset,
        0,
        0,
    );

    // scratch1 still has XK from stage 3 -- only load XV
    cube::load_st_direct(&recomp.xv, tile_b, stage_offset, 0, 0);

    sync_cube();

    // target = xv - xk (in tile_b)
    tile_b.sub(scratch1);

    sync_cube();

    // z1 = xk @ W_stage + bias (k_smem has XK^T, weight_stage has W_stage)
    let mut z1_recomp_reg = P::rt_cs_f();
    z1_recomp_reg.zero();
    cube::mma_AtB(&mut z1_recomp_reg, k_smem, weight_stage);

    z1_recomp_reg.add_row(&bias_reg);

    // Store z1 to tile_grad_z1_bar (will become x_hat_fused)
    cube::store_rt_to_st(&z1_recomp_reg, &mut tile_grad_z1_bar);

    sync_cube();

    // Normalize z1 -> x_hat, std_fused
    let std_fused =
        normalize_to_x_hat::<P::EVal, P::EAcc, P::CS, P::F>(&mut tile_grad_z1_bar, buf, epsilon);
    let x_hat_fused = tile_grad_z1_bar;

    // Compute grad_output = ln_weight * x_hat + ln_bias - target
    scratch2.copy_from(&x_hat_fused);
    scratch2.mul_row(&ln_weight_rv);
    scratch2.add_row(&ln_bias_rv);
    scratch2.sub(tile_b); // scratch2 = grad_output (tile_b had target)

    sync_cube();

    // Move grad_output to scratch1
    scratch1.copy_from(scratch2);

    sync_cube();

    // Compute grad_x_hat = grad_output * ln_weight -> scratch2
    scratch2.mul_row(&ln_weight_rv);

    sync_cube();

    // Compute sum_gxh_xh = sum_rows(grad_x_hat * x_hat) using tile_b as temp
    tile_b.copy_from(scratch2);
    tile_b.mul(&x_hat_fused);

    sync_cube();

    let mut sum_gxh_xh_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum_gxh_xh_acc, buf);
    let sum_gxh_xh = sum_gxh_xh_acc.cast::<P::EVal>();

    // Rename: scratch1/scratch2 take on their stage 2 dual-purpose identities
    let grad_out_then_Z1 = scratch1;
    let grad_xhat_then_target = scratch2;

    // =========================================================================
    // Stage 2: LN+L2 second derivative (inlined from backward_stage2_ln_l2)
    // =========================================================================
    // grad_out_then_Z1: grad_output on entry, grad_Z1 on exit
    // grad_xhat_then_target: grad_x_hat on entry, grad_target on exit
    // grad_l_smem: grad_l on entry, scratch after phase 0

    // === Phase 0: Precompute grad_l-dependent term (frees grad_l tile for scratch) ===
    // term7_partial = sum_rows(grad_grad_l * grad_l / std)
    tile_b.copy_from(&grad_grad_l);
    tile_b.mul(&grad_l_smem);
    tile_b.div_col(&std_fused);

    sync_cube();

    let mut term7_partial_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut term7_partial_acc, buf);
    let term7_partial = term7_partial_acc.cast::<P::EVal>();
    // grad_l consumed; grad_l_smem is scratch from here

    // === Phase 1: Compute sum1, sum2 ===
    tile_b.copy_from(&grad_grad_l);
    tile_b.neg();
    tile_b.div_col(&std_fused);

    sync_cube();

    let mut sum1_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum1_acc, buf);
    let sum1 = sum1_acc.cast::<P::EVal>();

    tile_b.mul(&x_hat_fused);

    sync_cube();

    let mut sum2_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum2_acc, buf);
    let sum2 = sum2_acc.cast::<P::EVal>();

    // === Phase 2: Compute grad_L_gxh in tile_grad_xk_combined (preserved for phase 7) ===
    // grad_L_gxh = grad_grad_l/std + (1/F)*sum1 + (1/F)*x_hat*sum2
    let mut grad_L_gxh = tile_grad_xk_combined;
    grad_L_gxh.copy_from(&grad_grad_l);
    grad_L_gxh.div_col(&std_fused);

    let mut s1 = sum1;
    s1.mul_scalar(f_inv);
    grad_L_gxh.add_col(&s1);

    sync_cube();

    grad_l_smem.copy_from(&x_hat_fused);
    grad_l_smem.mul_col(&sum2);
    grad_l_smem.mul_scalar(f_inv);

    sync_cube();

    grad_L_gxh.add(&grad_l_smem);

    sync_cube();

    // === Phase 3: Consume grad_output, compute grad_ln_weight + grad_ln_bias ===
    // grad_L_y = ln_weight * grad_L_gxh -> store in grad_l_smem
    grad_l_smem.copy_from(&grad_L_gxh);
    grad_l_smem.mul_row(&ln_weight_rv);

    sync_cube();

    // grad_ln_weight = reduce_cols(grad_output * grad_L_gxh + grad_L_y * x_hat)
    // Part 1: grad_out_then_Z1 *= grad_L_gxh (CONSUMES grad_output)
    grad_out_then_Z1.mul(&grad_L_gxh);

    // Part 2: tile_b = grad_L_y * x_hat
    tile_b.copy_from(&grad_l_smem);
    tile_b.mul(&x_hat_fused);

    sync_cube();

    grad_out_then_Z1.add(tile_b);

    sync_cube();

    let mut grad_ln_weight_s2 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        grad_out_then_Z1,
        &mut grad_ln_weight_s2,
        buf,
    );

    // grad_ln_bias = reduce_cols(grad_L_y)
    let mut grad_ln_bias_s2 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        &grad_l_smem,
        &mut grad_ln_bias_s2,
        buf,
    );

    // === Phase 4: Compute grad_L_x_hat in grad_out_then_Z1 (becomes grad_Z1) ===
    // grad_L_x_hat = grad_L_y*ln_weight + (1/F)*grad_x_hat*sum2 + (1/F)*sum_gxh_xh*(-grad_grad_l/std)

    // Term 1: grad_out_then_Z1 = grad_L_y * ln_weight
    grad_out_then_Z1.copy_from(&grad_l_smem);
    grad_out_then_Z1.mul_row(&ln_weight_rv);

    // Term 2: tile_b = (1/F)*grad_x_hat*sum2
    tile_b.copy_from(grad_xhat_then_target);
    tile_b.mul_col(&sum2);
    tile_b.mul_scalar(f_inv);

    sync_cube();

    grad_out_then_Z1.add(tile_b);

    sync_cube();

    // Term 3: tile_b = (1/F)*sum_gxh_xh*(-grad_grad_l/std)
    tile_b.copy_from(&grad_grad_l);
    tile_b.neg();
    tile_b.div_col(&std_fused);
    tile_b.mul_col(&sum_gxh_xh);
    tile_b.mul_scalar(f_inv);

    sync_cube();

    grad_out_then_Z1.add(tile_b); // grad_out_then_Z1 = grad_L_x_hat

    sync_cube();

    // === Phase 5: Compute grad_L_std ===
    // sum_grad_L_std = sum_rows(-grad_L_x_hat*x_hat/std) - term7_partial
    tile_b.copy_from(grad_out_then_Z1);
    tile_b.mul(&x_hat_fused);
    tile_b.div_col(&std_fused);
    tile_b.neg();

    sync_cube();

    let mut sum_grad_L_std_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(tile_b, &mut sum_grad_L_std_acc, buf);
    let mut sum_grad_L_std = sum_grad_L_std_acc.cast::<P::EVal>();
    sum_grad_L_std.sub(&term7_partial);

    // === Phase 6: Compute final grad_Z1 in grad_out_then_Z1 ===
    let mut sum_grad_L_x_hat_acc = P::rvb_cs_a();
    cube::sum_rows::<P::EVal, P::EAcc, P::CS, P::F>(
        grad_out_then_Z1,
        &mut sum_grad_L_x_hat_acc,
        buf,
    );
    let sum_grad_L_x_hat = sum_grad_L_x_hat_acc.cast::<P::EVal>();

    grad_out_then_Z1.div_col(&std_fused);

    let mut term2 = sum_grad_L_x_hat;
    term2.div(&std_fused);
    term2.mul_scalar(f_inv);
    grad_out_then_Z1.sub_col(&term2);

    sync_cube();

    tile_b.copy_from(&x_hat_fused);
    let mut term3 = sum_grad_L_std;
    term3.mul_scalar(f_inv);
    tile_b.mul_col(&term3);

    sync_cube();

    grad_out_then_Z1.add(tile_b); // grad_out_then_Z1 = grad_Z1

    sync_cube();

    // === Phase 7: Compute grad_target in grad_xhat_then_target ===
    // grad_target = -ln_weight * grad_L_gxh
    grad_xhat_then_target.copy_from(&grad_L_gxh);
    grad_xhat_then_target.mul_row(&ln_weight_rv);
    grad_xhat_then_target.neg(); // grad_xhat_then_target = grad_target

    sync_cube();

    // --- End stage 2 ---

    // Rename: dual-purpose tiles to their final identities
    let grad_Z1 = grad_out_then_Z1;
    let grad_target = grad_xhat_then_target;

    // =========================================================================
    // Stage 1: Final assembly (inlined from backward_stage1_assemble)
    // =========================================================================

    let mut tile_e = grad_grad_l;
    cube::load_st_direct(grad_L_XQW, &mut tile_e, stage_offset, 0, 0);

    let mut tile_grad_xk_combined = grad_L_gxh;
    cube::load_st_direct(
        &grads.grad_xk,
        &mut tile_grad_xk_combined,
        stage_offset,
        0,
        0,
    );

    sync_cube();

    // grad_XQ = grad_output + grad_xq_mini
    grad_l_smem.copy_from(&tile_e);
    grad_l_smem.add(grad_xq_mini);

    sync_cube();

    // Store grad_XQ
    cube::store_st_direct(&grad_l_smem, &mut grads.grad_xq, stage_offset, 0, 0);

    // grad_XV = grad_target
    cube::store_st_direct(grad_target, &mut grads.grad_xv, stage_offset, 0, 0);

    // grad_XK = -grad_target + grad_xk_combined + grad_Z1 @ W_stage^T
    // weight_stage still has W_stage
    let mut grad_xk_reg = P::rt_cs_f();
    grad_xk_reg.zero();
    cube::mma_ABt(&mut grad_xk_reg, grad_Z1, weight_stage);

    cube::store_rt_to_st(&grad_xk_reg, &mut grad_l_smem);

    sync_cube();

    grad_l_smem.sub(grad_target);
    grad_l_smem.add(&tile_grad_xk_combined);

    sync_cube();

    cube::store_st_direct(&grad_l_smem, &mut grads.grad_xk, stage_offset, 0, 0);

    // Accumulate weight gradients via global memory: dW = XK^T @ grad_Z1
    let mut dW_s1_reg = P::rt_ff();
    dW_s1_reg.zero();
    cube::mma_AB(&mut dW_s1_reg, k_smem, grad_Z1);

    // Load current global accumulator into weight_stage
    cube::load_st_direct(&grads.grad_weight, weight_stage, grad_weight_base, 0, 0);

    sync_cube();

    // Add dW to accumulator via register tiles
    let mut acc_rt_s1 = P::rt_ff();
    cube::load_rt_from_st(weight_stage, &mut acc_rt_s1);
    acc_rt_s1.add(&dW_s1_reg);
    cube::store_rt_to_st(&acc_rt_s1, weight_stage);

    sync_cube();

    // Store updated accumulator back to global
    cube::store_st_direct(weight_stage, &mut grads.grad_weight, grad_weight_base, 0, 0);

    // grad_b = grad_b_z1bar + sum(grad_Z1)
    let mut grad_b_z1 = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(grad_Z1, &mut grad_b_z1, buf);

    grad_L_b_last.add(&grad_b_z1);
    grad_L_b_last.add(&grad_b_z1bar);

    // Accumulate LN gradients
    grad_L_ln_weight_acc.add(&grad_ln_weight_s4);
    grad_L_ln_weight_acc.add(&grad_ln_weight_s2);
    grad_L_ln_bias_acc.add(&grad_ln_bias_s4);
    grad_L_ln_bias_acc.add(&grad_ln_bias_s2);

    // Store grad_ttt_lr_eta
    let grad_ttt_lr_eta_val = grad_ttt_lr_eta.cast::<P::EVal>();
    cube::broadcast::store_rv_direct(
        &grad_ttt_lr_eta_val,
        &mut grads.grad_ttt_lr_eta,
        ttt_lr_eta_idx,
    );

    // Atomically add grad_token_eta (shared across batch/head dimensions)
    atomic_add_rv::<_, P::CS>(&grad_token_eta, &mut grads.grad_token_eta, token_eta_base);
}
