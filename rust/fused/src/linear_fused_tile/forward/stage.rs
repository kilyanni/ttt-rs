use cubecl::prelude::*;
use thundercube::{cube::ReduceBuf, prelude::*, reduction_ops::SumOp};

use super::{
    super::{
        helpers::{ParamsTrait, RvbFV, StFF, build_eta_attn_fused, build_eta_matrix},
        layer_norm::{layer_norm_forward, layer_norm_l2_grad},
    },
    types::{Inputs, Outputs},
};

/// Process one mini-batch stage of the TTT-Linear forward pass.
///
/// This is the inner loop body.
/// Weight and bias are kept in shared memory / registers and updated in place.
///
/// # Arguments
/// * `inputs` - Input tensors (xq, xk, xv indexed by stage_offset)
/// * `outputs` - Output tensor (indexed by stage_offset)
/// * `weight_smem` - Weight matrix in shared memory [F, F], updated in place
/// * `bias_rv` - Bias vector in registers [F], updated in place (EVal for tensor I/O)
/// * `ln_weight_rv` - Layer norm weight [F]
/// * `ln_bias_rv` - Layer norm bias [F]
/// * `stage_offset` - Offset into qkv/output for this mini-batch (in elements)
/// * `ttt_lr_eta_idx` - Base offset for ttt_lr_eta
/// * `epsilon` - Layer norm epsilon
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_forward_stage<P: ParamsTrait>(
    inputs: &Inputs<P::EVal>,
    outputs: &mut Outputs<P::EVal>,
    weight_smem: &mut StFF<P>,
    bias_rv: &mut RvbFV<P>,
    ln_weight_rv: &RvbFV<P>,
    ln_bias_rv: &RvbFV<P>,
    stage_offset: usize,
    ttt_lr_eta_idx: usize,
    #[comptime] epsilon: f32,
) {
    let mut q_smem = P::st_f_cs();
    let mut k_smem = P::st_f_cs();
    let mut v_direct_smem = P::st_cs_f();
    let mut z1_smem = P::st_cs_f();
    let mut temp_cs_f_smem = P::st_cs_f();
    let mut eta_matrix_smem = P::st_cs_cs();
    let mut reduce_buf = ReduceBuf::<P::EAcc>::new();

    // Load QKV for this stage
    cube::load_st_transpose(&inputs.xq, &mut q_smem, stage_offset, 0, 0);
    cube::load_st_transpose(&inputs.xk, &mut k_smem, stage_offset, 0, 0);
    cube::load_st_direct(&inputs.xk, &mut temp_cs_f_smem, stage_offset, 0, 0);
    cube::load_st_direct(&inputs.xv, &mut v_direct_smem, stage_offset, 0, 0);

    sync_cube();

    // Step 1: z1 = xk @ W + b
    let mut z1_reg = P::rt_cs_f();
    z1_reg.zero();
    cube::mma_AtB(&mut z1_reg, &k_smem, weight_smem);

    // Add bias (need to broadcast from full bias_rv to the thread's portion, casting EVal -> EAcc)
    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let mut bias_reg = P::rv_f();
    #[unroll]
    for i in 0..P::F_Reg::LINES {
        let src_idx = thread_n * P::F_Reg::LINES + i;
        bias_reg.data[i] = thundercube::util::cast_line(bias_rv.data[src_idx]);
    }
    z1_reg.add_row(&bias_reg);

    cube::store_rt_to_st(&z1_reg, &mut z1_smem);

    // Step 2: reconstruction_target = xv - xk
    v_direct_smem.sub(&temp_cs_f_smem);

    sync_cube();

    // Step 3: grad_l_wrt_z1 = layer_norm_l2_grad(z1, reconstruction_target)
    layer_norm_l2_grad::<P::EVal, P::EAcc, P::CS, P::F>(
        &mut z1_smem,
        &mut v_direct_smem,
        ln_weight_rv,
        ln_bias_rv,
        &mut temp_cs_f_smem,
        &mut reduce_buf,
        epsilon,
    );

    // Step 4: eta_matrix = outer(token_eta, ttt_lr_eta).tril()
    build_eta_matrix::<P>(
        &inputs.token_eta,
        &inputs.ttt_lr_eta,
        &mut eta_matrix_smem,
        ttt_lr_eta_idx,
        false,
    );

    // Step 5: eta @ grad
    let mut eta_grad_reg = P::rt_cs_f();
    eta_grad_reg.zero();
    cube::mma_AB(&mut eta_grad_reg, &eta_matrix_smem, &z1_smem);

    // ILP: z1_bar = xq @ W (independent of eta_attn, computed early to overlap with step 6)
    let mut z1_bar_reg = P::rt_cs_f();
    z1_bar_reg.zero();
    cube::mma_AtB(&mut z1_bar_reg, &q_smem, weight_smem);

    // Step 6: Build (eta * attn) fused directly into eta_matrix
    // eta_matrix[i,j] = token_eta[i] * ttt_lr_eta[j] * (q[i] · k[j])
    build_eta_attn_fused::<P>(
        &q_smem,
        &k_smem,
        &inputs.token_eta,
        &inputs.ttt_lr_eta,
        &mut eta_matrix_smem,
        ttt_lr_eta_idx,
    );

    // (eta * attn) @ grad
    let mut eta_attn_grad_reg = P::rt_cs_f();
    eta_attn_grad_reg.zero();
    cube::mma_AB(&mut eta_attn_grad_reg, &eta_matrix_smem, &z1_smem);

    // z1_bar -= (eta * attn) @ grad
    z1_bar_reg.sub(&eta_attn_grad_reg);

    // z1_bar += bias
    z1_bar_reg.add_row(&bias_reg);

    // z1_bar -= eta @ grad
    z1_bar_reg.sub(&eta_grad_reg);

    // Store z1_bar to shared memory for layer norm
    cube::store_rt_to_st(&z1_bar_reg, &mut temp_cs_f_smem);

    sync_cube();

    // Step 8: layer_norm + add xq
    layer_norm_forward::<P::EVal, P::EAcc, P::CS, P::F>(
        &mut temp_cs_f_smem,
        ln_weight_rv,
        ln_bias_rv,
        &mut reduce_buf,
        epsilon,
    );

    cube::load_st_direct(&inputs.xq, &mut v_direct_smem, stage_offset, 0, 0);

    sync_cube();

    // Add: output = xq + layer_norm(z1_bar)
    temp_cs_f_smem.add(&v_direct_smem);

    sync_cube();

    // Store output for this stage
    cube::store_st_direct(&temp_cs_f_smem, &mut outputs.output, stage_offset, 0, 0);

    // === Steps 9-10: Weight and bias updates (in place) ===
    let last_token_eta_idx = P::CS::VALUE - 1;
    let last_line_idx = last_token_eta_idx / comptime!(LINE_SIZE);
    let last_elem_in_line = last_token_eta_idx % comptime!(LINE_SIZE);
    let token_eta_line = inputs.token_eta[last_line_idx];
    let last_token_eta_scalar = token_eta_line[last_elem_in_line];

    // Load ttt_lr_eta and scale by token_eta[last]
    let mut last_eta_rv = P::rvb_cs_v();
    cube::broadcast::load_rv_direct(&inputs.ttt_lr_eta, &mut last_eta_rv, ttt_lr_eta_idx);
    last_eta_rv.mul_scalar(last_token_eta_scalar);

    // k_smem already holds XK^T from initial load (line 70), never modified
    // Scale in place for weight update (k_smem is dead after this)
    k_smem.mul_row(&last_eta_rv);

    sync_cube();

    // Compute weight_update = scaled_xk^T @ grad = k_smem @ z1_smem
    let mut weight_update_reg = P::rt_ff();
    weight_update_reg.zero();
    cube::mma_AB(&mut weight_update_reg, &k_smem, &z1_smem);

    // Update weight in place: weight -= weight_update
    let mut weight_reg = P::rt_ff();
    cube::load_rt_from_st(weight_smem, &mut weight_reg);
    weight_reg.sub(&weight_update_reg);
    cube::store_rt_to_st(&weight_reg, weight_smem);

    // Bias update: bias -= last_eta^T @ grad
    temp_cs_f_smem.copy_from(&z1_smem);

    sync_cube();

    temp_cs_f_smem.mul_col(&last_eta_rv);

    sync_cube();

    let mut bias_update_rv = P::rvb_f_a();
    cube::reduce_cols::<P::EVal, P::EAcc, P::CS, P::F, SumOp>(
        &temp_cs_f_smem,
        &mut bias_update_rv,
        &mut reduce_buf,
    );

    // Update bias in place
    let bias_update_val = bias_update_rv.cast::<P::EVal>();
    bias_rv.sub(&bias_update_val);
}
