use cubecl::prelude::*;
use thundercube::{prelude::*, util::index_2d};

use super::{
    super::helpers::ParamsTrait,
    stage::fused_ttt_forward_stage,
    types::{Inputs, Outputs},
};
use crate::FusedTttConfig;

/// Fused TTT-Linear forward pass kernel (single mini-batch).
///
/// Each CUBE handles one (batch, head) pair.
/// Computes the TTT-Linear forward pass with online weight updates.
///
/// Algorithm:
/// 1. z1 = xk @ W + b
/// 2. reconstruction_target = xv - xk
/// 3. grad_l_wrt_z1 = layer_norm_l2_grad(z1, reconstruction_target)
/// 4. eta_matrix = outer(token_eta, ttt_lr_eta).tril()
/// 5. attn_scores = xq @ xk^T, attn1 = attn_scores.tril()
/// 6. b1_bar = bias - eta_matrix @ grad_l_wrt_z1
/// 7. z1_bar = xq @ W - (eta_matrix * attn1) @ grad_l_wrt_z1 + b1_bar
/// 8. output = xq + layer_norm(z1_bar)
/// 9. weight_out = weight - (last_eta_col * xk).T @ grad
/// 10. bias_out = bias - sum_rows(last_eta_col * grad)
#[cube(launch, launch_unchecked)]
pub fn fused_ttt_forward_kernel<P: ParamsTrait>(
    inputs: &Inputs<P::EVal>,
    outputs: &mut Outputs<P::EVal>,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let epsilon = comptime!(config.epsilon());

    // Compute base offsets
    let base_qkv = index_2d(&inputs.xq, batch_idx, head_idx);
    let base_weight = index_2d(&inputs.weight, batch_idx, head_idx);
    let base_bias = index_2d(&inputs.bias, batch_idx, head_idx);
    let ttt_lr_eta_idx = index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx);

    // Initialize weight in shared memory
    let mut weight_smem = P::st_ff();
    cube::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

    sync_cube();

    // Initialize bias in register vector (EVal for tensor I/O)
    let mut bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&inputs.bias, &mut bias_rv, base_bias);

    // Load layer norm params (EVal to match St types in layer_norm functions)
    let base_ln = index_2d(&inputs.ln_weight, head_idx, 0);
    let mut ln_weight_rv = P::rvb_f_v();
    let mut ln_bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
    cube::broadcast::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

    // Process single stage
    fused_ttt_forward_stage::<P>(
        inputs,
        outputs,
        &mut weight_smem,
        &mut bias_rv,
        &ln_weight_rv,
        &ln_bias_rv,
        base_qkv,
        ttt_lr_eta_idx,
        epsilon,
    );

    sync_cube();

    // Store final weight and bias
    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);

    cube::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);
    cube::broadcast::store_rv_direct(&bias_rv, &mut outputs.bias_out, base_bias_out);
}

/// Fused TTT-Linear forward pass kernel with multiple mini-batch stages.
///
/// Processes `num_stages` mini-batches in a single kernel launch, keeping
/// weight and bias in shared memory between stages to avoid global memory
/// round-trips.
///
/// Input tensors xq, xk, xv should have shape [batch, heads, num_stages * mini_batch_len, head_dim]
/// Output tensor should have the same shape.
#[cube(launch, launch_unchecked)]
pub fn fused_ttt_forward_kernel_multi<P: ParamsTrait>(
    inputs: &Inputs<P::EVal>,
    outputs: &mut Outputs<P::EVal>,
    weight_checkpoints: &mut Tensor<Line<P::EVal>>,
    bias_checkpoints: &mut Tensor<Line<P::EVal>>,
    num_stages: u32,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;
    let num_heads = inputs.xq.shape(1);
    let epsilon = comptime!(config.epsilon());
    let mini_batch_len = comptime!(config.mini_batch_len);
    let head_dim = comptime!(config.head_dim);

    // Stride to advance by one mini-batch in the sequence dimension (scalars)
    let stage_stride = mini_batch_len * head_dim;

    // Compute base offsets
    let base_qkv = index_2d(&inputs.xq, batch_idx, head_idx);
    let base_weight = index_2d(&inputs.weight, batch_idx, head_idx);
    let base_bias = index_2d(&inputs.bias, batch_idx, head_idx);
    let ttt_lr_eta_idx = index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx);

    // Checkpoint layout: one checkpoint per `checkpoint_interval` stages
    let checkpoint_interval = comptime!(config.checkpoint_interval);
    let num_stages_usize = num_stages as usize;
    let num_checkpoints = num_stages_usize.div_ceil(checkpoint_interval);
    let ckpt_bh = (batch_idx * num_heads + head_idx) * num_checkpoints;

    // Initialize weight in shared memory
    let mut weight_smem = P::st_ff();
    cube::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

    sync_cube();

    // Initialize bias in register vector
    let mut bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&inputs.bias, &mut bias_rv, base_bias);

    // Load layer norm params
    let base_ln = index_2d(&inputs.ln_weight, head_idx, 0);
    let mut ln_weight_rv = P::rvb_f_v();
    let mut ln_bias_rv = P::rvb_f_v();
    cube::broadcast::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
    cube::broadcast::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

    // Process all stages
    for stage in 0..num_stages {
        let stage_usize = stage as usize;

        // Checkpoint weight/bias BEFORE this stage's update (every checkpoint_interval stages)
        if stage_usize.is_multiple_of(checkpoint_interval) {
            let ckpt_idx = stage_usize / checkpoint_interval;
            let ckpt_weight_offset = (ckpt_bh + ckpt_idx) * (head_dim * head_dim);
            let ckpt_bias_offset = (ckpt_bh + ckpt_idx) * head_dim;
            cube::store_st_direct(&weight_smem, weight_checkpoints, ckpt_weight_offset, 0, 0);
            cube::broadcast::store_rv_direct(&bias_rv, bias_checkpoints, ckpt_bias_offset);
        }

        let stage_offset = base_qkv + stage_usize * stage_stride;
        let ttt_lr_offset = ttt_lr_eta_idx + stage_usize * mini_batch_len;

        fused_ttt_forward_stage::<P>(
            inputs,
            outputs,
            &mut weight_smem,
            &mut bias_rv,
            &ln_weight_rv,
            &ln_bias_rv,
            stage_offset,
            ttt_lr_offset,
            epsilon,
        );

        sync_cube();
    }

    // Store final weight and bias
    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);

    cube::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);
    cube::broadcast::store_rv_direct(&bias_rv, &mut outputs.bias_out, base_bias_out);
}
