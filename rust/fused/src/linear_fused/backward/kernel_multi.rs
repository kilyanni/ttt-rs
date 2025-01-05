use cubecl::prelude::*;

use super::types::{ForwardInputs, GradOutputs};
use crate::FusedTttConfig;

/// Multi-stage fused TTT-Linear backward kernel.
///
/// Processes all stages in reverse order in a single kernel launch.
/// For each stage:
/// 1. Reconstruct W from nearest checkpoint + forward-simulate intermediate stages
/// 2. Run backward logic for that stage
/// 3. Accumulate gradients across stages
#[cube(launch, launch_unchecked)]
pub fn fused_ttt_backward_kernel_multi<F: Float>(
    inputs: &ForwardInputs<F>,
    grad_output: &Tensor<F>,
    outputs: &mut GradOutputs<F>,
    weight_checkpoints: &Tensor<F>,
    bias_checkpoints: &Tensor<F>,
    num_stages: u32,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;

    let batch_size = inputs.xq.shape(0);
    let num_heads = inputs.xq.shape(1);
    let mini_batch_len = config.mini_batch_len;
    let head_dim = config.head_dim;
    let epsilon = config.epsilon();
    let checkpoint_interval = config.checkpoint_interval;

    let dim_idx = UNIT_POS_X as usize;
    let seq_idx = UNIT_POS_Y as usize;

    // Shared memory for intermediates (one mini-batch at a time)
    let mut z1_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);
    let mut grad_l_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);
    let mut z1_bar_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);
    let mut dl_dz1_bar_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);
    let mut dl_dgrad_l_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);
    let mut dl_d_z1_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);

    let mut z1_mean_shared = SharedMemory::<F>::new(mini_batch_len);
    let mut z1_inv_std_shared = SharedMemory::<F>::new(mini_batch_len);
    let mut z1_bar_mean_shared = SharedMemory::<F>::new(mini_batch_len);
    let mut z1_bar_inv_std_shared = SharedMemory::<F>::new(mini_batch_len);

    // Shared memory for reconstructed W and bias
    let mut w_shared = SharedMemory::<F>::new(head_dim * head_dim);
    let mut bias_shared = SharedMemory::<F>::new(head_dim);

    // Accumulated weight gradient (in shared memory)
    let mut grad_w_shared = SharedMemory::<F>::new(head_dim * head_dim);

    // Accumulated bias gradient (in shared memory, one per dim)
    let mut grad_bias_shared = SharedMemory::<F>::new(head_dim);

    if batch_idx < batch_size && head_idx < num_heads {
        let num_stages_usize = num_stages as usize;
        let full_seq_len = num_stages_usize * mini_batch_len;

        let bh_full_seq_dim_base =
            batch_idx * num_heads * full_seq_len * head_dim + head_idx * full_seq_len * head_dim;
        let bh_dim_dim_base =
            batch_idx * num_heads * head_dim * head_dim + head_idx * head_dim * head_dim;
        let bh_dim_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
        let bh_full_seq_base = batch_idx * num_heads * full_seq_len + head_idx * full_seq_len;
        let h_dim_base = head_idx * head_dim;

        let num_checkpoints = num_stages_usize.div_ceil(checkpoint_interval);
        let ckpt_bh = (batch_idx * num_heads + head_idx) * num_checkpoints;

        // Zero weight and bias gradient accumulators
        if seq_idx == 0 && dim_idx < head_dim {
            for row in 0..head_dim {
                grad_w_shared[row * head_dim + dim_idx] = F::new(0.0);
            }
            grad_bias_shared[dim_idx] = F::new(0.0);
        }

        sync_cube();

        // Process stages in reverse order
        for stage in 0..num_stages {
            let stage_idx = num_stages_usize - 1 - stage as usize;
            let stage_seq_dim_base = bh_full_seq_dim_base + stage_idx * mini_batch_len * head_dim;
            let stage_seq_base = bh_full_seq_base + stage_idx * mini_batch_len;

            // === Reconstruct W[stage_idx] from checkpoint + forward-simulate ===
            let ckpt_stage = (stage_idx / checkpoint_interval) * checkpoint_interval;
            let ckpt_idx = ckpt_stage / checkpoint_interval;

            // Load W from checkpoint
            if seq_idx == 0 && dim_idx < head_dim {
                let ckpt_w_base = (ckpt_bh + ckpt_idx) * head_dim * head_dim;
                for row in 0..head_dim {
                    w_shared[row * head_dim + dim_idx] =
                        weight_checkpoints[ckpt_w_base + row * head_dim + dim_idx];
                }
            }

            // Load b from checkpoint
            if seq_idx == 0 && dim_idx < head_dim {
                let ckpt_b_base = (ckpt_bh + ckpt_idx) * head_dim;
                bias_shared[dim_idx] = bias_checkpoints[ckpt_b_base + dim_idx];
            }

            sync_cube();

            // Forward-simulate from checkpoint to stage_idx
            for fwd in ckpt_stage..stage_idx {
                let fwd_seq_dim_base = bh_full_seq_dim_base + fwd * mini_batch_len * head_dim;
                let fwd_seq_base = bh_full_seq_base + fwd * mini_batch_len;

                // Recompute z1 = xk @ W + b for this forward stage
                if seq_idx < mini_batch_len && dim_idx < head_dim {
                    let xk_base = fwd_seq_dim_base + seq_idx * head_dim;
                    let mut z1_val = F::new(0.0);
                    for k in 0..head_dim {
                        z1_val += inputs.xk[xk_base + k] * w_shared[k * head_dim + dim_idx];
                    }
                    z1_val += bias_shared[dim_idx];
                    z1_shared[seq_idx * head_dim + dim_idx] = z1_val;
                }
                sync_cube();

                // LN stats
                if seq_idx < mini_batch_len && dim_idx == 0 {
                    let mut sum = F::new(0.0);
                    for d in 0..head_dim {
                        sum += z1_shared[seq_idx * head_dim + d];
                    }
                    let mean = sum / F::cast_from(head_dim);
                    let mut var_sum = F::new(0.0);
                    for d in 0..head_dim {
                        let diff = z1_shared[seq_idx * head_dim + d] - mean;
                        var_sum += diff * diff;
                    }
                    let variance = var_sum / F::cast_from(head_dim);
                    let inv_std = F::new(1.0) / F::sqrt(variance + F::new(epsilon));
                    z1_mean_shared[seq_idx] = mean;
                    z1_inv_std_shared[seq_idx] = inv_std;
                }
                sync_cube();

                // grad_l through LN
                if seq_idx < mini_batch_len && dim_idx < head_dim {
                    let idx = fwd_seq_dim_base + seq_idx * head_dim + dim_idx;
                    let mean = z1_mean_shared[seq_idx];
                    let inv_std = z1_inv_std_shared[seq_idx];
                    let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
                    let z1_norm = (z1_val - mean) * inv_std;
                    let ln_w = inputs.ln_weight[h_dim_base + dim_idx];
                    let ln_b = inputs.ln_bias[h_dim_base + dim_idx];
                    let ln_out = ln_w * z1_norm + ln_b;
                    let target = inputs.xv[idx] - inputs.xk[idx];
                    let dl_dnorm = (ln_out - target) * ln_w;
                    grad_l_shared[seq_idx * head_dim + dim_idx] = dl_dnorm;
                }
                sync_cube();

                // LN backward
                if seq_idx < mini_batch_len && dim_idx < head_dim {
                    let mean = z1_mean_shared[seq_idx];
                    let inv_std = z1_inv_std_shared[seq_idx];
                    let mut dl_dnorm_sum = F::new(0.0);
                    let mut dl_dnorm_norm_sum = F::new(0.0);
                    for d in 0..head_dim {
                        let z1_d = z1_shared[seq_idx * head_dim + d];
                        let z1_norm_d = (z1_d - mean) * inv_std;
                        let dl_dnorm_d = grad_l_shared[seq_idx * head_dim + d];
                        dl_dnorm_sum += dl_dnorm_d;
                        dl_dnorm_norm_sum += dl_dnorm_d * z1_norm_d;
                    }
                    let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
                    let z1_norm = (z1_val - mean) * inv_std;
                    let dl_dnorm = grad_l_shared[seq_idx * head_dim + dim_idx];
                    let n = F::cast_from(head_dim);
                    let grad_val =
                        (dl_dnorm * n - dl_dnorm_sum - z1_norm * dl_dnorm_norm_sum) * inv_std / n;
                    grad_l_shared[seq_idx * head_dim + dim_idx] = grad_val;
                }
                sync_cube();

                // Weight/bias update
                let last_seq = mini_batch_len - 1;
                let token_eta_last = inputs.token_eta[last_seq];

                if dim_idx < head_dim && seq_idx == 0 {
                    for row in 0..head_dim {
                        let mut update = F::new(0.0);
                        for k in 0..mini_batch_len {
                            let eta_k = token_eta_last * inputs.ttt_lr_eta[fwd_seq_base + k];
                            let xk_kr = inputs.xk[fwd_seq_dim_base + k * head_dim + row];
                            let grad_kc = grad_l_shared[k * head_dim + dim_idx];
                            update += eta_k * xk_kr * grad_kc;
                        }
                        w_shared[row * head_dim + dim_idx] =
                            w_shared[row * head_dim + dim_idx] - update;
                    }
                }

                if seq_idx == 0 && dim_idx < head_dim {
                    let mut update = F::new(0.0);
                    for k in 0..mini_batch_len {
                        let eta_k = token_eta_last * inputs.ttt_lr_eta[fwd_seq_base + k];
                        let grad_kd = grad_l_shared[k * head_dim + dim_idx];
                        update += eta_k * grad_kd;
                    }
                    bias_shared[dim_idx] = bias_shared[dim_idx] - update;
                }
                sync_cube();
            }
            // W now contains W[stage_idx], bias_shared[dim_idx] contains b[stage_idx]

            // ============================================================
            // Now run the backward pass for this stage, exactly like the
            // single-stage backward but using w_shared/bias_shared[dim_idx] as W/b
            // and stage-offset indices for QKV/eta
            // ============================================================

            let mut grad_xq_local = F::new(0.0);
            let mut grad_xk_local = F::new(0.0);
            let mut grad_xv_local = F::new(0.0);

            // === Recompute forward for this stage ===
            // z1 = xk @ W + b
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let xk_base = stage_seq_dim_base + seq_idx * head_dim;
                let mut z1_val = F::new(0.0);
                for k in 0..head_dim {
                    z1_val += inputs.xk[xk_base + k] * w_shared[k * head_dim + dim_idx];
                }
                z1_val += bias_shared[dim_idx];
                z1_shared[seq_idx * head_dim + dim_idx] = z1_val;
            }
            sync_cube();

            // LN stats for z1
            if seq_idx < mini_batch_len && dim_idx == 0 {
                let mut sum = F::new(0.0);
                for d in 0..head_dim {
                    sum += z1_shared[seq_idx * head_dim + d];
                }
                let mean = sum / F::cast_from(head_dim);
                let mut var_sum = F::new(0.0);
                for d in 0..head_dim {
                    let diff = z1_shared[seq_idx * head_dim + d] - mean;
                    var_sum += diff * diff;
                }
                let variance = var_sum / F::cast_from(head_dim);
                let inv_std = F::new(1.0) / F::sqrt(variance + F::new(epsilon));
                z1_mean_shared[seq_idx] = mean;
                z1_inv_std_shared[seq_idx] = inv_std;
            }
            sync_cube();

            // grad_l
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let idx = stage_seq_dim_base + seq_idx * head_dim + dim_idx;
                let mean = z1_mean_shared[seq_idx];
                let inv_std = z1_inv_std_shared[seq_idx];
                let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
                let z1_norm = (z1_val - mean) * inv_std;
                let ln_w = inputs.ln_weight[h_dim_base + dim_idx];
                let ln_b = inputs.ln_bias[h_dim_base + dim_idx];
                let ln_out = ln_w * z1_norm + ln_b;
                let target = inputs.xv[idx] - inputs.xk[idx];
                let dl_d_z1_norm = (ln_out - target) * ln_w;
                grad_l_shared[seq_idx * head_dim + dim_idx] = dl_d_z1_norm;
            }
            sync_cube();

            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let mean = z1_mean_shared[seq_idx];
                let inv_std = z1_inv_std_shared[seq_idx];
                let mut dl_dnorm_sum = F::new(0.0);
                let mut dl_dnorm_norm_sum = F::new(0.0);
                for d in 0..head_dim {
                    let z1_d = z1_shared[seq_idx * head_dim + d];
                    let z1_norm_d = (z1_d - mean) * inv_std;
                    let dl_dnorm_d = grad_l_shared[seq_idx * head_dim + d];
                    dl_dnorm_sum += dl_dnorm_d;
                    dl_dnorm_norm_sum += dl_dnorm_d * z1_norm_d;
                }
                let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
                let z1_norm = (z1_val - mean) * inv_std;
                let dl_dnorm = grad_l_shared[seq_idx * head_dim + dim_idx];
                let n = F::cast_from(head_dim);
                let grad_l_val =
                    (dl_dnorm * n - dl_dnorm_sum - z1_norm * dl_dnorm_norm_sum) * inv_std / n;
                grad_l_shared[seq_idx * head_dim + dim_idx] = grad_l_val;
            }
            sync_cube();

            // z1_bar recomputation
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let token_eta_i = inputs.token_eta[seq_idx];
                let xq_base = stage_seq_dim_base + seq_idx * head_dim;

                let mut eta_grad_sum = F::new(0.0);
                for j in 0..mini_batch_len {
                    if j <= seq_idx {
                        let eta_ij = token_eta_i * inputs.ttt_lr_eta[stage_seq_base + j];
                        eta_grad_sum += eta_ij * grad_l_shared[j * head_dim + dim_idx];
                    }
                }
                let b1_bar = bias_shared[dim_idx] - eta_grad_sum;

                let mut xq_w = F::new(0.0);
                for k in 0..head_dim {
                    xq_w += inputs.xq[xq_base + k] * w_shared[k * head_dim + dim_idx];
                }

                let mut correction = F::new(0.0);
                for j in 0..mini_batch_len {
                    if j <= seq_idx {
                        let xk_base_j = stage_seq_dim_base + j * head_dim;
                        let mut attn_ij = F::new(0.0);
                        for k in 0..head_dim {
                            attn_ij += inputs.xq[xq_base + k] * inputs.xk[xk_base_j + k];
                        }
                        let eta_ij = token_eta_i * inputs.ttt_lr_eta[stage_seq_base + j];
                        correction += eta_ij * attn_ij * grad_l_shared[j * head_dim + dim_idx];
                    }
                }

                let z1_bar = xq_w - correction + b1_bar;
                z1_bar_shared[seq_idx * head_dim + dim_idx] = z1_bar;
            }
            sync_cube();

            // LN stats for z1_bar
            if seq_idx < mini_batch_len && dim_idx == 0 {
                let mut sum = F::new(0.0);
                for d in 0..head_dim {
                    sum += z1_bar_shared[seq_idx * head_dim + d];
                }
                let mean = sum / F::cast_from(head_dim);
                let mut var_sum = F::new(0.0);
                for d in 0..head_dim {
                    let diff = z1_bar_shared[seq_idx * head_dim + d] - mean;
                    var_sum += diff * diff;
                }
                let variance = var_sum / F::cast_from(head_dim);
                let inv_std = F::new(1.0) / F::sqrt(variance + F::new(epsilon));
                z1_bar_mean_shared[seq_idx] = mean;
                z1_bar_inv_std_shared[seq_idx] = inv_std;
            }
            sync_cube();

            // === Backward through output = xq + LN(z1_bar) ===
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let idx = stage_seq_dim_base + seq_idx * head_dim + dim_idx;
                let grad_out = grad_output[idx];
                grad_xq_local = grad_out;

                let ln_w = inputs.ln_weight[h_dim_base + dim_idx];
                let dl_d_z1_bar_norm = grad_out * ln_w;
                dl_dz1_bar_shared[seq_idx * head_dim + dim_idx] = dl_d_z1_bar_norm;
            }
            sync_cube();

            // LN backward for z1_bar
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let mean = z1_bar_mean_shared[seq_idx];
                let inv_std = z1_bar_inv_std_shared[seq_idx];
                let mut dl_dnorm_sum = F::new(0.0);
                let mut dl_dnorm_norm_sum = F::new(0.0);
                for d in 0..head_dim {
                    let z1_bar_d = z1_bar_shared[seq_idx * head_dim + d];
                    let z1_bar_norm_d = (z1_bar_d - mean) * inv_std;
                    let dl_dnorm_d = dl_dz1_bar_shared[seq_idx * head_dim + d];
                    dl_dnorm_sum += dl_dnorm_d;
                    dl_dnorm_norm_sum += dl_dnorm_d * z1_bar_norm_d;
                }
                let z1_bar_val = z1_bar_shared[seq_idx * head_dim + dim_idx];
                let z1_bar_norm = (z1_bar_val - mean) * inv_std;
                let dl_dnorm = dl_dz1_bar_shared[seq_idx * head_dim + dim_idx];
                let n = F::cast_from(head_dim);
                let dl_dz1_bar =
                    (dl_dnorm * n - dl_dnorm_sum - z1_bar_norm * dl_dnorm_norm_sum) * inv_std / n;
                dl_dz1_bar_shared[seq_idx * head_dim + dim_idx] = dl_dz1_bar;
            }
            sync_cube();

            // dL/d_xq += dL/d_z1_bar @ W^T (from z1_bar = xq @ W)
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                for k in 0..head_dim {
                    let dl_dz1_bar_k = dl_dz1_bar_shared[seq_idx * head_dim + k];
                    let w_val = w_shared[dim_idx * head_dim + k];
                    grad_xq_local += dl_dz1_bar_k * w_val;
                }
            }
            sync_cube();

            // dL/d_grad_l (includes chain rule from accumulated weight/bias gradient)
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let mut dl_dgrad_l_val = F::new(0.0);
                let j = seq_idx;
                for i in j..mini_batch_len {
                    let token_eta_i = inputs.token_eta[i];
                    let eta_ij = token_eta_i * inputs.ttt_lr_eta[stage_seq_base + j];

                    let xq_base_i = stage_seq_dim_base + i * head_dim;
                    let xk_base_j = stage_seq_dim_base + j * head_dim;
                    let mut attn_ij = F::new(0.0);
                    for k in 0..head_dim {
                        attn_ij += inputs.xq[xq_base_i + k] * inputs.xk[xk_base_j + k];
                    }

                    let dl_dz1_bar_id = dl_dz1_bar_shared[i * head_dim + dim_idx];
                    dl_dgrad_l_val -= eta_ij * dl_dz1_bar_id;
                    dl_dgrad_l_val -= eta_ij * attn_ij * dl_dz1_bar_id;
                }

                // Chain rule: gradient flowing back through the weight update
                // W_next = W - sum_k(last_eta[k] * outer(xk[k], grad_l[k]))
                // dL/d_grad_l[j,d] += -last_eta[j] * sum_r(xk[j,r] * grad_W_acc[r,d])
                let last_seq = mini_batch_len - 1;
                let token_eta_last = inputs.token_eta[last_seq];
                let last_eta_j = token_eta_last * inputs.ttt_lr_eta[stage_seq_base + j];
                let xk_base_j = stage_seq_dim_base + j * head_dim;

                let mut xk_dot_grad_w = F::new(0.0);
                for r in 0..head_dim {
                    xk_dot_grad_w +=
                        inputs.xk[xk_base_j + r] * grad_w_shared[r * head_dim + dim_idx];
                }
                dl_dgrad_l_val -= last_eta_j * xk_dot_grad_w;

                // dL/d_grad_l[j,d] += -last_eta[j] * grad_b_acc[d]
                dl_dgrad_l_val -= last_eta_j * grad_bias_shared[dim_idx];

                dl_dgrad_l_shared[seq_idx * head_dim + dim_idx] = dl_dgrad_l_val;
            }
            sync_cube();

            // Backprop through LN to get dL/d_z1
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let mean = z1_mean_shared[seq_idx];
                let inv_std = z1_inv_std_shared[seq_idx];
                let ln_w = inputs.ln_weight[h_dim_base + dim_idx];

                let mut dl_dgrad_l_sum = F::new(0.0);
                let mut dl_dgrad_l_znorm_sum = F::new(0.0);
                for d in 0..head_dim {
                    let z1_d = z1_shared[seq_idx * head_dim + d];
                    let z1_norm_d = (z1_d - mean) * inv_std;
                    let dl_dgrad_l_d = dl_dgrad_l_shared[seq_idx * head_dim + d];
                    dl_dgrad_l_sum += dl_dgrad_l_d;
                    dl_dgrad_l_znorm_sum += dl_dgrad_l_d * z1_norm_d;
                }

                let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
                let z1_norm = (z1_val - mean) * inv_std;
                let dl_dgrad_l = dl_dgrad_l_shared[seq_idx * head_dim + dim_idx];
                let n = F::cast_from(head_dim);

                let dl_d_dl_dnorm =
                    (dl_dgrad_l * n - dl_dgrad_l_sum - z1_norm * dl_dgrad_l_znorm_sum) * inv_std
                        / n;
                let dl_d_dl_dln_out = dl_d_dl_dnorm * ln_w;

                let dl_d_target = -dl_d_dl_dln_out;
                grad_xv_local = dl_d_target;
                grad_xk_local = -dl_d_target;

                let dl_d_z1_norm_via_dl_dnorm = dl_d_dl_dln_out * ln_w;

                let mut dl_dnorm_znorm_sum = F::new(0.0);
                let mut dl_dgrad_l_znorm_sum_for_path_b = F::new(0.0);
                for d in 0..head_dim {
                    let z1_d = z1_shared[seq_idx * head_dim + d];
                    let z1_norm_d = (z1_d - mean) * inv_std;
                    let ln_w_d = inputs.ln_weight[h_dim_base + d];
                    let ln_b_d = inputs.ln_bias[h_dim_base + d];
                    let ln_out_d = ln_w_d * z1_norm_d + ln_b_d;
                    let target_d = inputs.xv[stage_seq_dim_base + seq_idx * head_dim + d]
                        - inputs.xk[stage_seq_dim_base + seq_idx * head_dim + d];
                    let dl_dnorm_d = (ln_out_d - target_d) * ln_w_d;
                    dl_dnorm_znorm_sum += dl_dnorm_d * z1_norm_d;
                    dl_dgrad_l_znorm_sum_for_path_b +=
                        dl_dgrad_l_shared[seq_idx * head_dim + d] * z1_norm_d;
                }

                let dl_dgrad_l_here = dl_dgrad_l_shared[seq_idx * head_dim + dim_idx];
                let ln_w_here = inputs.ln_weight[h_dim_base + dim_idx];
                let ln_b_here = inputs.ln_bias[h_dim_base + dim_idx];
                let ln_out_here = ln_w_here * z1_norm + ln_b_here;
                let target_here = inputs.xv[stage_seq_dim_base + seq_idx * head_dim + dim_idx]
                    - inputs.xk[stage_seq_dim_base + seq_idx * head_dim + dim_idx];
                let dl_dnorm_here = (ln_out_here - target_here) * ln_w_here;
                let n = F::cast_from(head_dim);

                let dl_d_z1_norm_via_direct = (-dl_dgrad_l_here * dl_dnorm_znorm_sum
                    - dl_dgrad_l_znorm_sum_for_path_b * dl_dnorm_here)
                    * inv_std
                    / n;

                let dl_d_z1_norm = dl_d_z1_norm_via_dl_dnorm + dl_d_z1_norm_via_direct;
                dl_d_z1_shared[seq_idx * head_dim + dim_idx] = dl_d_z1_norm;
            }
            sync_cube();

            // Complete dL/d_z1
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let mean = z1_mean_shared[seq_idx];
                let inv_std = z1_inv_std_shared[seq_idx];

                let mut dl_d_z1_norm_sum = F::new(0.0);
                let mut dl_d_z1_norm_znorm_sum = F::new(0.0);
                let mut dl_dgrad_l_times_grad_l_sum = F::new(0.0);
                for d in 0..head_dim {
                    let z1_d = z1_shared[seq_idx * head_dim + d];
                    let z1_norm_d = (z1_d - mean) * inv_std;
                    let dl_d_z1_norm_d = dl_d_z1_shared[seq_idx * head_dim + d];
                    dl_d_z1_norm_sum += dl_d_z1_norm_d;
                    dl_d_z1_norm_znorm_sum += dl_d_z1_norm_d * z1_norm_d;
                    let dl_dgrad_l_d = dl_dgrad_l_shared[seq_idx * head_dim + d];
                    let grad_l_d = grad_l_shared[seq_idx * head_dim + d];
                    dl_dgrad_l_times_grad_l_sum += dl_dgrad_l_d * grad_l_d;
                }

                let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
                let z1_norm = (z1_val - mean) * inv_std;
                let dl_d_z1_norm_val = dl_d_z1_shared[seq_idx * head_dim + dim_idx];
                let n = F::cast_from(head_dim);

                let dl_d_z1_from_norm =
                    (dl_d_z1_norm_val * n - dl_d_z1_norm_sum - z1_norm * dl_d_z1_norm_znorm_sum)
                        * inv_std
                        / n;
                let dl_d_z1_via_std = -dl_dgrad_l_times_grad_l_sum * z1_norm * inv_std / n;

                let dl_d_z1 = dl_d_z1_from_norm + dl_d_z1_via_std;
                dl_d_z1_shared[seq_idx * head_dim + dim_idx] = dl_d_z1;
            }
            sync_cube();

            // dL/d_xq from attn
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let i = seq_idx;
                let token_eta_i = inputs.token_eta[i];
                for j in 0..mini_batch_len {
                    if j <= i {
                        let eta_ij = token_eta_i * inputs.ttt_lr_eta[stage_seq_base + j];
                        let mut grad_attn_contrib = F::new(0.0);
                        for d in 0..head_dim {
                            grad_attn_contrib += grad_l_shared[j * head_dim + d]
                                * dl_dz1_bar_shared[i * head_dim + d];
                        }
                        let dl_d_attn_ij = -eta_ij * grad_attn_contrib;
                        let xk_jd = inputs.xk[stage_seq_dim_base + j * head_dim + dim_idx];
                        grad_xq_local += dl_d_attn_ij * xk_jd;
                    }
                }
            }
            sync_cube();

            // dL/d_xk from attn
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let j = seq_idx;
                for i in j..mini_batch_len {
                    let token_eta_i = inputs.token_eta[i];
                    let eta_ij = token_eta_i * inputs.ttt_lr_eta[stage_seq_base + j];
                    let mut grad_attn_contrib = F::new(0.0);
                    for d in 0..head_dim {
                        grad_attn_contrib +=
                            grad_l_shared[j * head_dim + d] * dl_dz1_bar_shared[i * head_dim + d];
                    }
                    let dl_d_attn_ij = -eta_ij * grad_attn_contrib;
                    let xq_id = inputs.xq[stage_seq_dim_base + i * head_dim + dim_idx];
                    grad_xk_local += dl_d_attn_ij * xq_id;
                }
            }
            sync_cube();

            // dL/d_xk from z1 = xk @ W
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                for k in 0..head_dim {
                    let dl_d_z1_k = dl_d_z1_shared[seq_idx * head_dim + k];
                    let w_val = w_shared[dim_idx * head_dim + k];
                    grad_xk_local += dl_d_z1_k * w_val;
                }
            }

            // Chain rule: dL/d_xk from weight update
            // dL/d_xk[j,r] += -last_eta[j] * sum_d(grad_l[j,d] * grad_W_acc[r,d])
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let j = seq_idx;
                let last_seq = mini_batch_len - 1;
                let token_eta_last = inputs.token_eta[last_seq];
                let last_eta_j = token_eta_last * inputs.ttt_lr_eta[stage_seq_base + j];

                let mut grad_l_dot_w = F::new(0.0);
                for d in 0..head_dim {
                    grad_l_dot_w +=
                        grad_l_shared[j * head_dim + d] * grad_w_shared[dim_idx * head_dim + d];
                }
                grad_xk_local -= last_eta_j * grad_l_dot_w;
            }
            sync_cube();

            // Write xq, xk, xv gradients for this stage
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let idx = stage_seq_dim_base + seq_idx * head_dim + dim_idx;
                outputs.xq[idx] = grad_xq_local;
                outputs.xk[idx] = grad_xk_local;
                outputs.xv[idx] = grad_xv_local;
            }
            sync_cube();

            // ttt_lr_eta gradient (includes chain rule from weight update)
            // NOTE: This must come BEFORE the weight/bias gradient accumulation,
            // because the chain rule terms use grad_w_shared/grad_bias_shared which
            // should only contain contributions from LATER stages, not the current one.
            if seq_idx < mini_batch_len && dim_idx == 0 {
                let j = seq_idx;
                let mut grad_lr = F::new(0.0);
                for i in j..mini_batch_len {
                    let token_eta_i = inputs.token_eta[i];
                    let xq_base_i = stage_seq_dim_base + i * head_dim;
                    let xk_base_j = stage_seq_dim_base + j * head_dim;
                    let mut attn_ij = F::new(0.0);
                    for k in 0..head_dim {
                        attn_ij += inputs.xq[xq_base_i + k] * inputs.xk[xk_base_j + k];
                    }
                    let mut grad_dot = F::new(0.0);
                    for d in 0..head_dim {
                        grad_dot +=
                            grad_l_shared[j * head_dim + d] * dl_dz1_bar_shared[i * head_dim + d];
                    }
                    let dl_d_eta_ij = -(F::new(1.0) + attn_ij) * grad_dot;
                    grad_lr += dl_d_eta_ij * token_eta_i;
                }

                // Chain rule: dL/d_ttt_lr_eta[j] from weight/bias update
                let last_seq = mini_batch_len - 1;
                let token_eta_last = inputs.token_eta[last_seq];
                let xk_base_j = stage_seq_dim_base + j * head_dim;
                let mut weight_update_contrib = F::new(0.0);
                for d in 0..head_dim {
                    let grad_l_jd = grad_l_shared[j * head_dim + d];
                    // From W update: -xk[j,r] * grad_l[j,d] * grad_W_acc[r,d]
                    let mut xk_grad_w = F::new(0.0);
                    for r in 0..head_dim {
                        xk_grad_w += inputs.xk[xk_base_j + r] * grad_w_shared[r * head_dim + d];
                    }
                    weight_update_contrib -= grad_l_jd * xk_grad_w;
                    // From b update: -grad_l[j,d] * grad_b_acc[d]
                    weight_update_contrib -= grad_l_jd * grad_bias_shared[d];
                }
                grad_lr += token_eta_last * weight_update_contrib;

                outputs.ttt_lr_eta[stage_seq_base + j] = grad_lr;
            }
            sync_cube();

            // token_eta gradient (atomic — offset by stage, includes chain rule)
            if seq_idx < mini_batch_len && dim_idx == 0 {
                let i = seq_idx;
                let mut grad_te = F::new(0.0);
                for j in 0..mini_batch_len {
                    if j <= i {
                        let xq_base_i = stage_seq_dim_base + i * head_dim;
                        let xk_base_j = stage_seq_dim_base + j * head_dim;
                        let mut attn_ij = F::new(0.0);
                        for k in 0..head_dim {
                            attn_ij += inputs.xq[xq_base_i + k] * inputs.xk[xk_base_j + k];
                        }
                        let mut grad_dot = F::new(0.0);
                        for d in 0..head_dim {
                            grad_dot += grad_l_shared[j * head_dim + d]
                                * dl_dz1_bar_shared[i * head_dim + d];
                        }
                        let dl_d_eta_ij = -(F::new(1.0) + attn_ij) * grad_dot;
                        grad_te += dl_d_eta_ij * inputs.ttt_lr_eta[stage_seq_base + j];
                    }
                }

                // Chain rule: token_eta[last_seq] contribution from weight/bias update
                let last_seq = mini_batch_len - 1;
                if i == last_seq {
                    let mut weight_update_token_eta_grad = F::new(0.0);
                    for j in 0..mini_batch_len {
                        let lr_j = inputs.ttt_lr_eta[stage_seq_base + j];
                        let xk_base_j = stage_seq_dim_base + j * head_dim;
                        let mut contrib = F::new(0.0);
                        for d in 0..head_dim {
                            let grad_l_jd = grad_l_shared[j * head_dim + d];
                            let mut xk_grad_w = F::new(0.0);
                            for r in 0..head_dim {
                                xk_grad_w +=
                                    inputs.xk[xk_base_j + r] * grad_w_shared[r * head_dim + d];
                            }
                            contrib -= grad_l_jd * xk_grad_w;
                            contrib -= grad_l_jd * grad_bias_shared[d];
                        }
                        weight_update_token_eta_grad += lr_j * contrib;
                    }
                    grad_te += weight_update_token_eta_grad;
                }

                outputs.token_eta[stage_idx * mini_batch_len + i]
                    .fetch_add(f32::cast_from(grad_te));
            }
            sync_cube();

            // Accumulate weight gradient for chain rule in earlier stages
            // NOTE: This must come AFTER ttt_lr_eta and token_eta gradient computation,
            // so those gradients only see contributions from later stages.
            if dim_idx < head_dim && seq_idx == 0 {
                for col in 0..head_dim {
                    let mut grad_w_xq = F::new(0.0);
                    let mut grad_w_xk = F::new(0.0);
                    for s in 0..mini_batch_len {
                        let xq_val = inputs.xq[stage_seq_dim_base + s * head_dim + dim_idx];
                        let dl_dz1_bar = dl_dz1_bar_shared[s * head_dim + col];
                        grad_w_xq += xq_val * dl_dz1_bar;
                        let xk_val = inputs.xk[stage_seq_dim_base + s * head_dim + dim_idx];
                        let dl_dz1 = dl_d_z1_shared[s * head_dim + col];
                        grad_w_xk += xk_val * dl_dz1;
                    }
                    grad_w_shared[dim_idx * head_dim + col] =
                        grad_w_shared[dim_idx * head_dim + col] + grad_w_xq + grad_w_xk;
                }

                let mut grad_b = F::new(0.0);
                for s in 0..mini_batch_len {
                    grad_b += dl_dz1_bar_shared[s * head_dim + dim_idx];
                    grad_b += dl_d_z1_shared[s * head_dim + dim_idx];
                }
                grad_bias_shared[dim_idx] += grad_b;
            }
            sync_cube();

            // LN weight/bias gradients (accumulate per stage)
            if seq_idx == 0 && dim_idx < head_dim {
                let mut grad_ln_w = F::new(0.0);
                let mut grad_ln_b = F::new(0.0);
                for s in 0..mini_batch_len {
                    let mean = z1_bar_mean_shared[s];
                    let inv_std = z1_bar_inv_std_shared[s];
                    let z1_bar_val = z1_bar_shared[s * head_dim + dim_idx];
                    let z1_bar_norm = (z1_bar_val - mean) * inv_std;
                    let grad_out = grad_output[stage_seq_dim_base + s * head_dim + dim_idx];
                    grad_ln_w += grad_out * z1_bar_norm;
                    grad_ln_b += grad_out;
                }
                outputs.ln_weight[h_dim_base + dim_idx].fetch_add(f32::cast_from(grad_ln_w));
                outputs.ln_bias[h_dim_base + dim_idx].fetch_add(f32::cast_from(grad_ln_b));
            }
            sync_cube();
        }

        // Write accumulated weight and bias gradients
        if seq_idx == 0 && dim_idx < head_dim {
            for row in 0..head_dim {
                outputs.weight[bh_dim_dim_base + row * head_dim + dim_idx] =
                    grad_w_shared[row * head_dim + dim_idx];
            }
            outputs.bias[bh_dim_base + dim_idx] = grad_bias_shared[dim_idx];
        }
    }
}
