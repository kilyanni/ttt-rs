use cubecl::prelude::*;

use crate::FusedTttConfig;

/// Multi-stage fused TTT-Linear forward kernel.
///
/// Processes all stages in a single kernel launch, keeping W and b in shared memory
/// across stages. Checkpoints W/b at regular intervals for backward pass.
///
/// cube_dim = (head_dim, mini_batch_len) — same threads process each stage sequentially.
///
/// Extra inputs:
/// - weight_checkpoints: [batch*heads*num_checkpoints, head_dim, head_dim]
/// - bias_checkpoints: [batch*heads*num_checkpoints, head_dim]
/// - num_stages: number of mini-batches to process
#[cube(launch, launch_unchecked)]
pub fn fused_ttt_forward_kernel_multi<F: Float>(
    // Inputs
    xq: &Tensor<F>,
    xk: &Tensor<F>,
    xv: &Tensor<F>,
    token_eta: &Tensor<F>,
    ttt_lr_eta: &Tensor<F>,
    ln_weight: &Tensor<F>,
    ln_bias: &Tensor<F>,
    // State (read-only initial)
    weight_init: &Tensor<F>,
    bias_init: &Tensor<F>,
    // Outputs
    weight_out: &mut Tensor<F>,
    bias_out: &mut Tensor<F>,
    output: &mut Tensor<F>,
    // Checkpoints
    weight_checkpoints: &mut Tensor<F>,
    bias_checkpoints: &mut Tensor<F>,
    num_stages: u32,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;

    let batch_size = xq.shape(0);
    let num_heads = xq.shape(1);
    let mini_batch_len = config.mini_batch_len;
    let head_dim = config.head_dim;
    let epsilon = config.epsilon();
    let checkpoint_interval = config.checkpoint_interval;

    let dim_idx = UNIT_POS_X as usize;
    let seq_idx = UNIT_POS_Y as usize;

    // Shared memory for intermediates (sized for one mini-batch)
    let mut z1_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);
    let mut grad_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);
    let mut z1_bar_shared = SharedMemory::<F>::new(mini_batch_len * head_dim);
    let mut ln_mean_shared = SharedMemory::<F>::new(mini_batch_len);
    let mut ln_inv_std_shared = SharedMemory::<F>::new(mini_batch_len);

    // Shared memory for W [head_dim, head_dim] — persists across stages
    let mut w_shared = SharedMemory::<F>::new(head_dim * head_dim);
    // Shared memory for bias [head_dim] — persists across stages
    let mut bias_shared = SharedMemory::<F>::new(head_dim);

    if batch_idx < batch_size && head_idx < num_heads {
        let num_stages_usize = num_stages as usize;

        // Full-sequence bases (xq/xk/xv have shape [B, H, seq_len, D])
        let full_seq_len = num_stages_usize * mini_batch_len;
        let bh_full_seq_dim_base =
            batch_idx * num_heads * full_seq_len * head_dim + head_idx * full_seq_len * head_dim;
        let bh_dim_dim_base =
            batch_idx * num_heads * head_dim * head_dim + head_idx * head_dim * head_dim;
        let bh_dim_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
        let bh_full_seq_base = batch_idx * num_heads * full_seq_len + head_idx * full_seq_len;
        let h_dim_base = head_idx * head_dim;

        // Checkpoint layout
        let num_checkpoints = num_stages_usize.div_ceil(checkpoint_interval);
        let ckpt_bh = (batch_idx * num_heads + head_idx) * num_checkpoints;

        // Load initial W into shared memory
        if seq_idx == 0 && dim_idx < head_dim {
            for row in 0..head_dim {
                w_shared[row * head_dim + dim_idx] =
                    weight_init[bh_dim_dim_base + row * head_dim + dim_idx];
            }
        }

        // Load initial bias
        if seq_idx == 0 && dim_idx < head_dim {
            bias_shared[dim_idx] = bias_init[bh_dim_base + dim_idx];
        }

        sync_cube();

        // Process each stage
        for stage in 0..num_stages {
            let stage_usize = stage as usize;

            // Checkpoint W and b before this stage's update
            if stage_usize.is_multiple_of(checkpoint_interval) {
                let ckpt_idx = stage_usize / checkpoint_interval;

                // Store W checkpoint
                if seq_idx == 0 && dim_idx < head_dim {
                    let ckpt_w_base = (ckpt_bh + ckpt_idx) * head_dim * head_dim;
                    for row in 0..head_dim {
                        weight_checkpoints[ckpt_w_base + row * head_dim + dim_idx] =
                            w_shared[row * head_dim + dim_idx];
                    }
                }

                // Store b checkpoint
                if seq_idx == 0 && dim_idx < head_dim {
                    let ckpt_b_base = (ckpt_bh + ckpt_idx) * head_dim;
                    bias_checkpoints[ckpt_b_base + dim_idx] = bias_shared[dim_idx];
                }

                sync_cube();
            }

            // Compute offsets for this stage
            let stage_seq_dim_base = bh_full_seq_dim_base + stage_usize * mini_batch_len * head_dim;
            let stage_seq_base = bh_full_seq_base + stage_usize * mini_batch_len;

            // === z1 = xk @ W + b ===
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let xk_base = stage_seq_dim_base + seq_idx * head_dim;
                let mut z1_val = F::new(0.0);

                for k in 0..head_dim {
                    let xk_val = xk[xk_base + k];
                    let w_val = w_shared[k * head_dim + dim_idx];
                    z1_val += xk_val * w_val;
                }
                z1_val += bias_shared[dim_idx];

                z1_shared[seq_idx * head_dim + dim_idx] = z1_val;
            }

            sync_cube();

            // === Layer norm stats for z1 ===
            if seq_idx < mini_batch_len {
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

                if dim_idx == 0 {
                    ln_mean_shared[seq_idx] = mean;
                    ln_inv_std_shared[seq_idx] = inv_std;
                }
            }

            sync_cube();

            // === grad = d/dz1 of ||LN(z1) - target||^2 ===
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let idx = stage_seq_dim_base + seq_idx * head_dim + dim_idx;
                let mean = ln_mean_shared[seq_idx];
                let inv_std = ln_inv_std_shared[seq_idx];

                let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
                let z1_norm = (z1_val - mean) * inv_std;

                let ln_w = ln_weight[h_dim_base + dim_idx];
                let ln_b = ln_bias[h_dim_base + dim_idx];
                let ln_out = ln_w * z1_norm + ln_b;

                let target = xv[idx] - xk[idx];
                let dl_dout = ln_out - target;
                let dl_dnorm = dl_dout * ln_w;

                grad_shared[seq_idx * head_dim + dim_idx] = dl_dnorm;
            }

            sync_cube();

            // === Layer norm backward for grad ===
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let mean = ln_mean_shared[seq_idx];
                let inv_std = ln_inv_std_shared[seq_idx];

                let mut dl_dnorm_sum = F::new(0.0);
                let mut dl_dnorm_norm_sum = F::new(0.0);

                for d in 0..head_dim {
                    let z1_d = z1_shared[seq_idx * head_dim + d];
                    let z1_norm_d = (z1_d - mean) * inv_std;
                    let dl_dnorm_d = grad_shared[seq_idx * head_dim + d];

                    dl_dnorm_sum += dl_dnorm_d;
                    dl_dnorm_norm_sum += dl_dnorm_d * z1_norm_d;
                }

                let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
                let z1_norm = (z1_val - mean) * inv_std;
                let dl_dnorm = grad_shared[seq_idx * head_dim + dim_idx];
                let n = F::cast_from(head_dim);

                let grad_val =
                    (dl_dnorm * n - dl_dnorm_sum - z1_norm * dl_dnorm_norm_sum) * inv_std / n;

                grad_shared[seq_idx * head_dim + dim_idx] = grad_val;
            }

            sync_cube();

            // === b1_bar and z1_bar ===
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let token_eta_i = token_eta[seq_idx];
                let mut eta_grad_sum = F::new(0.0);

                for j in 0..mini_batch_len {
                    if j <= seq_idx {
                        let eta_ij = token_eta_i * ttt_lr_eta[stage_seq_base + j];
                        let grad_jd = grad_shared[j * head_dim + dim_idx];
                        eta_grad_sum += eta_ij * grad_jd;
                    }
                }

                let b1_bar_val = bias_shared[dim_idx] - eta_grad_sum;
                z1_bar_shared[seq_idx * head_dim + dim_idx] = b1_bar_val;
            }

            sync_cube();

            // === z1_bar = xq @ W - (eta * attn1) @ grad + b1_bar ===
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let xq_base = stage_seq_dim_base + seq_idx * head_dim;
                let token_eta_i = token_eta[seq_idx];

                let mut xq_w = F::new(0.0);
                for k in 0..head_dim {
                    xq_w += xq[xq_base + k] * w_shared[k * head_dim + dim_idx];
                }

                let mut correction = F::new(0.0);
                for j in 0..mini_batch_len {
                    if j <= seq_idx {
                        let xk_base_j = stage_seq_dim_base + j * head_dim;
                        let mut attn_ij = F::new(0.0);
                        for k in 0..head_dim {
                            attn_ij += xq[xq_base + k] * xk[xk_base_j + k];
                        }

                        let eta_ij = token_eta_i * ttt_lr_eta[stage_seq_base + j];
                        let grad_jd = grad_shared[j * head_dim + dim_idx];
                        correction += eta_ij * attn_ij * grad_jd;
                    }
                }

                let b1_bar_val = z1_bar_shared[seq_idx * head_dim + dim_idx];
                let z1_bar_val = xq_w - correction + b1_bar_val;
                z1_bar_shared[seq_idx * head_dim + dim_idx] = z1_bar_val;
            }

            sync_cube();

            // === Weight and bias update: W -= last_eta * XK^T @ grad, b -= last_eta @ grad ===
            let last_seq = mini_batch_len - 1;
            let token_eta_last = token_eta[last_seq];

            if dim_idx < head_dim && seq_idx == 0 {
                for row in 0..head_dim {
                    let mut update = F::new(0.0);
                    for k in 0..mini_batch_len {
                        let eta_k = token_eta_last * ttt_lr_eta[stage_seq_base + k];
                        let xk_kr = xk[stage_seq_dim_base + k * head_dim + row];
                        let grad_kc = grad_shared[k * head_dim + dim_idx];
                        update += eta_k * xk_kr * grad_kc;
                    }
                    w_shared[row * head_dim + dim_idx] =
                        w_shared[row * head_dim + dim_idx] - update;
                }
            }

            if seq_idx == 0 && dim_idx < head_dim {
                let mut update = F::new(0.0);
                for k in 0..mini_batch_len {
                    let eta_k = token_eta_last * ttt_lr_eta[stage_seq_base + k];
                    let grad_kd = grad_shared[k * head_dim + dim_idx];
                    update += eta_k * grad_kd;
                }
                bias_shared[dim_idx] = bias_shared[dim_idx] - update;
            }

            sync_cube();

            // === Layer norm of z1_bar for output ===
            if seq_idx < mini_batch_len {
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

                if dim_idx == 0 {
                    ln_mean_shared[seq_idx] = mean;
                    ln_inv_std_shared[seq_idx] = inv_std;
                }
            }

            sync_cube();

            // === output = xq + LN(z1_bar) ===
            if seq_idx < mini_batch_len && dim_idx < head_dim {
                let idx = stage_seq_dim_base + seq_idx * head_dim + dim_idx;
                let mean = ln_mean_shared[seq_idx];
                let inv_std = ln_inv_std_shared[seq_idx];

                let z1_bar_val = z1_bar_shared[seq_idx * head_dim + dim_idx];
                let z1_bar_norm = (z1_bar_val - mean) * inv_std;

                let ln_w = ln_weight[h_dim_base + dim_idx];
                let ln_b = ln_bias[h_dim_base + dim_idx];
                let ln_out = ln_w * z1_bar_norm + ln_b;

                output[idx] = xq[idx] + ln_out;
            }

            sync_cube();
        }

        // Store final W and b
        if seq_idx == 0 && dim_idx < head_dim {
            for row in 0..head_dim {
                weight_out[bh_dim_dim_base + row * head_dim + dim_idx] =
                    w_shared[row * head_dim + dim_idx];
            }
        }

        if seq_idx == 0 && dim_idx < head_dim {
            bias_out[bh_dim_base + dim_idx] = bias_shared[dim_idx];
        }
    }
}
