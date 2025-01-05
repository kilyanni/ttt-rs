//! Layer normalization kernels and their backward passes.
//!
//! This module contains:
//! - `layer_norm_forward`: Basic layer norm forward pass
//! - `layer_norm_l2_grad`: Layer norm forward + L2 gradient backward (fused)
//! - `layer_norm_backward`: Standard layer norm backward pass
//! - `layer_norm_l2_grad_backward`: Second derivative through LN+L2 (backward-backward)

#![allow(non_camel_case_types, non_snake_case)]

use cubecl::prelude::*;
use thundercube::{
    cube::ReduceBuf,
    impl_reduction_ops,
    prelude::*,
    reduction_ops::{ReductionOp, SumOp},
};

// Sum of squares reduction operation for variance computation.
// Results in SumSqOp struct (macro adds Op suffix).
impl_reduction_ops! {
    SumSq<F> {
        identity => Line::<F>::empty(LINE_SIZE).fill(F::new(0.0));
        combine(a, b) => a + b * b;
        finalize(line) => line[0] + line[1] + line[2] + line[3];
        plane_reduce(val) => plane_sum(val);
        plane_combine(a, b) => a + b;  // Merge partials by addition, not squaring
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Normalize x in place to x_hat = (x - mean) / std.
/// Returns std (standard deviation per row).
///
/// After this function:
/// - `x` contains x_hat (normalized, zero mean, unit variance)
/// - returned `std` contains the standard deviation per row
#[cube]
pub fn normalize_to_x_hat<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    x: &mut St<FVal, R, C>,
    buf: &mut ReduceBuf<FAcc>,
    #[comptime] epsilon: f32,
) -> Rv<FVal, R> {
    let c_inv = FAcc::cast_from(1.0f32 / (C::VALUE as f32));

    // mean = sum_rows(x) / C
    let mut mean_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(x, &mut mean_acc, buf);
    mean_acc.mul_scalar(c_inv);
    let mean = mean_acc.cast::<FVal>();

    // x -= mean
    x.sub_col(&mean);
    sync_cube();

    // var = sum_rows(x^2) / C
    let mut std_acc = Rv::<FAcc, R>::new();
    cube::reduce_rows::<FVal, FAcc, R, C, SumSqOp>(x, &mut std_acc, buf);
    std_acc.mul_scalar(c_inv);

    // std = sqrt(var + epsilon)
    std_acc.add_scalar(FAcc::cast_from(epsilon));
    std_acc.sqrt();

    let std = std_acc.cast::<FVal>();

    // x /= std -> x_hat
    x.div_col(&std);
    sync_cube();

    std
}

/// Compute grad_x from grad_x_hat using the layer norm backward formula:
/// grad_x = (grad_x_hat * C - sum(grad_x_hat) - x_hat * sum(grad_x_hat * x_hat)) / (std * C)
///
/// Memory-optimized version that writes the result directly to x_hat, eliminating
/// the need for a separate output tile. Operations are reordered so x_hat is only
/// read before being overwritten.
///
/// # Arguments
/// * `grad_x_hat` - Gradient w.r.t. normalized input [R, C]
/// * `x_hat` - Normalized input [R, C], will be overwritten with grad_x result
/// * `std` - Standard deviation per row [R]
/// * `temp` - Scratch tile [R, C]
#[cube]
pub fn compute_grad_x_from_grad_x_hat<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    grad_x_hat: &St<FVal, R, C>,
    x_hat: &mut St<FVal, R, C>,
    std: &Rv<FVal, R>,
    temp: &mut St<FVal, R, C>,
    buf: &mut ReduceBuf<FAcc>,
) {
    let c_f = FVal::cast_from(C::VALUE as f32);
    let c_inv = FVal::cast_from(1.0f32 / (C::VALUE as f32));

    // Step 1: Compute sums (need both grad_x_hat and x_hat)
    // sum_grad_x_hat = sum(grad_x_hat) per row
    let mut sum_grad_x_hat_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(grad_x_hat, &mut sum_grad_x_hat_acc, buf);
    let sum_grad_x_hat = sum_grad_x_hat_acc.cast::<FVal>();

    // sum_grad_x_hat_x_hat = sum(grad_x_hat * x_hat) per row
    temp.copy_from(grad_x_hat);
    temp.mul(x_hat);
    sync_cube();

    let mut sum_grad_x_hat_x_hat_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(temp, &mut sum_grad_x_hat_x_hat_acc, buf);
    let sum_grad_x_hat_x_hat = sum_grad_x_hat_x_hat_acc.cast::<FVal>();

    // Step 2: Compute x_hat * sum_grad_x_hat_x_hat
    temp.copy_from(x_hat);
    temp.mul_col(&sum_grad_x_hat_x_hat);
    sync_cube();

    // x_hat = grad_x_hat * C
    x_hat.copy_from(grad_x_hat);
    x_hat.mul_scalar(c_f);

    // x_hat -= sum_grad_x_hat
    x_hat.sub_col(&sum_grad_x_hat);
    sync_cube();

    // x_hat -= temp
    x_hat.sub(temp);

    // x_hat /= (std * C)
    x_hat.div_col(std);
    x_hat.mul_scalar(c_inv);
}

// =============================================================================
// Public layer norm functions
// =============================================================================

/// Computes layer norm forward pass only.
///
/// out = weight * ((x - mean) / std) + bias
///
/// # Arguments
/// * `x` - Input tile [R, C], will be modified to contain the normalized output
/// * `ln_weight` - Layer norm weight vector [C]
/// * `ln_bias` - Layer norm bias vector [C]
/// * `epsilon` - Small constant for numerical stability
#[cube]
pub fn layer_norm_forward<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    x: &mut St<FVal, R, C>,
    ln_weight: &Rv<FVal, C>,
    ln_bias: &Rv<FVal, C>,
    buf: &mut ReduceBuf<FAcc>,
    #[comptime] epsilon: f32,
) {
    // Normalize: x -> x_hat = (x - mean) / std
    let _std = normalize_to_x_hat::<FVal, FAcc, R, C>(x, buf, epsilon);

    // Apply affine: x = weight * x_hat + bias
    x.mul_row(ln_weight);
    x.add_row(ln_bias);
}

/// Computes layer norm forward pass and returns intermediate values needed for backward.
///
/// # Arguments
/// * `x` - Input tile [R, C], will be modified to contain x_hat (normalized, pre-affine)
/// * `ln_weight` - Layer norm weight vector [C]
/// * `ln_bias` - Layer norm bias vector [C]
/// * `output` - Output tile [R, C], will contain the final layer norm output
/// * `std_out` - Output vector [R], will contain std per row
/// * `epsilon` - Small constant for numerical stability
#[cube]
pub fn layer_norm_forward_with_intermediates<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    x: &mut St<FVal, R, C>,
    ln_weight: &Rv<FVal, C>,
    ln_bias: &Rv<FVal, C>,
    output: &mut St<FVal, R, C>,
    std_out: &mut Rv<FVal, R>,
    buf: &mut ReduceBuf<FAcc>,
    #[comptime] epsilon: f32,
) {
    // Normalize: x -> x_hat, save std
    let std = normalize_to_x_hat::<FVal, FAcc, R, C>(x, buf, epsilon);
    std_out.set(&std);

    // output = weight * x_hat + bias
    output.copy_from(x);
    output.mul_row(ln_weight);
    output.add_row(ln_bias);
}

/// Computes layer norm forward and L2 loss gradient backpropagated through layer norm.
///
/// Forward: y = weight * ((x - mean) / std) + bias
/// L2 gradient: dl_dx = backward through y w.r.t. L2 loss (y - target)^2
///
/// # Arguments
/// * `x` - Input tile [R, C], will be modified to contain dl_dx (the gradient)
/// * `target` - Target tile [R, C] for L2 loss
/// * `ln_weight` - Layer norm weight vector [C]
/// * `ln_bias` - Layer norm bias vector [C]
/// * `temp` - Scratch tile [R, C] for intermediate computations
/// * `epsilon` - Small constant for numerical stability
#[cube]
pub fn layer_norm_l2_grad<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    x: &mut St<FVal, R, C>,
    target: &mut St<FVal, R, C>,
    ln_weight: &Rv<FVal, C>,
    ln_bias: &Rv<FVal, C>,
    temp: &mut St<FVal, R, C>,
    buf: &mut ReduceBuf<FAcc>,
    #[comptime] epsilon: f32,
) {
    // Normalize: x -> x_hat
    let std = normalize_to_x_hat::<FVal, FAcc, R, C>(x, buf, epsilon);

    // Compute y = weight * x_hat + bias, then dl_dout = y - target
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);
    temp.sub(target);

    // dl_dnorm = dl_dout * weight (grad_x_hat)
    temp.mul_row(ln_weight);
    sync_cube();

    // Compute grad_x from grad_x_hat (writes result directly to x)
    compute_grad_x_from_grad_x_hat::<FVal, FAcc, R, C>(temp, x, &std, target, buf);

    sync_cube();
}

/// Fused layer norm + L2 gradient computation that streams intermediates directly to global memory.
///
/// Same as `layer_norm_l2_grad` but saves the intermediate values needed for
/// the backward pass by storing directly to global tensors (no intermediate smem tiles).
///
/// Saves (directly to global):
/// - x_hat: normalized input (x - mean) / std [R, C]
/// - grad_output: y - target [R, C]
/// - grad_x_hat: grad_output * ln_weight [R, C]
///
/// Returns std (caller should store to global separately).
/// The grad_l result is written to `x` as usual.
///
/// `scratch` is an external scratch tile (caller can pass a dead tile).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_l2_grad_stream_intermediates<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    x: &mut St<FVal, R, C>,
    target: &St<FVal, R, C>,
    ln_weight: &Rv<FVal, C>,
    ln_bias: &Rv<FVal, C>,
    temp: &mut St<FVal, R, C>,
    scratch: &mut St<FVal, R, C>,
    buf: &mut ReduceBuf<FAcc>,
    x_hat_global: &mut Tensor<Line<FVal>>,
    grad_output_global: &mut Tensor<Line<FVal>>,
    grad_x_hat_global: &mut Tensor<Line<FVal>>,
    store_offset: usize,
    #[comptime] epsilon: f32,
) -> Rv<FVal, R> {
    // Normalize: x -> x_hat
    let std = normalize_to_x_hat::<FVal, FAcc, R, C>(x, buf, epsilon);

    // Store x_hat directly to global (x still contains x_hat for subsequent use)
    cube::store_st_direct(x, x_hat_global, store_offset, 0, 0);

    // Compute y = weight * x_hat + bias
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);

    // grad_output = y - target
    temp.sub(target);
    sync_cube();

    // Store grad_output directly to global
    cube::store_st_direct(temp, grad_output_global, store_offset, 0, 0);

    // grad_x_hat = grad_output * ln_weight
    temp.mul_row(ln_weight);
    sync_cube();

    // Store grad_x_hat directly to global
    cube::store_st_direct(temp, grad_x_hat_global, store_offset, 0, 0);

    // Compute grad_x from grad_x_hat (writes result directly to x)
    compute_grad_x_from_grad_x_hat::<FVal, FAcc, R, C>(temp, x, &std, scratch, buf);

    sync_cube();

    std
}

/// Computes layer norm forward and saves x_hat and std for backward.
///
/// Same as `layer_norm_forward` but saves:
/// - x_hat: normalized input before weight/bias [R, C]
/// - std: standard deviation per row [R]
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_forward_save_intermediates<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    x: &mut St<FVal, R, C>,
    ln_weight: &Rv<FVal, C>,
    ln_bias: &Rv<FVal, C>,
    buf: &mut ReduceBuf<FAcc>,
    x_hat_out: &mut St<FVal, R, C>,
    std_out: &mut Rv<FVal, R>,
    #[comptime] epsilon: f32,
) {
    // Normalize: x -> x_hat, save std
    let std = normalize_to_x_hat::<FVal, FAcc, R, C>(x, buf, epsilon);
    std_out.set(&std);

    // Save x_hat for backward
    x_hat_out.copy_from(x);

    // Apply affine: x = weight * x_hat + bias
    x.mul_row(ln_weight);
    x.add_row(ln_bias);
}

/// Computes layer norm forward and streams x_hat directly to global memory.
///
/// Memory-optimized version that stores x_hat directly to global instead of
/// buffering in shared memory. Returns std for the caller to store.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_forward_stream_intermediates<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    x: &mut St<FVal, R, C>,
    ln_weight: &Rv<FVal, C>,
    ln_bias: &Rv<FVal, C>,
    buf: &mut ReduceBuf<FAcc>,
    x_hat_global: &mut Tensor<Line<FVal>>,
    store_offset: usize,
    #[comptime] epsilon: f32,
) -> Rv<FVal, R> {
    // Normalize: x -> x_hat
    let std = normalize_to_x_hat::<FVal, FAcc, R, C>(x, buf, epsilon);

    // Store x_hat directly to global (x still contains x_hat for affine transform)
    cube::store_st_direct(x, x_hat_global, store_offset, 0, 0);

    // Apply affine: x = weight * x_hat + bias
    x.mul_row(ln_weight);
    x.add_row(ln_bias);

    std
}

/// Standard layer norm backward pass with temp storage.
/// Computes gradients w.r.t. input, weight, and bias.
///
/// Given:
/// - grad_output: upstream gradient [R, C]
/// - x_hat: normalized input (x - mean) / std [R, C]
/// - std: standard deviation per row [R]
/// - ln_weight: layer norm weight [C]
///
/// Computes:
/// - grad_x: gradient w.r.t. input [R, C]
/// - grad_ln_weight: gradient w.r.t. weight [C]
/// - grad_ln_bias: gradient w.r.t. bias [C]
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    grad_output: &St<FVal, R, C>,
    x_hat: &St<FVal, R, C>,
    std: &Rv<FVal, R>,
    ln_weight: &Rv<FVal, C>,
    temp: &mut St<FVal, R, C>,
    grad_x: &mut St<FVal, R, C>,
    grad_ln_weight: &mut Rv<FAcc, C>,
    grad_ln_bias: &mut Rv<FAcc, C>,
    buf: &mut ReduceBuf<FAcc>,
) {
    // grad_ln_bias = sum(grad_output)
    cube::reduce_cols::<FVal, FAcc, R, C, SumOp>(grad_output, grad_ln_bias, buf);

    // grad_ln_weight = sum(grad_output * x_hat)
    temp.copy_from(grad_output);
    temp.mul(x_hat);
    sync_cube();
    cube::reduce_cols::<FVal, FAcc, R, C, SumOp>(temp, grad_ln_weight, buf);

    // grad_x_hat = grad_output * weight
    temp.copy_from(grad_output);
    temp.mul_row(ln_weight);
    sync_cube();

    // grad_x from grad_x_hat using the backward formula
    // Copy x_hat to grad_x first (since compute_grad_x_from_grad_x_hat overwrites x_hat in place)
    grad_x.copy_from(x_hat);
    sync_cube();

    let mut scratch = St::<FVal, R, C>::new();
    compute_grad_x_from_grad_x_hat::<FVal, FAcc, R, C>(temp, grad_x, std, &mut scratch, buf);
}

/// Backward through the fused layer norm + L2 gradient computation.
/// This is the second derivative (backward-backward) through the LN+L2 forward.
///
/// The forward computed:
///   x_hat = (Z1 - mean) / std
///   y = ln_weight * x_hat + ln_bias
///   grad_output = y - target
///   grad_x_hat = grad_output * ln_weight
///   grad_l = (grad_x_hat * F - sum(grad_x_hat) - x_hat * sum(grad_x_hat * x_hat)) / (std * F)
///
/// Given upstream gradient `grad_L_grad_l` (gradient w.r.t. grad_l),
/// this computes gradients w.r.t. Z1, target, ln_weight, ln_bias.
///
/// # Arguments
/// * `grad_L_grad_l` - Upstream gradient w.r.t. grad_l [R, C]
/// * `x_hat` - Saved x_hat from forward [R, C]
/// * `std` - Saved std from forward [R]
/// * `grad_output` - Saved (y - target) from forward [R, C]
/// * `grad_x_hat` - Saved grad_output * ln_weight from forward [R, C]
/// * `ln_weight` - Layer norm weight [C]
/// * `temp1`, `temp2` - Scratch tiles [R, C]
/// * `grad_L_Z1` - Output: gradient w.r.t. Z1 [R, C]
/// * `grad_L_target` - Output: gradient w.r.t. target [R, C]
/// * `grad_L_ln_weight` - Output: gradient w.r.t. ln_weight [C]
/// * `grad_L_ln_bias` - Output: gradient w.r.t. ln_bias [C]
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_l2_grad_backward<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    // Upstream gradient
    grad_L_grad_l: &St<FVal, R, C>,
    // Saved from forward
    x_hat: &St<FVal, R, C>,
    std: &Rv<FVal, R>,
    grad_output: &St<FVal, R, C>, // y - target from forward
    grad_x_hat: &St<FVal, R, C>,  // grad_output * ln_weight from forward
    ln_weight: &Rv<FVal, C>,
    // Temp storage
    temp1: &mut St<FVal, R, C>,
    temp2: &mut St<FVal, R, C>,
    // Outputs (shared memory)
    grad_L_Z1: &mut St<FVal, R, C>,
    grad_L_target: &mut St<FVal, R, C>,
    // Outputs
    grad_L_ln_weight: &mut Rv<FAcc, C>,
    grad_L_ln_bias: &mut Rv<FAcc, C>,
    buf: &mut ReduceBuf<FAcc>,
) {
    let f_f = FVal::cast_from(C::VALUE as f32);
    let f_inv = FVal::cast_from(1.0f32 / (C::VALUE as f32));

    // From Triton reference:
    // grad_L_grad_x_hat = (1/std) * grad_L_grad_l
    //                   + (1/F) * sum(-grad_L_grad_l / std, axis=1)
    //                   + (1/F) * x_hat * sum(-grad_L_grad_l / std * x_hat, axis=1)

    // First compute -grad_L_grad_l / std
    temp1.copy_from(grad_L_grad_l);
    temp1.neg();
    temp1.div_col(std);

    sync_cube();

    // sum1 = sum(-grad_L_grad_l / std) per row
    let mut sum1_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(temp1, &mut sum1_acc, buf);
    let sum1 = sum1_acc.cast::<FVal>();

    // sum2 = sum(-grad_L_grad_l / std * x_hat) per row
    temp1.mul(x_hat);

    sync_cube();

    let mut sum2_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(temp1, &mut sum2_acc, buf);
    let sum2 = sum2_acc.cast::<FVal>();

    // grad_L_grad_x_hat = (1/std) * grad_L_grad_l + (1/F) * sum1 + (1/F) * x_hat * sum2
    // temp1 = (1/std) * grad_L_grad_l
    temp1.copy_from(grad_L_grad_l);
    temp1.div_col(std);

    // Add (1/F) * sum1 broadcast
    let mut scaled_sum1 = sum1;
    scaled_sum1.mul_scalar(f_inv);
    temp1.add_col(&scaled_sum1);

    sync_cube();

    // Add (1/F) * x_hat * sum2
    // temp2 = x_hat * sum2
    temp2.copy_from(x_hat);
    temp2.mul_col(&sum2);
    temp2.mul_scalar(f_inv);

    sync_cube();

    temp1.add(temp2);

    sync_cube();

    // temp1 = grad_L_grad_x_hat

    // grad_L_y = ln_weight * grad_L_grad_x_hat
    temp2.copy_from(temp1);
    temp2.mul_row(ln_weight);

    sync_cube();

    // grad_L_ln_weight = sum(grad_output * grad_L_grad_x_hat + grad_L_y * x_hat)
    // First term: grad_output * grad_L_grad_x_hat
    grad_L_Z1.copy_from(grad_output);
    grad_L_Z1.mul(temp1);

    sync_cube();

    // Second term: grad_L_y * x_hat = temp2 * x_hat
    grad_L_target.copy_from(temp2);
    grad_L_target.mul(x_hat);

    sync_cube();

    grad_L_Z1.add(grad_L_target);

    sync_cube();

    cube::reduce_cols::<FVal, FAcc, R, C, SumOp>(grad_L_Z1, grad_L_ln_weight, buf);

    // grad_L_ln_bias = sum(grad_L_y) = sum(temp2)
    cube::reduce_cols::<FVal, FAcc, R, C, SumOp>(temp2, grad_L_ln_bias, buf);

    // grad_L_x_hat = grad_L_y * ln_weight
    //              + (1/F) * grad_x_hat * sum(-grad_L_grad_l / std * x_hat)
    //              + (1/F) * sum(grad_x_hat * x_hat) * (-grad_L_grad_l / std)

    // Start fresh: temp1 = grad_L_y * ln_weight
    temp1.copy_from(temp2);
    temp1.mul_row(ln_weight);

    sync_cube();

    // Compute sum(grad_x_hat * x_hat) per row
    temp2.copy_from(grad_x_hat);
    temp2.mul(x_hat);

    sync_cube();

    let mut sum_gxh_xh_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(temp2, &mut sum_gxh_xh_acc, buf);
    let sum_gxh_xh = sum_gxh_xh_acc.cast::<FVal>();

    // Term 2: (1/F) * grad_x_hat * sum2 (sum2 = sum(-grad_L_grad_l / std * x_hat))
    temp2.copy_from(grad_x_hat);
    temp2.mul_col(&sum2);
    temp2.mul_scalar(f_inv);

    sync_cube();

    temp1.add(temp2);

    sync_cube();

    // Term 3: (1/F) * sum(grad_x_hat * x_hat) * (-grad_L_grad_l / std)
    temp2.copy_from(grad_L_grad_l);
    temp2.neg();
    temp2.div_col(std);
    temp2.mul_col(&sum_gxh_xh);
    temp2.mul_scalar(f_inv);

    sync_cube();

    temp1.add(temp2);

    sync_cube();

    // temp1 = grad_L_x_hat

    // We need to compute grad_l for the std gradient
    // grad_l = (grad_x_hat * F - sum(grad_x_hat) - x_hat * sum(grad_x_hat * x_hat)) / (std * F)

    // Compute sum(grad_x_hat) per row
    let mut sum_gxh_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(grad_x_hat, &mut sum_gxh_acc, buf);
    let sum_gxh = sum_gxh_acc.cast::<FVal>();

    // grad_l = (grad_x_hat * F - sum_gxh - x_hat * sum_gxh_xh) / (std * F)
    temp2.copy_from(grad_x_hat);
    temp2.mul_scalar(f_f);
    temp2.sub_col(&sum_gxh);

    sync_cube();

    // Subtract x_hat * sum_gxh_xh
    grad_L_Z1.copy_from(x_hat);
    grad_L_Z1.mul_col(&sum_gxh_xh);

    sync_cube();

    temp2.sub(grad_L_Z1);

    // Divide by std * F = divide by std, then divide by F
    temp2.div_col(std);
    temp2.mul_scalar(f_inv);

    sync_cube();

    // temp2 = grad_l

    // grad_L_std = -grad_L_x_hat * (x_hat / std) - grad_L_grad_l * (grad_l / std)
    //            = -(temp1 * x_hat + grad_L_grad_l * temp2) / std
    grad_L_Z1.copy_from(temp1);
    grad_L_Z1.mul(x_hat);

    sync_cube();

    grad_L_target.copy_from(grad_L_grad_l);
    grad_L_target.mul(temp2);

    sync_cube();

    grad_L_Z1.add(grad_L_target);
    grad_L_Z1.neg();
    grad_L_Z1.div_col(std);

    sync_cube();

    // grad_L_Z1 = grad_L_std

    // Compute sum(grad_L_std) per row
    let mut sum_grad_L_std_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(grad_L_Z1, &mut sum_grad_L_std_acc, buf);
    let sum_grad_L_std = sum_grad_L_std_acc.cast::<FVal>();

    // Final: grad_L_Z1 = grad_L_x_hat / std - (1/F) * sum(grad_L_x_hat) / std + (1/F) * sum(grad_L_std) * x_hat

    let mut sum_grad_L_x_hat_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(temp1, &mut sum_grad_L_x_hat_acc, buf);
    let sum_grad_L_x_hat = sum_grad_L_x_hat_acc.cast::<FVal>();

    // grad_L_Z1 = temp1 / std
    grad_L_Z1.copy_from(temp1);
    grad_L_Z1.div_col(std);

    // Subtract (1/F) * sum(grad_L_x_hat) / std
    let mut term2 = sum_grad_L_x_hat;
    term2.div(std);
    term2.mul_scalar(f_inv);
    grad_L_Z1.sub_col(&term2);

    sync_cube();

    // Add (1/F) * sum(grad_L_std) * x_hat
    temp2.copy_from(x_hat);
    let mut scaled_sum_std = sum_grad_L_std;
    scaled_sum_std.mul_scalar(f_inv);
    temp2.mul_col(&scaled_sum_std);

    sync_cube();

    grad_L_Z1.add(temp2);

    sync_cube();

    // grad_L_target = -ln_weight * grad_L_grad_x_hat
    // Recompute -grad_L_grad_l / std
    temp1.copy_from(grad_L_grad_l);
    temp1.neg();
    temp1.div_col(std);

    sync_cube();

    let mut sum1_recomputed_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(temp1, &mut sum1_recomputed_acc, buf);
    let sum1_recomputed = sum1_recomputed_acc.cast::<FVal>();

    temp1.mul(x_hat);

    sync_cube();

    let mut sum2_recomputed_acc = Rv::<FAcc, R>::new();
    cube::sum_rows::<FVal, FAcc, R, C>(temp1, &mut sum2_recomputed_acc, buf);
    let sum2_recomputed = sum2_recomputed_acc.cast::<FVal>();

    // grad_L_grad_x_hat = (1/std) * grad_L_grad_l + (1/F) * sum1 + (1/F) * x_hat * sum2
    temp1.copy_from(grad_L_grad_l);
    temp1.div_col(std);

    let mut s1 = sum1_recomputed;
    s1.mul_scalar(f_inv);
    temp1.add_col(&s1);

    sync_cube();

    temp2.copy_from(x_hat);
    temp2.mul_col(&sum2_recomputed);
    temp2.mul_scalar(f_inv);

    sync_cube();

    temp1.add(temp2);

    sync_cube();

    // grad_L_target = -ln_weight * grad_L_grad_x_hat
    grad_L_target.copy_from(temp1);
    grad_L_target.mul_row(ln_weight);
    grad_L_target.neg();
}

#[cfg(test)]
mod tests {
    use test_case::test_matrix;
    use thundercube::{test_kernel, test_utils::TestFloat};

    use super::*;

    const ROWS: usize = 8;
    const COLS: usize = 32;
    const EPSILON: f32 = 1e-5;

    // =========================================================================
    // Test kernels that wrap the layer norm functions
    // =========================================================================

    /// Test kernel for layer_norm_forward
    #[cube(launch)]
    fn test_ln_forward_kernel<F: Float>(
        input: &Tensor<Line<F>>,
        ln_weight: &Tensor<Line<F>>,
        ln_bias: &Tensor<Line<F>>,
        output: &mut Tensor<Line<F>>,
    ) {
        let mut x = St::<F, D8, D32>::new();
        let mut weight = Rv::<F, D32>::new();
        let mut bias = Rv::<F, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        // Load using library functions
        cube::load_st_direct(input, &mut x, 0, 0, 0);
        cube::broadcast::load_rv_direct(ln_weight, &mut weight, 0);
        cube::broadcast::load_rv_direct(ln_bias, &mut bias, 0);

        sync_cube();

        // Run layer norm forward
        layer_norm_forward::<F, F, D8, D32>(&mut x, &weight, &bias, &mut buf, EPSILON);

        sync_cube();

        // Store using library function
        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Test kernel for layer_norm_l2_grad
    #[cube(launch)]
    fn test_ln_l2_grad_kernel<F: Float>(
        input: &Tensor<Line<F>>,
        target: &Tensor<Line<F>>,
        ln_weight: &Tensor<Line<F>>,
        ln_bias: &Tensor<Line<F>>,
        output: &mut Tensor<Line<F>>,
    ) {
        let mut x = St::<F, D8, D32>::new();
        let mut tgt = St::<F, D8, D32>::new();
        let mut temp = St::<F, D8, D32>::new();
        let mut weight = Rv::<F, D32>::new();
        let mut bias = Rv::<F, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        // Load using library functions
        cube::load_st_direct(input, &mut x, 0, 0, 0);
        cube::load_st_direct(target, &mut tgt, 0, 0, 0);
        cube::broadcast::load_rv_direct(ln_weight, &mut weight, 0);
        cube::broadcast::load_rv_direct(ln_bias, &mut bias, 0);

        sync_cube();

        // Run layer norm + L2 grad
        layer_norm_l2_grad::<F, F, D8, D32>(
            &mut x, &mut tgt, &weight, &bias, &mut temp, &mut buf, EPSILON,
        );

        sync_cube();

        // Store using library function
        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Identity test kernel - just load and store to verify load/store works
    #[cube(launch)]
    fn test_identity_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();
        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Test kernel for just computing row means (diagnostic)
    #[cube(launch)]
    fn test_row_mean_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();

        // Compute row means: sum across columns, divide by C
        let mut mean = Rv::<F, D8>::new();
        cube::sum_rows::<F, F, D8, D32>(&x, &mut mean, &mut buf);
        mean.mul_scalar(F::cast_from(1.0f32 / 32.0f32));

        // Output is just the row means (broadcast to full row width)
        // Write mean to first row of output
        if UNIT_POS == 0 {
            for i in 0..D8::LINES {
                let line_idx = i;
                output[line_idx] = mean.data[i];
            }
        }
    }

    /// Test kernel for center (subtract mean) - steps 1-4 of layer norm
    #[cube(launch)]
    fn test_center_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();

        // Step 1: mean = sum_rows(x) / C
        let mut mean = Rv::<F, D8>::new();
        cube::sum_rows::<F, F, D8, D32>(&x, &mut mean, &mut buf);
        mean.mul_scalar(F::cast_from(1.0f32 / 32.0f32));

        // Step 2: x -= mean
        x.sub_col(&mean);

        sync_cube();

        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Test kernel for sum of squares reduction (diagnostic)
    #[cube(launch)]
    fn test_sum_sq_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();

        // Compute sum of squares per row
        let mut sum_sq = Rv::<F, D8>::new();
        cube::reduce_rows::<F, F, D8, D32, SumSqOp>(&x, &mut sum_sq, &mut buf);

        // Output the sum of squares
        if UNIT_POS == 0 {
            for i in 0..D8::LINES {
                output[i] = sum_sq.data[i];
            }
        }
    }

    /// Test kernel for normalize (center + divide by std)
    #[cube(launch)]
    fn test_normalize_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();

        // Step 1: mean = sum_rows(x) / C
        let mut mean = Rv::<F, D8>::new();
        cube::sum_rows::<F, F, D8, D32>(&x, &mut mean, &mut buf);
        mean.mul_scalar(F::cast_from(1.0f32 / 32.0f32));

        // Step 2: x -= mean
        x.sub_col(&mean);
        sync_cube();

        // Step 3: var = sum_rows(x^2) / C
        let mut std = Rv::<F, D8>::new();
        cube::reduce_rows::<F, F, D8, D32, SumSqOp>(&x, &mut std, &mut buf);
        std.mul_scalar(F::cast_from(1.0f32 / 32.0f32));

        // Step 4: std = sqrt(var + epsilon)
        std.add_scalar(F::cast_from(EPSILON));
        std.sqrt();

        // Step 5: x /= std
        x.div_col(&std);
        sync_cube();

        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Diagnostic kernel to check thread indices
    #[cube(launch)]
    fn test_thread_indices<F: Float>(output: &mut Tensor<Line<F>>) {
        let tid = UNIT_POS;
        let cube_dim = CUBE_DIM;
        let plane_dim = PLANE_DIM;

        // Each thread writes its index to its position
        // output[0] = tid for thread 0, output[1] = tid for thread 1, etc.
        if (tid as usize) < 64 {
            let val = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(tid));
            output[tid as usize] = val;
        }

        // Thread 0 also writes cube_dim and plane_dim to slots 64 and 65
        if tid == 0 {
            output[64] = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(cube_dim));
            output[65] = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(plane_dim));
        }
    }

    // =========================================================================
    // Tests using test_kernel! macro
    // =========================================================================

    test_kernel! {
        #[test_matrix([64])]
        fn test_thread_indices_check(threads: usize) for F in [f32] {
            let output: Tensor = [66 * LINE_SIZE] as Range;

            assert_eq!(
                test_thread_indices(output()) for (1, 1, 1) @ (threads),
                {
                    // Each thread should write its tid
                    for i in 0..64 {
                        for j in 0..LINE_SIZE {
                            output[i * LINE_SIZE + j] = F::from_f64(i as f64);
                        }
                    }
                    // Thread 0 writes cube_dim (64) and plane_dim (32 on AMD)
                    for j in 0..LINE_SIZE {
                        output[64 * LINE_SIZE + j] = F::from_f64(64.0);
                        output[65 * LINE_SIZE + j] = F::from_f64(32.0);
                    }
                }
            );
        }

        #[test_matrix([64])]
        fn test_identity(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_identity_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Output should equal input
                    for i in 0..(ROWS * COLS) {
                        output[i] = input[i];
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_row_mean(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS] as Range;

            assert_eq!(
                test_row_mean_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Compute row means
                    for r in 0..ROWS {
                        let mut sum = 0.0;
                        for c in 0..COLS {
                            sum += input[r * COLS + c].into_f64();
                        }
                        output[r] = F::from_f64(sum / COLS as f64);
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_center(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_center_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Compute mean and subtract
                    for r in 0..ROWS {
                        let mut mean = 0.0;
                        for c in 0..COLS {
                            mean += input[r * COLS + c].into_f64();
                        }
                        mean /= COLS as f64;

                        for c in 0..COLS {
                            output[r * COLS + c] = F::from_f64(input[r * COLS + c].into_f64() - mean);
                        }
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_sum_sq(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS] as Range;

            assert_eq!(
                test_sum_sq_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Compute sum of squares per row
                    for r in 0..ROWS {
                        let mut sum_sq = 0.0;
                        for c in 0..COLS {
                            let val = input[r * COLS + c].into_f64();
                            sum_sq += val * val;
                        }
                        output[r] = F::from_f64(sum_sq);
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_normalize(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_normalize_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Compute x_hat = (x - mean) / std
                    for r in 0..ROWS {
                        let mut mean = 0.0;
                        for c in 0..COLS {
                            mean += input[r * COLS + c].into_f64();
                        }
                        mean /= COLS as f64;

                        let mut var = 0.0;
                        for c in 0..COLS {
                            let diff = input[r * COLS + c].into_f64() - mean;
                            var += diff * diff;
                        }
                        var /= COLS as f64;

                        let std = (var + EPSILON as f64).sqrt();

                        for c in 0..COLS {
                            let x_hat = (input[r * COLS + c].into_f64() - mean) / std;
                            output[r * COLS + c] = F::from_f64(x_hat);
                        }
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_layer_norm_forward(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let ln_weight: Tensor = [COLS];
            let ln_bias: Tensor = [COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_ln_forward_kernel(input(), ln_weight(), ln_bias(), output()) for (1, 1, 1) @ (threads),
                {
                    // Reference implementation
                    for r in 0..ROWS {
                        // Compute mean
                        let mut mean = 0.0;
                        for c in 0..COLS {
                            mean += input[r * COLS + c].into_f64();
                        }
                        mean /= COLS as f64;

                        // Compute variance
                        let mut var = 0.0;
                        for c in 0..COLS {
                            let diff = input[r * COLS + c].into_f64() - mean;
                            var += diff * diff;
                        }
                        var /= COLS as f64;

                        // Compute std
                        let std = (var + EPSILON as f64).sqrt();

                        // Normalize and apply affine
                        for c in 0..COLS {
                            let x_hat = (input[r * COLS + c].into_f64() - mean) / std;
                            let out = ln_weight[c].into_f64() * x_hat + ln_bias[c].into_f64();
                            output[r * COLS + c] = F::from_f64(out);
                        }
                    }
                }
            );
        }

        #[test_matrix([64])]
        fn test_layer_norm_l2_grad(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let target: Tensor = [ROWS, COLS];
            let ln_weight: Tensor = [COLS];
            let ln_bias: Tensor = [COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_ln_l2_grad_kernel(input(), target(), ln_weight(), ln_bias(), output()) for (1, 1, 1) @ (threads),
                {
                    // Reference implementation
                    for r in 0..ROWS {
                        // Forward pass: compute mean, var, std, x_hat, y
                        let mut mean = 0.0;
                        for c in 0..COLS {
                            mean += input[r * COLS + c].into_f64();
                        }
                        mean /= COLS as f64;

                        let mut var = 0.0;
                        for c in 0..COLS {
                            let diff = input[r * COLS + c].into_f64() - mean;
                            var += diff * diff;
                        }
                        var /= COLS as f64;

                        let std = (var + EPSILON as f64).sqrt();

                        let mut x_hat = vec![0.0f64; COLS];
                        let mut y = vec![0.0f64; COLS];
                        for c in 0..COLS {
                            x_hat[c] = (input[r * COLS + c].into_f64() - mean) / std;
                            y[c] = ln_weight[c].into_f64() * x_hat[c] + ln_bias[c].into_f64();
                        }

                        // L2 grad: dl_dout = y - target
                        let mut dl_dout = vec![0.0f64; COLS];
                        for c in 0..COLS {
                            dl_dout[c] = y[c] - target[r * COLS + c].into_f64();
                        }

                        // dl_dnorm = dl_dout * weight
                        let mut dl_dnorm = vec![0.0f64; COLS];
                        for c in 0..COLS {
                            dl_dnorm[c] = dl_dout[c] * ln_weight[c].into_f64();
                        }

                        // sum(dl_dnorm), sum(dl_dnorm * x_hat)
                        let mut sum_dl_dnorm = 0.0;
                        let mut sum_dl_dnorm_xhat = 0.0;
                        for c in 0..COLS {
                            sum_dl_dnorm += dl_dnorm[c];
                            sum_dl_dnorm_xhat += dl_dnorm[c] * x_hat[c];
                        }

                        // dl_dx = (dl_dnorm * C - sum_dl_dnorm - x_hat * sum_dl_dnorm_xhat) / (std * C)
                        let c_f = COLS as f64;
                        for c in 0..COLS {
                            let grad = (dl_dnorm[c] * c_f - sum_dl_dnorm - x_hat[c] * sum_dl_dnorm_xhat) / (std * c_f);
                            output[r * COLS + c] = F::from_f64(grad);
                        }
                    }
                }
            );
        }
    }
}
