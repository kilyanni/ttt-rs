//! Shared types and FusedKernel implementations for the tiled TTT-Linear kernel.

use std::fmt::Debug;

use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};
use ttt_kernels::{kernel::FusedKernel, tensor_bundle};

use super::{
    backward::{TttSavedTensors, backward, backward_multi},
    forward::{forward, forward_multi},
};
use crate::{
    FusedTttConfig,
    ttt::{TttInputs, TttOutputs},
};

tensor_bundle! {
    /// Saved state for backward pass (single-stage).
    /// Contains all inputs needed for activation recomputation.
    /// No forward intermediates are saved - they are recomputed during backward.
    pub struct TttSavedState {
        // Saved inputs for recomputation
        xq, xk, xv,           // All three projections (xv for target = xv - xk)
        weight_init, bias,    // Initial weight and bias for recomputation
        token_eta, ttt_lr_eta,
        ln_weight, ln_bias    // Layer norm params for recomputation
    }
}

tensor_bundle! {
    /// Saved state for backward pass (multi-stage).
    /// Includes per-stage weight/bias checkpoints from the forward pass
    /// to avoid O(N^2) forward re-simulation during backward.
    pub struct TttSavedStateMulti {
        xq, xk, xv,
        weight_init, bias,
        token_eta, ttt_lr_eta,
        ln_weight, ln_bias,
        weight_checkpoints, bias_checkpoints
    }
}

// TODO: 64 subtiles (full cubes?)

/// Marker type for the tiled TTT kernel (single mini-batch).
#[derive(Debug, Clone, Copy)]
pub struct TttTileKernel;

/// Marker type for the multi-stage tiled TTT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttTileMultiKernel;

impl FusedKernel for TttTileKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type SavedState<T: Debug + Clone + Send> = TttSavedState<T>;
    type Config = FusedTttConfig;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> (TttOutputs<CubeTensor<R>>, TttSavedState<CubeTensor<R>>) {
        // Clone inputs needed for saved state before moving into forward
        let saved = TttSavedState {
            xq: inputs.xq.clone(),
            xk: inputs.xk.clone(),
            xv: inputs.xv.clone(),
            weight_init: inputs.weight.clone(),
            bias: inputs.bias.clone(),
            token_eta: inputs.token_eta.clone(),
            ttt_lr_eta: inputs.ttt_lr_eta.clone(),
            ln_weight: inputs.ln_weight.clone(),
            ln_bias: inputs.ln_bias.clone(),
        };

        let outputs = forward::<R, F>(inputs, config);

        (outputs, saved)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: TttSavedState<CubeTensor<R>>,
        grad_outputs: TttOutputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> TttInputs<CubeTensor<R>> {
        let epsilon = config.epsilon();
        let threads = config.threads;

        let saved_tensors = TttSavedTensors {
            xq: saved.xq,
            xk: saved.xk,
            xv: saved.xv,
            weight_init: saved.weight_init,
            bias: saved.bias,
            token_eta: saved.token_eta,
            ttt_lr_eta: saved.ttt_lr_eta,
            ln_weight: saved.ln_weight,
            ln_bias: saved.ln_bias,
        };

        let grad_inputs = backward::<R, F>(saved_tensors, grad_outputs.output, epsilon, threads);

        TttInputs {
            xq: grad_inputs.grad_xq,
            xk: grad_inputs.grad_xk,
            xv: grad_inputs.grad_xv,
            weight: grad_inputs.grad_weight,
            bias: grad_inputs.grad_bias,
            token_eta: grad_inputs.grad_token_eta,
            ttt_lr_eta: grad_inputs.grad_ttt_lr_eta,
            ln_weight: grad_inputs.grad_ln_weight,
            ln_bias: grad_inputs.grad_ln_bias,
        }
    }
}

impl FusedKernel for TttTileMultiKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type SavedState<T: Debug + Clone + Send> = TttSavedStateMulti<T>;
    type Config = FusedTttConfig;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> (TttOutputs<CubeTensor<R>>, TttSavedStateMulti<CubeTensor<R>>) {
        let mini_batch_len = config.mini_batch_len;
        assert!(
            mini_batch_len > 0,
            "mini_batch_len must be set for TttTileMultiKernel"
        );

        // Clone inputs needed for saved state before moving into forward
        let xq = inputs.xq.clone();
        let xk = inputs.xk.clone();
        let xv = inputs.xv.clone();
        let weight_init = inputs.weight.clone();
        let bias = inputs.bias.clone();
        let token_eta = inputs.token_eta.clone();
        let ttt_lr_eta = inputs.ttt_lr_eta.clone();
        let ln_weight = inputs.ln_weight.clone();
        let ln_bias = inputs.ln_bias.clone();

        let result = forward_multi::<R, F>(inputs, config);

        let saved = TttSavedStateMulti {
            xq,
            xk,
            xv,
            weight_init,
            bias,
            token_eta,
            ttt_lr_eta,
            ln_weight,
            ln_bias,
            weight_checkpoints: result.weight_checkpoints,
            bias_checkpoints: result.bias_checkpoints,
        };

        (result.outputs, saved)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: TttSavedStateMulti<CubeTensor<R>>,
        grad_outputs: TttOutputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> TttInputs<CubeTensor<R>> {
        assert!(
            config.mini_batch_len > 0,
            "mini_batch_len must be set for TttTileMultiKernel backward"
        );

        let weight_checkpoints = saved.weight_checkpoints;
        let bias_checkpoints = saved.bias_checkpoints;

        let saved_tensors = TttSavedTensors {
            xq: saved.xq,
            xk: saved.xk,
            xv: saved.xv,
            weight_init: saved.weight_init,
            bias: saved.bias,
            token_eta: saved.token_eta,
            ttt_lr_eta: saved.ttt_lr_eta,
            ln_weight: saved.ln_weight,
            ln_bias: saved.ln_bias,
        };

        let grad_inputs = backward_multi::<R, F>(
            saved_tensors,
            weight_checkpoints,
            bias_checkpoints,
            grad_outputs.output,
            config,
        );

        TttInputs {
            xq: grad_inputs.grad_xq,
            xk: grad_inputs.grad_xk,
            xv: grad_inputs.grad_xv,
            weight: grad_inputs.grad_weight,
            bias: grad_inputs.grad_bias,
            token_eta: grad_inputs.grad_token_eta,
            ttt_lr_eta: grad_inputs.grad_ttt_lr_eta,
            ln_weight: grad_inputs.grad_ln_weight,
            ln_bias: grad_inputs.grad_ln_bias,
        }
    }
}
