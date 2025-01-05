use std::fmt::Debug;

use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};
use ttt_kernels::{kernel::FusedKernel, tensor_bundle};

use super::{
    backward::{backward, backward_multi},
    forward::{forward, forward_multi},
};
use crate::{
    FusedTttConfig,
    ttt::{TttInputs, TttNaiveKernel, TttNaiveMultiKernel, TttOutputs},
};

tensor_bundle! {
    /// Saved state for multi-stage naive backward pass.
    /// Includes per-stage weight/bias checkpoints from forward pass.
    pub struct TttSavedStateMulti {
        xq, xk, xv,
        weight_init, bias,
        token_eta, ttt_lr_eta,
        ln_weight, ln_bias,
        weight_checkpoints, bias_checkpoints
    }
}

impl FusedKernel for TttNaiveMultiKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type SavedState<T: Debug + Clone + Send> = TttSavedStateMulti<T>;
    type Config = FusedTttConfig;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        config: FusedTttConfig,
    ) -> (TttOutputs<CubeTensor<R>>, TttSavedStateMulti<CubeTensor<R>>) {
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
        let weight_checkpoints = saved.weight_checkpoints;
        let bias_checkpoints = saved.bias_checkpoints;

        let inputs = TttInputs {
            xq: saved.xq,
            xk: saved.xk,
            xv: saved.xv,
            weight: saved.weight_init,
            bias: saved.bias,
            token_eta: saved.token_eta,
            ttt_lr_eta: saved.ttt_lr_eta,
            ln_weight: saved.ln_weight,
            ln_bias: saved.ln_bias,
        };

        backward_multi::<R, F>(
            inputs,
            weight_checkpoints,
            bias_checkpoints,
            grad_outputs,
            config,
        )
    }
}

impl FusedKernel for TttNaiveKernel {
    type Inputs<T: Debug + Clone + Send> = TttInputs<T>;
    type Outputs<T: Debug + Clone + Send> = TttOutputs<T>;
    type SavedState<T: Debug + Clone + Send> = TttInputs<T>;
    type Config = f32;

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: TttInputs<CubeTensor<R>>,
        epsilon: f32,
    ) -> (TttOutputs<CubeTensor<R>>, TttInputs<CubeTensor<R>>) {
        let saved = inputs.clone();
        let outputs = forward::<R, F>(inputs, epsilon);
        (outputs, saved)
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: TttInputs<CubeTensor<R>>,
        grad_outputs: TttOutputs<CubeTensor<R>>,
        epsilon: f32,
    ) -> TttInputs<CubeTensor<R>> {
        backward::<R, F>(saved, grad_outputs, epsilon)
    }
}
