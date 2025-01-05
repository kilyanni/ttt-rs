//! High-level API for the tiled TTT-Linear forward kernel.

use burn::tensor::{Tensor, TensorPrimitive};
use ttt_kernels::FusedKernelBackend;

use super::launch::{TttTileKernel, TttTileMultiKernel};
use crate::{FusedTttBackend, FusedTttConfig, ttt::TttInputs};

pub fn default_threads(mini_batch_len: usize, head_dim: usize) -> usize {
    match (mini_batch_len, head_dim) {
        (8, 32) | (8, 64) | (16, 32) | (16, 64) | (32, 32) => 64,
        _ => panic!(
            "No default thread count for tile config: mini_batch_len={mini_batch_len}, head_dim={head_dim}"
        ),
    }
}

pub fn fused_ttt_tile_forward<B: FusedTttBackend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    xv: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Tensor<B, 3>,
    token_eta: Tensor<B, 1>,
    ttt_lr_eta: Tensor<B, 3>,
    ln_weight: Tensor<B, 2>,
    ln_bias: Tensor<B, 2>,
    config: FusedTttConfig,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
    let inputs = TttInputs {
        xq: xq.into_primitive().tensor(),
        xk: xk.into_primitive().tensor(),
        xv: xv.into_primitive().tensor(),
        weight: weight.into_primitive().tensor(),
        bias: bias.into_primitive().tensor(),
        token_eta: token_eta.into_primitive().tensor(),
        ttt_lr_eta: ttt_lr_eta.into_primitive().tensor(),
        ln_weight: ln_weight.into_primitive().tensor(),
        ln_bias: ln_bias.into_primitive().tensor(),
    };

    let (outputs, _saved) = <B as FusedKernelBackend<TttTileKernel>>::forward(inputs, config);

    (
        Tensor::from_primitive(TensorPrimitive::Float(outputs.output)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.weight)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.bias)),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fused_ttt_tile_forward_multi<B: FusedTttBackend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    xv: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Tensor<B, 3>,
    token_eta: Tensor<B, 1>,
    ttt_lr_eta: Tensor<B, 3>,
    ln_weight: Tensor<B, 2>,
    ln_bias: Tensor<B, 2>,
    config: FusedTttConfig,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
    let inputs = TttInputs {
        xq: xq.into_primitive().tensor(),
        xk: xk.into_primitive().tensor(),
        xv: xv.into_primitive().tensor(),
        weight: weight.into_primitive().tensor(),
        bias: bias.into_primitive().tensor(),
        token_eta: token_eta.into_primitive().tensor(),
        ttt_lr_eta: ttt_lr_eta.into_primitive().tensor(),
        ln_weight: ln_weight.into_primitive().tensor(),
        ln_bias: ln_bias.into_primitive().tensor(),
    };

    let (outputs, _saved) = <B as FusedKernelBackend<TttTileMultiKernel>>::forward(inputs, config);

    (
        Tensor::from_primitive(TensorPrimitive::Float(outputs.output)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.weight)),
        Tensor::from_primitive(TensorPrimitive::Float(outputs.bias)),
    )
}
