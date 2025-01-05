//! Core traits for fused GPU kernels.
//!
//! [`FusedKernel`] defines a hard boundary for Burn's fusion and autodiff systems.
//! At the boundary, wrapper types are stripped to concrete `CubeTensor`s, the kernel
//! runs, and results are rewrapped. The `impls` module provides [`FusedKernelBackend`]
//! implementations for `Autodiff<B>` and `Fusion<B>` that handle the unwrapping,
//! gradient tracking, and rewrapping boilerplate.
//!
//! # Usage
//!
//! 1. Define input/output/saved bundles using [`tensor_bundle!`](crate::tensor_bundle!)
//! 2. Implement [`FusedKernel`] with `forward_launch` and `backward_launch` on `CubeTensor`
//! 3. Call via `FusedKernelBackend::forward()`
//!
//! # Limitations
//!
//! Due to Burn's autodiff design, backward runs once per output tensor. If the kernel
//! has 3 outputs and all are used in the loss, `backward_launch` is called 3 times
//! with gradients accumulated. `SavedState` is cloned for each call.

use std::fmt::Debug;

use burn::tensor::{backend::Backend, ops::FloatTensor};
use burn_cubecl::{CubeRuntime, FloatElement, tensor::CubeTensor};

use crate::bundle::TensorBundle;

/// Trait for defining fused `CubeCL` kernels.
///
/// The tensor counts are encoded in the `Array` associated types of each bundle,
/// avoiding const generics on the trait itself.
pub trait FusedKernel: 'static + Send + Debug + Clone {
    type Inputs<T: Debug + Clone + Send>: TensorBundle<T>;
    type Outputs<T: Debug + Clone + Send>: TensorBundle<T>;
    /// State saved from forward pass for backward. Only includes what's actually needed.
    type SavedState<T: Debug + Clone + Send>: TensorBundle<T>;
    type Config: Debug + Clone + Send;

    /// Run the forward pass, returning outputs and the state needed for backward.
    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: Self::Inputs<CubeTensor<R>>,
        config: Self::Config,
    ) -> (
        Self::Outputs<CubeTensor<R>>,
        Self::SavedState<CubeTensor<R>>,
    );

    /// Run the backward pass using saved state and upstream gradients.
    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: Self::SavedState<CubeTensor<R>>,
        grad_outputs: Self::Outputs<CubeTensor<R>>,
        config: Self::Config,
    ) -> Self::Inputs<CubeTensor<R>>;
}

// =============================================================================
// Backend trait
// =============================================================================

/// Backend trait for a specific kernel.
/// Allows different backends to implement it.
pub trait FusedKernelBackend<K: FusedKernel>: Backend {
    fn forward(
        inputs: K::Inputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> (
        K::Outputs<FloatTensor<Self>>,
        K::SavedState<FloatTensor<Self>>,
    );

    fn backward(
        saved: K::SavedState<FloatTensor<Self>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>>;
}
