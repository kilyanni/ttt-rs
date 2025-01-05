//! Shared utilities for `cubecl_kernels`.

use burn_backend::Shape;
use burn_cubecl::{CubeRuntime, FloatElement, ops::numeric::empty_device, tensor::CubeTensor};

/// Create an empty tensor with the same client/device as the template.
pub fn empty_like<R: CubeRuntime, F: FloatElement>(
    template: &CubeTensor<R>,
    shape: impl Into<Shape>,
) -> CubeTensor<R> {
    empty_device::<R, F>(
        template.client.clone(),
        template.device.clone(),
        shape.into(),
    )
}
