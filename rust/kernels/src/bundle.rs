//! Tensor bundle abstraction for working with multiple tensor types.
//!
//! Burn has multiple tensor wrapper types: `CubeTensor` (raw GPU), `FloatTensor<Autodiff<B>>`
//! (gradient tracking), `FloatTensor<Fusion<B>>` (operation fusion), etc. A fused kernel
//! needs to work with all of them using a single struct definition.
//!
//! # The Problem
//!
//! A kernel's inputs might be `(xq, xk, xv, weight, bias)`. We need:
//! - A single struct definition that works with any tensor type
//! - Conversion between tensor types (e.g., unwrap autodiff → run kernel → rewrap)
//! - Type-level tensor count for generic gradient routing
//!
//! # Solution
//!
//! [`TensorBundle<T>`] is generic over the tensor type. The [`tensor_bundle!`] macro
//! generates a struct that implements it:
//!
//! ```ignore
//! tensor_bundle! {
//!     pub struct MyInputs { xq, xk, xv, weight, bias }
//! }
//!
//! // Same struct, different tensor types:
//! let cube_inputs: MyInputs<CubeTensor<R>> = ...;           // For kernel launch
//! let autodiff_inputs: MyInputs<FloatTensor<Autodiff<B>>> = ...; // With gradients
//! let fusion_inputs: MyInputs<FloatTensor<Fusion<B>>> = ...;     // For fusion
//!
//! // Convert between types with map():
//! let primitives = autodiff_inputs.map(|t| t.primitive);
//! ```
//!
//! The `Mapped<U>` associated type ensures the struct type is preserved across conversions.
//! The `Array` associated type encodes tensor count, avoiding const generics on
//! [`FusedKernel`](crate::FusedKernel).
//!
//! # Why the loose associated types?
//!
//! `Mapped<U>` is always `Self<U>` and `Array` is always `[T; N]`. These are expressed as
//! associated types rather than being hardcoded because Rust can't express higher-kinded
//! bounds like `Bundle: for<T> TensorBundle<T>`. The loose types let us write bounds like
//! `K::Inputs<CubeTensor>: TensorBundle<..., Mapped<Primitive> = K::Inputs<Primitive>>`.

use std::fmt::Debug;

/// Generic trait for tensor bundles.
///
/// The array size is encoded in the `Array` associated type rather than
/// as a const generic, allowing traits like `FusedKernel` to avoid const generics.
pub trait TensorBundle<T: Debug + Clone + Send>: Sized + Clone + Send + Debug {
    /// The array type for this bundle, e.g. `[T; 9]` for a 9-tensor bundle.
    type Array;
    /// The bundle type with a different element type.
    type Mapped<U: Debug + Clone + Send>: TensorBundle<U, Array = Self::ArrayMapped<U>>;
    /// The array type with a different element type.
    type ArrayMapped<U>;

    fn map<U: Debug + Clone + Send>(self, f: impl FnMut(T) -> U) -> Self::Mapped<U>;
    fn into_array(self) -> Self::Array;
    fn from_array(arr: Self::Array) -> Self;
}

/// Helper macro to replace a token with an expression (used for counting).
#[doc(hidden)]
#[macro_export]
macro_rules! __replace_expr {
    ($_t:tt, $sub:expr) => {
        $sub
    };
}

/// Declares a tensor bundle struct with automatic `TensorBundle` implementation.
///
/// # Example
/// ```ignore
/// tensor_bundle! {
///     /// My bundle of tensors
///     pub struct MyInputs { xq, xk, xv }
///     scalars { epsilon: f32 = 0.0 }
/// }
/// ```
///
/// This generates:
/// - The struct with all fields public
/// - `TensorBundle<T>` impl with map, `into_array`, `from_array`
/// - `HasClient` impl for Fusion (using first field)
/// - `<scalar>()` builder methods for each scalar
#[macro_export]
macro_rules! tensor_bundle {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident { $first_field:ident $(, $field:ident)* $(,)? }
        $(scalars { $($scalar:ident : $scalar_ty:ty = $scalar_default:expr),* $(,)? })?
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        $vis struct $name<T> {
            pub $first_field: T,
            $(pub $field: T,)*
            $($(
                pub $scalar: $scalar_ty,
            )*)?
        }

        impl<T: std::fmt::Debug + Clone + Send> $crate::TensorBundle<T> for $name<T> {
            type Array = [T; 1usize $(+ $crate::__replace_expr!($field, 1usize))*];
            type Mapped<U: std::fmt::Debug + Clone + Send> = $name<U>;
            type ArrayMapped<U> = [U; 1usize $(+ $crate::__replace_expr!($field, 1usize))*];

            fn map<U: std::fmt::Debug + Clone + Send>(self, mut f: impl FnMut(T) -> U) -> $name<U> {
                $name {
                    $first_field: f(self.$first_field),
                    $($field: f(self.$field),)*
                    $($($scalar: self.$scalar,)*)?
                }
            }

            fn into_array(self) -> [T; 1usize $(+ $crate::__replace_expr!($field, 1usize))*] {
                [self.$first_field $(, self.$field)*]
            }

            fn from_array(arr: [T; 1usize $(+ $crate::__replace_expr!($field, 1usize))*]) -> Self {
                let [$first_field $(, $field)*] = arr;
                $name {
                    $first_field,
                    $($field,)*
                    $($($scalar: $scalar_default,)*)?
                }
            }
        }

        impl<B: burn_fusion::FusionBackend>
            $crate::impls::HasClient<B>
            for $name<burn::tensor::ops::FloatTensor<burn_fusion::Fusion<B>>>
        {
            fn client(&self) -> &burn_fusion::client::GlobalFusionClient<B::FusionRuntime> {
                &self.$first_field.client
            }
        }

        $crate::tensor_bundle!(@setters $name $($(, $scalar : $scalar_ty)*)?);
    };

    // Generate setters for scalars
    (@setters $name:ident) => {};
    (@setters $name:ident, $($scalar:ident : $scalar_ty:ty),+) => {
        impl<T> $name<T> {
            $(
                #[must_use]
                pub fn $scalar(mut self, value: $scalar_ty) -> Self {
                    self.$scalar = value;
                    self
                }
            )+
        }
    };
}

pub use crate::tensor_bundle;
