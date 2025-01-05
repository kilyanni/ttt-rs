//! Fused TTT-Linear backward pass kernel (tiled implementation).
//!
//! # Mathematical Structure
//!
//! The forward pass computes:
//! ```text
//! z1 = XK @ W + b
//! grad_l = layer_norm_l2_grad(z1, XV - XK)
//! z1_bar = XQ @ W + b - (eta + eta*attn) @ grad_l
//! output = XQ + layer_norm(z1_bar)
//! W_out = W - last_eta * XK^T @ grad_l
//! b_out = b - last_eta @ grad_l
//! ```
//!
//! where `eta[i,j] = token_eta[i] * ttt_lr_eta[j]` is lower triangular,
//! and `attn = tril(XQ @ XK^T)`.
//!
//! The backward pass computes gradients in reverse order:
//! - Stage 4: LN backward (output -> z1_bar)
//! - Stage 3: Update backward (z1_bar -> W, b, grad_l dependencies)
//! - Stage 2: LN+L2 second derivative (grad_l -> Z1)
//! - Stage 1: MatMul backward (Z1 -> inputs)

mod kernel;
mod launch;
mod stage;
mod types;

#[allow(unused_imports)]
pub use kernel::*;
#[allow(unused_imports)]
pub use launch::*;
#[allow(unused_imports)]
pub use stage::*;
#[allow(unused_imports)]
pub use types::*;
