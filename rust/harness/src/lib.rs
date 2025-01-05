#![warn(clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

//! TTT Training Harness
//!
//! Coordinates parallel training runs, estimates VRAM, handles crashes/resume,
//! and auto-shuts down runpod.

pub mod config;
pub mod runner;
pub mod runpod;
pub mod scheduler;
pub mod state;
pub mod vram;

pub use config::{HarnessConfig, RunConfig};
pub use runner::Runner;
pub use scheduler::Scheduler;
pub use state::{RunState, RunStatus, StateManager};
pub use vram::VramEstimator;
