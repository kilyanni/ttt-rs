#![warn(clippy::pedantic)]
#![allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::default_trait_access,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::type_complexity
)]

pub mod eval;
pub mod inference;
pub mod text_generation;
pub mod training;

pub use eval::{EvalResult, eval_pretokenized};
pub use inference::{TTTTextGenerator, generate, interactive};
pub use text_generation::{TTTTextGenerationConfig, TTTTextGenerationModel};
pub use training::{TTTTrainingConfig, train_dataset, train_dataset_pretokenized};
