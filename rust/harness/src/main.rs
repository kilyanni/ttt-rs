//! TTT Training Harness CLI
//!
//! Coordinates parallel training runs, estimates VRAM, handles crashes/resume,
//! and auto-shuts down runpod.

use clap::{Parser, Subcommand};
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};
use ttt_harness::{
    config::HarnessConfig,
    runpod::Runpod,
    scheduler::Scheduler,
    state::{RunStatus, StateManager},
    vram::VramEstimator,
};

#[derive(Parser)]
#[command(name = "ttt-harness", about = "TTT Training Run Harness")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the harness
    Run {
        /// Path to harness.toml config file
        #[arg(short, long, default_value = "harness.toml")]
        config: String,

        /// Dry run mode (estimate only, no execution)
        #[arg(long)]
        dry_run: bool,

        /// Path to ttt binary (auto-detected if not specified)
        #[arg(long)]
        ttt_binary: Option<String>,
    },

    /// Show status of runs
    Status {
        /// Path to state file
        #[arg(short, long, default_value = "./harness_state.json")]
        state: String,
    },

    /// Reset specific runs to pending
    Reset {
        /// Path to state file
        #[arg(short, long, default_value = "./harness_state.json")]
        state: String,

        /// Run names to reset
        runs: Vec<String>,
    },

    /// Print VRAM estimates for all runs
    Estimate {
        /// Path to harness.toml config file
        #[arg(short, long, default_value = "harness.toml")]
        config: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tracing needs to be initialized with indicatif_layer to not clobber progress bars
    let indicatif_layer = IndicatifLayer::new();

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        )
        .with(indicatif_layer)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            config,
            dry_run,
            ttt_binary,
        } => {
            let harness_config = HarnessConfig::load(&config)?;
            let ttt_binary = ttt_binary
                .or_else(ttt_harness::runner::Runner::find_ttt_binary)
                .ok_or("Could not find ttt binary. Please specify --ttt-binary")?;

            let state_manager = StateManager::new(&harness_config.harness.state_file);
            let scheduler = Scheduler::new(harness_config.clone(), ttt_binary, state_manager);

            if dry_run {
                let result = scheduler.dry_run()?;
                println!("=== Dry Run Results ===");
                println!(
                    "VRAM: {:.2} / {:.2} GB ({:.1}%)",
                    result.used_vram_gb,
                    result.usable_vram_gb,
                    result.used_vram_gb / result.usable_vram_gb * 100.0
                );
                println!();

                println!("Will start {} runs:", result.scheduled.len());
                for run in &result.scheduled {
                    println!(
                        "  - {} ({:.2} GB){}",
                        run.name,
                        run.vram_estimate.total_gb(),
                        run.resume_epoch
                            .map(|e| format!(" [resume from epoch {}]", e))
                            .unwrap_or_default()
                    );
                }

                if !result.too_large.is_empty() {
                    println!();
                    println!("Too large for available VRAM:");
                    for name in &result.too_large {
                        println!("  - {name}");
                    }
                }
            } else {
                println!("Starting harness with config: {config}");
                let runpod = Runpod::new(harness_config.harness.runpod.enabled);

                // Check runpod config and fail early if misconfigured
                if let Some(warning) = runpod.check_config() {
                    eprintln!("ERROR: {warning}");
                    eprintln!("Fix the configuration or set [harness.runpod] enabled = false");
                    return Err(warning.into());
                }

                if runpod.is_available() {
                    println!("Runpod auto-shutdown: enabled");
                }

                let result = scheduler.run().await?;

                println!();
                println!("=== Harness Complete ===");
                println!("Total: {}", result.total);
                println!("Completed: {}", result.completed);
                println!("Failed: {}", result.failed);
                println!("Skipped: {}", result.skipped);

                // Auto-shutdown if enabled and all succeeded
                if runpod.is_available() && result.failed == 0 {
                    println!();
                    println!("All runs completed successfully. Shutting down runpod...");
                    if let Err(e) = runpod.stop() {
                        eprintln!("Warning: Failed to stop runpod: {e}");
                    }
                }
            }
        }

        Commands::Status { state } => {
            let state_manager = StateManager::new(&state);
            let harness_state = state_manager.load()?;

            if harness_state.runs.is_empty() {
                println!("No runs in state file.");
                return Ok(());
            }

            println!("=== Run Status ===");
            println!();

            let mut runs: Vec<_> = harness_state.runs.iter().collect();
            runs.sort_by_key(|(name, _)| *name);

            for (name, run) in runs {
                let status = match run.status {
                    RunStatus::Pending => "PENDING",
                    RunStatus::Running => "RUNNING",
                    RunStatus::Completed => "COMPLETED",
                    RunStatus::Failed => "FAILED",
                    RunStatus::Skipped => "SKIPPED",
                };

                print!("{name}: {status}");

                if let Some(pid) = run.pid {
                    print!(" (PID: {pid})");
                }

                if let Some(epoch) = run.checkpoint_epoch {
                    print!(" [checkpoint: epoch {epoch}]");
                }

                if run.retry_count > 0 {
                    print!(" [retries: {}]", run.retry_count);
                }

                println!();

                if !run.errors.is_empty() {
                    for error in &run.errors {
                        println!("  Error: {error}");
                    }
                }
            }
        }

        Commands::Reset { state, runs } => {
            let state_manager = StateManager::new(&state);

            for name in &runs {
                match state_manager.reset_run(name) {
                    Ok(true) => println!("Reset: {name}"),
                    Ok(false) => println!("Not found: {name}"),
                    Err(e) => eprintln!("Error resetting {name}: {e}"),
                }
            }
        }

        Commands::Estimate { config } => {
            let harness_config = HarnessConfig::load(&config)?;
            let estimator = VramEstimator::default();
            let runs = harness_config.runs.clone();

            println!("=== VRAM Estimates ===");
            println!(
                "Total VRAM: {:.2} GB (usable after margin: {:.2} GB)",
                harness_config.harness.total_vram_gb,
                harness_config.usable_vram_gb()
            );
            println!();

            let mut total_sequential = 0.0;

            for run in &runs {
                let estimate = estimator.estimate(run);
                println!("{}: {:.2} GB", run.name, estimate.total_gb());
                println!("  {}", estimate.breakdown());
                total_sequential += estimate.total_gb();
                println!();
            }

            println!("Total (sequential): {:.2} GB", total_sequential);

            // Show what can run in parallel
            let mut parallel_vram = 0.0;
            let mut parallel_runs = Vec::new();
            let usable = harness_config.usable_vram_gb();

            for run in &runs {
                let estimate = estimator.estimate(run);
                if parallel_vram + estimate.total_gb() <= usable {
                    parallel_vram += estimate.total_gb();
                    parallel_runs.push(run.name.clone());
                }
            }

            if parallel_runs.len() > 1 {
                println!();
                println!(
                    "Can run in parallel ({:.2} GB / {:.2} GB):",
                    parallel_vram, usable
                );
                for name in &parallel_runs {
                    println!("  - {name}");
                }
            }
        }
    }

    Ok(())
}
