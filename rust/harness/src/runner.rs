//! Subprocess execution for training runs.

use std::{
    collections::VecDeque,
    path::Path,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use tokio::{
    fs::OpenOptions,
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, Command},
    sync::watch,
};

use crate::{
    config::RunConfig,
    state::{StateError, StateManager, now_timestamp},
};

/// Result of a training run.
#[derive(Debug)]
pub struct RunResult {
    /// Run name.
    pub name: String,
    /// Whether the run succeeded.
    pub success: bool,
    /// Exit code if available.
    pub exit_code: Option<i32>,
    /// Error message if failed.
    pub error: Option<String>,
    /// Final checkpoint epoch if available.
    pub checkpoint_epoch: Option<usize>,
}

/// Progress update parsed from training output.
#[derive(Debug, Clone, Default)]
pub struct ProgressUpdate {
    pub epoch: usize,
    pub epoch_total: usize,
    pub items_processed: usize,
    pub items_total: usize,
}

fn extract_field(s: &str, key: &str) -> Option<usize> {
    let idx = s.find(key)?;
    let rest = &s[idx + key.len()..];
    rest.trim_start_matches(|c: char| !c.is_ascii_digit())
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

impl ProgressUpdate {
    /// Parse from `TrainingProgress` debug output.
    fn parse(line: &str) -> Option<Self> {
        if !line.starts_with("TrainingProgress {") {
            return None;
        }

        Some(Self {
            epoch: extract_field(line, "epoch:")?,
            epoch_total: extract_field(line, "epoch_total:")?,
            items_processed: extract_field(line, "items_processed:")?,
            items_total: extract_field(line, "items_total:")?,
        })
    }
}

/// Manages subprocess execution for training runs.
pub struct Runner {
    /// Path to the ttt binary.
    ttt_binary: String,
    /// State manager for persistence.
    state_manager: StateManager,
    /// `RUST_LOG` value for child processes.
    rust_log: Option<String>,
}

impl Runner {
    /// Create a new runner.
    #[must_use]
    pub fn new(
        ttt_binary: impl Into<String>,
        state_manager: StateManager,
        rust_log: Option<String>,
    ) -> Self {
        Self {
            ttt_binary: ttt_binary.into(),
            state_manager,
            rust_log,
        }
    }

    /// Find the ttt binary.
    #[must_use]
    pub fn find_ttt_binary() -> Option<String> {
        // Try common locations
        let candidates = ["./target/release/ttt", "./target/debug/ttt", "ttt"];

        for candidate in candidates {
            if std::path::Path::new(candidate).exists() {
                return Some(candidate.to_string());
            }
        }

        // Try which
        if let Ok(output) = std::process::Command::new("which").arg("ttt").output()
            && output.status.success()
        {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }

        None
    }

    /// Spawn a training run as a subprocess.
    pub fn spawn(
        &self,
        run: &RunConfig,
        resume_epoch: Option<usize>,
    ) -> Result<RunHandle, RunError> {
        let mut args = run.to_args();

        // Add resume flag if we have a checkpoint
        if let Some(epoch) = resume_epoch {
            tracing::info!("Resuming {} from epoch {}", run.name, epoch);
            args.push("--resume".to_string());
            args.push(run.artifact_dir());
        }

        tracing::debug!("Spawning: {} {}", self.ttt_binary, args.join(" "));

        let mut cmd = Command::new(&self.ttt_binary);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(ref rust_log) = self.rust_log {
            cmd.env("RUST_LOG", rust_log);
        }
        let child = cmd
            .spawn()
            .map_err(|e| RunError::Spawn(run.name.clone(), e))?;

        let pid = child
            .id()
            .ok_or_else(|| RunError::Spawn(run.name.clone(), std::io::Error::other("no PID")))?;

        // Mark as started in state
        self.state_manager
            .mark_started(&run.name, pid)
            .map_err(RunError::State)?;

        Ok(RunHandle {
            name: run.name.clone(),
            artifact_dir: run.artifact_dir(),
            child,
            pid,
        })
    }

    /// Wait for a run to complete and update state, sending progress updates.
    ///
    /// If `last_activity` is provided, it is updated (unix seconds) on every
    /// stdout/stderr line so an external watchdog can detect idle processes.
    pub async fn wait(
        &self,
        mut handle: RunHandle,
        progress_tx: Option<Arc<watch::Sender<ProgressUpdate>>>,
        last_activity: Option<Arc<AtomicU64>>,
    ) -> RunResult {
        // Create log directory
        let log_dir = Path::new(&handle.artifact_dir);
        let _ = tokio::fs::create_dir_all(log_dir).await;

        let stdout_path = log_dir.join("stdout.log");
        let stderr_path = log_dir.join("stderr.log");

        let timestamp = now_timestamp();

        // Stream stdout to file and parse for progress
        let stdout = handle.child.stdout.take();
        let stdout_task = if let Some(stdout) = stdout {
            let path = stdout_path.clone();
            let ts = timestamp.clone();
            let activity = last_activity.clone();
            Some(tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .ok();

                // Write retry separator if file already has content
                if let Some(ref mut f) = file
                    && f.metadata().await.is_ok_and(|m| m.len() > 0)
                {
                    let _ = f
                        .write_all(format!("\n--- retry at {ts} ---\n\n").as_bytes())
                        .await;
                }

                while let Ok(Some(line)) = lines.next_line().await {
                    if let Some(ref a) = activity {
                        a.store(unix_now(), Ordering::Relaxed);
                    }
                    if let Some(ref mut f) = file {
                        let _ = f.write_all(line.as_bytes()).await;
                        let _ = f.write_all(b"\n").await;
                    }

                    // Parse progress from output
                    if let Some(ref tx) = progress_tx
                        && let Some(update) = ProgressUpdate::parse(&line)
                    {
                        let _ = tx.send(update);
                    }
                }
            }))
        } else {
            None
        };

        // Stream stderr to file and keep last 20 lines for error reporting
        let stderr = handle.child.stderr.take();
        let stderr_task = if let Some(stderr) = stderr {
            let path = stderr_path.clone();
            let ts = timestamp;
            let activity = last_activity;
            Some(tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .ok();

                // Write retry separator if file already has content
                if let Some(ref mut f) = file
                    && f.metadata().await.is_ok_and(|m| m.len() > 0)
                {
                    let _ = f
                        .write_all(format!("\n--- retry at {ts} ---\n\n").as_bytes())
                        .await;
                }

                let mut tail = VecDeque::with_capacity(20);
                while let Ok(Some(line)) = lines.next_line().await {
                    if let Some(ref a) = activity {
                        a.store(unix_now(), Ordering::Relaxed);
                    }
                    if let Some(ref mut f) = file {
                        let _ = f.write_all(line.as_bytes()).await;
                        let _ = f.write_all(b"\n").await;
                    }
                    if tail.len() >= 20 {
                        tail.pop_front();
                    }
                    tail.push_back(line);
                }
                tail.into_iter().collect::<Vec<_>>()
            }))
        } else {
            None
        };

        // Wait for process
        let status = handle.child.wait().await;

        // Wait for log tasks
        if let Some(task) = stdout_task {
            let _ = task.await;
        }
        let stderr_output = if let Some(task) = stderr_task {
            task.await.ok()
        } else {
            None
        };

        match status {
            Ok(status) if status.success() => {
                let checkpoint = StateManager::find_checkpoint(&handle.artifact_dir);
                if let Err(e) = self.state_manager.mark_completed(&handle.name, checkpoint) {
                    tracing::error!("Failed to mark {} as completed: {}", handle.name, e);
                }
                RunResult {
                    name: handle.name,
                    success: true,
                    exit_code: status.code(),
                    error: None,
                    checkpoint_epoch: checkpoint,
                }
            }
            Ok(status) => {
                let error_msg = stderr_output.map_or_else(
                    || format!("Exit code: {:?}", status.code()),
                    |lines| lines.join("\n"),
                );
                let checkpoint = StateManager::find_checkpoint(&handle.artifact_dir);
                if let Err(e) = self
                    .state_manager
                    .mark_failed(&handle.name, &error_msg, checkpoint)
                {
                    tracing::error!("Failed to mark {} as failed: {}", handle.name, e);
                }
                RunResult {
                    name: handle.name,
                    success: false,
                    exit_code: status.code(),
                    error: Some(error_msg),
                    checkpoint_epoch: checkpoint,
                }
            }
            Err(e) => {
                let error_msg = format!("Process error: {e}");
                let checkpoint = StateManager::find_checkpoint(&handle.artifact_dir);
                if let Err(e) = self
                    .state_manager
                    .mark_failed(&handle.name, &error_msg, checkpoint)
                {
                    tracing::error!("Failed to mark {} as failed: {}", handle.name, e);
                }
                RunResult {
                    name: handle.name,
                    success: false,
                    exit_code: None,
                    error: Some(error_msg),
                    checkpoint_epoch: checkpoint,
                }
            }
        }
    }

    /// Get the state manager.
    #[must_use]
    pub fn state_manager(&self) -> &StateManager {
        &self.state_manager
    }
}

/// Get current unix timestamp in seconds.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Create a new last-activity tracker initialized to now.
#[must_use]
pub fn new_activity_tracker() -> Arc<AtomicU64> {
    Arc::new(AtomicU64::new(unix_now()))
}

/// Handle to a running subprocess.
pub struct RunHandle {
    /// Run name.
    pub name: String,
    /// Artifact directory.
    pub artifact_dir: String,
    /// Child process.
    child: Child,
    /// Process ID.
    pub pid: u32,
}

/// Errors that can occur when running a subprocess.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("failed to spawn process for {0}: {1}")]
    Spawn(String, std::io::Error),
    #[error("state error: {0}")]
    State(#[from] StateError),
}
