//! Run state tracking and persistence.
//!
//! Tracks the status of each run and persists to JSON for crash recovery.

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Seek, SeekFrom},
    path::Path,
};

use fs2::FileExt;
use serde::{Deserialize, Serialize};

/// Status of a training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Run has not started yet.
    Pending,
    /// Run is currently executing.
    Running,
    /// Run completed successfully.
    Completed,
    /// Run failed (may be retried).
    Failed,
    /// Run was skipped (e.g., after max retries).
    Skipped,
}

/// State of a single run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunState {
    /// Current status.
    pub status: RunStatus,
    /// Artifact directory path.
    pub artifact_dir: String,
    /// Last completed checkpoint epoch (for resume).
    pub checkpoint_epoch: Option<usize>,
    /// Process ID when running (for crash detection).
    pub pid: Option<u32>,
    /// Number of retry attempts so far.
    pub retry_count: u32,
    /// Error messages from failures.
    #[serde(default)]
    pub errors: Vec<String>,
    /// Timestamp when run started.
    pub started_at: Option<String>,
    /// Timestamp when run completed/failed.
    pub finished_at: Option<String>,
}

impl RunState {
    /// Create a new pending run state.
    #[must_use]
    pub fn new(artifact_dir: String) -> Self {
        Self {
            status: RunStatus::Pending,
            artifact_dir,
            checkpoint_epoch: None,
            pid: None,
            retry_count: 0,
            errors: Vec::new(),
            started_at: None,
            finished_at: None,
        }
    }

    /// Check if this run can be started (pending or retryable).
    #[must_use]
    pub fn can_start(&self, max_retries: u32) -> bool {
        match self.status {
            RunStatus::Pending => true,
            RunStatus::Failed => self.retry_count < max_retries,
            _ => false,
        }
    }

    /// Check if this run is finished (completed, failed with max retries, or skipped).
    #[must_use]
    pub fn is_finished(&self, max_retries: u32) -> bool {
        match self.status {
            RunStatus::Completed | RunStatus::Skipped => true,
            RunStatus::Failed => self.retry_count >= max_retries,
            _ => false,
        }
    }
}

/// Persistent state for all runs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HarnessState {
    /// State of each run, keyed by run name.
    pub runs: HashMap<String, RunState>,
    /// Version for future compatibility.
    #[serde(default = "default_version")]
    pub version: u32,
}

fn default_version() -> u32 {
    1
}

/// Manages state persistence with file locking.
pub struct StateManager {
    /// Path to the state file.
    pub path: std::path::PathBuf,
}

impl StateManager {
    /// Create a new state manager for the given path.
    #[must_use]
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Load state from file, or create empty state if file doesn't exist.
    pub fn load(&self) -> Result<HarnessState, StateError> {
        if !self.path.exists() {
            return Ok(HarnessState::default());
        }

        let file = File::open(&self.path).map_err(|e| StateError::Io(self.path.clone(), e))?;
        file.lock_shared()
            .map_err(|e| StateError::Lock(self.path.clone(), e))?;

        let reader = BufReader::new(&file);
        let state = serde_json::from_reader(reader)
            .map_err(|e| StateError::Parse(self.path.clone(), e.to_string()))?;

        file.unlock()
            .map_err(|e| StateError::Lock(self.path.clone(), e))?;

        Ok(state)
    }

    /// Save state to file with exclusive lock.
    pub fn save(&self, state: &HarnessState) -> Result<(), StateError> {
        // Create parent directories if needed
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| StateError::Io(parent.to_path_buf(), e))?;
        }

        let file = File::create(&self.path).map_err(|e| StateError::Io(self.path.clone(), e))?;
        file.lock_exclusive()
            .map_err(|e| StateError::Lock(self.path.clone(), e))?;

        let writer = BufWriter::new(&file);
        serde_json::to_writer_pretty(writer, state)
            .map_err(|e| StateError::Write(self.path.clone(), e.to_string()))?;

        file.unlock()
            .map_err(|e| StateError::Lock(self.path.clone(), e))?;

        Ok(())
    }

    /// Update state atomically with a closure (holds lock for entire operation).
    pub fn update<F>(&self, f: F) -> Result<HarnessState, StateError>
    where
        F: FnOnce(&mut HarnessState),
    {
        // Create parent directories if needed
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| StateError::Io(parent.to_path_buf(), e))?;
        }

        // Open with read+write, create if needed
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)
            .map_err(|e| StateError::Io(self.path.clone(), e))?;

        // Hold exclusive lock for entire operation
        file.lock_exclusive()
            .map_err(|e| StateError::Lock(self.path.clone(), e))?;

        // Read current state (or default if empty/new file)
        let mut state: HarnessState = if file.metadata().map(|m| m.len()).unwrap_or(0) > 0 {
            let reader = BufReader::new(&file);
            serde_json::from_reader(reader)
                .map_err(|e| StateError::Parse(self.path.clone(), e.to_string()))?
        } else {
            HarnessState::default()
        };

        // Apply the update
        f(&mut state);

        // Truncate and write back
        file.set_len(0)
            .map_err(|e| StateError::Io(self.path.clone(), e))?;
        (&file)
            .seek(SeekFrom::Start(0))
            .map_err(|e| StateError::Io(self.path.clone(), e))?;

        let writer = BufWriter::new(&file);
        serde_json::to_writer_pretty(writer, &state)
            .map_err(|e| StateError::Write(self.path.clone(), e.to_string()))?;

        file.unlock()
            .map_err(|e| StateError::Lock(self.path.clone(), e))?;

        Ok(state)
    }

    /// Initialize state for a set of runs, preserving existing state.
    pub fn initialize_runs(
        &self,
        runs: &[crate::config::RunConfig],
    ) -> Result<HarnessState, StateError> {
        self.update(|state| {
            for run in runs {
                state
                    .runs
                    .entry(run.name.clone())
                    .or_insert_with(|| RunState::new(run.artifact_dir()));
            }
        })
    }

    /// Mark a run as started.
    pub fn mark_started(&self, name: &str, pid: u32) -> Result<(), StateError> {
        self.update(|state| {
            if let Some(run) = state.runs.get_mut(name) {
                run.status = RunStatus::Running;
                run.pid = Some(pid);
                run.started_at = Some(now_timestamp());
            }
        })?;
        Ok(())
    }

    /// Mark a run as completed.
    pub fn mark_completed(
        &self,
        name: &str,
        checkpoint_epoch: Option<usize>,
    ) -> Result<(), StateError> {
        self.update(|state| {
            if let Some(run) = state.runs.get_mut(name) {
                run.status = RunStatus::Completed;
                run.pid = None;
                run.checkpoint_epoch = checkpoint_epoch;
                run.finished_at = Some(now_timestamp());
            }
        })?;
        Ok(())
    }

    /// Mark a run as failed.
    pub fn mark_failed(
        &self,
        name: &str,
        error: &str,
        checkpoint: Option<usize>,
    ) -> Result<(), StateError> {
        self.update(|state| {
            if let Some(run) = state.runs.get_mut(name) {
                run.status = RunStatus::Failed;
                run.pid = None;
                run.retry_count += 1;
                run.errors.push(error.to_string());
                run.finished_at = Some(now_timestamp());

                if checkpoint.is_some() {
                    run.checkpoint_epoch = checkpoint;
                }
            }
        })?;
        Ok(())
    }

    /// Mark a run as skipped.
    pub fn mark_skipped(&self, name: &str, reason: &str) -> Result<(), StateError> {
        self.update(|state| {
            if let Some(run) = state.runs.get_mut(name) {
                run.status = RunStatus::Skipped;
                run.pid = None;
                run.errors.push(format!("Skipped: {reason}"));
                run.finished_at = Some(now_timestamp());
            }
        })?;
        Ok(())
    }

    /// Reset a run to pending state.
    pub fn reset_run(&self, name: &str) -> Result<bool, StateError> {
        let mut found = false;
        self.update(|state| {
            if let Some(run) = state.runs.get_mut(name) {
                run.status = RunStatus::Pending;
                run.pid = None;
                run.retry_count = 0;
                run.errors.clear();
                run.started_at = None;
                run.finished_at = None;
                found = true;
            }
        })?;
        Ok(found)
    }

    /// Detect crashed runs (status=Running but PID dead) and mark for retry.
    pub fn recover_crashed_runs(&self) -> Result<Vec<String>, StateError> {
        let mut crashed = Vec::new();

        self.update(|state| {
            for (name, run) in &mut state.runs {
                if run.status == RunStatus::Running {
                    let is_alive = run.pid.is_some_and(is_process_alive);
                    if !is_alive {
                        run.status = RunStatus::Failed;
                        run.pid = None;
                        run.errors
                            .push("Process crashed or harness restarted".to_string());
                        crashed.push(name.clone());
                    }
                }
            }
        })?;

        Ok(crashed)
    }

    /// Find the latest checkpoint in an artifact directory.
    #[must_use]
    pub fn find_checkpoint(artifact_dir: &str) -> Option<usize> {
        let checkpoint_dir = Path::new(artifact_dir).join("checkpoint");
        std::fs::read_dir(checkpoint_dir)
            .ok()?
            .filter_map(std::result::Result::ok)
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.strip_prefix("model-")?
                    .strip_suffix(".mpk")?
                    .parse()
                    .ok()
            })
            .max()
    }
}

/// Check if a process is alive by PID.
fn is_process_alive(pid: u32) -> bool {
    // On Unix, send signal 0 to check if process exists
    #[cfg(unix)]
    {
        // kill -0 checks if process exists without sending a signal
        // SAFETY: kill with signal 0 is safe and just checks process existence
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    #[cfg(not(unix))]
    {
        panic!("Non-Unix platform not supported")
    }
}

/// Get current time as Unix timestamp string.
#[must_use]
pub fn now_timestamp() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Errors that can occur with state management.
#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("failed to read/write state file {0}: {1}")]
    Io(std::path::PathBuf, std::io::Error),
    #[error("failed to lock state file {0}: {1}")]
    Lock(std::path::PathBuf, std::io::Error),
    #[error("failed to parse state file {0}: {1}")]
    Parse(std::path::PathBuf, String),
    #[error("failed to write state file {0}: {1}")]
    Write(std::path::PathBuf, String),
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_state_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("state.json");
        let manager = StateManager::new(&path);

        let mut state = HarnessState::default();
        state.runs.insert(
            "test-run".to_string(),
            RunState::new("./artifacts/test".to_string()),
        );

        manager.save(&state).unwrap();
        let loaded = manager.load().unwrap();

        assert_eq!(loaded.runs.len(), 1);
        assert!(loaded.runs.contains_key("test-run"));
    }

    #[test]
    fn test_mark_started() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("state.json");
        let manager = StateManager::new(&path);

        let mut state = HarnessState::default();
        state.runs.insert(
            "test-run".to_string(),
            RunState::new("./artifacts/test".to_string()),
        );
        manager.save(&state).unwrap();

        manager.mark_started("test-run", 12345).unwrap();

        let loaded = manager.load().unwrap();
        let run = loaded.runs.get("test-run").unwrap();
        assert_eq!(run.status, RunStatus::Running);
        assert_eq!(run.pid, Some(12345));
    }
}
