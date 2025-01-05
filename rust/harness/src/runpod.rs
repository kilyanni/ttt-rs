//! Runpod shutdown integration.
//!
//! Uses runpodctl CLI (pre-authenticated inside pod) to stop the pod
//! when all runs are complete.

use std::{env, process::Command};

/// Runpod integration for auto-shutdown.
pub struct Runpod {
    /// Pod ID from environment.
    pod_id: Option<String>,
    /// Whether shutdown is enabled in config.
    enabled: bool,
    /// Whether runpodctl CLI is available.
    has_cli: bool,
}

impl Runpod {
    /// Create a new Runpod integration.
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        let pod_id = env::var("RUNPOD_POD_ID").ok();
        let has_cli = Self::check_runpodctl();
        Self {
            pod_id,
            enabled,
            has_cli,
        }
    }

    /// Check if Runpod shutdown is fully available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.enabled && self.pod_id.is_some() && self.has_cli
    }

    /// Check configuration and return any problems.
    /// Returns None if everything is OK, Some(warning) if there's an issue.
    #[must_use]
    pub fn check_config(&self) -> Option<String> {
        if !self.enabled {
            return None; // Disabled is fine
        }

        let mut problems = Vec::new();

        if self.pod_id.is_none() {
            problems.push("RUNPOD_POD_ID environment variable not set");
        }

        if !self.has_cli {
            problems.push("runpodctl CLI not found in PATH");
        }

        if problems.is_empty() {
            None
        } else {
            Some(format!(
                "Runpod auto-shutdown enabled but won't work: {}",
                problems.join(", ")
            ))
        }
    }

    /// Check if runpodctl is installed.
    fn check_runpodctl() -> bool {
        Command::new("which")
            .arg("runpodctl")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    /// Stop the pod.
    pub fn stop(&self) -> Result<(), RunpodError> {
        if !self.enabled {
            return Err(RunpodError::Disabled);
        }

        let pod_id = self.pod_id.as_ref().ok_or(RunpodError::NoPodId)?;

        tracing::info!("Stopping runpod {}", pod_id);

        let output = Command::new("runpodctl")
            .args(["stop", "pod", pod_id])
            .output()
            .map_err(|e| RunpodError::Command(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(RunpodError::Command(stderr.into_owned()));
        }

        Ok(())
    }
}

/// Errors that can occur with Runpod operations.
#[derive(Debug, thiserror::Error)]
pub enum RunpodError {
    #[error("runpod shutdown is disabled")]
    Disabled,
    #[error("RUNPOD_POD_ID not set")]
    NoPodId,
    #[error("runpodctl command failed: {0}")]
    Command(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled() {
        let runpod = Runpod::new(false);
        assert!(!runpod.is_available());
        assert!(runpod.check_config().is_none()); // No warning when disabled
    }
}
