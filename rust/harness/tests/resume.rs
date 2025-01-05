//! Integration tests for resume functionality.
//!
//! These tests verify that the harness correctly:
//! 1. Detects checkpoints when a run fails
//! 2. Passes --resume flag when retrying with a checkpoint
//! 3. Updates state with checkpoint info on failure

use std::{fs, path::Path};

use tempfile::tempdir;
use ttt_harness::state::{RunState, RunStatus, StateManager};

/// Helper to initialize a single run in state.
fn init_run(sm: &StateManager, name: &str, artifact_dir: &str) {
    sm.update(|state| {
        state
            .runs
            .insert(name.to_string(), RunState::new(artifact_dir.to_string()));
    })
    .unwrap();
}

/// Create fake checkpoint files to simulate a partial training run.
fn create_fake_checkpoints(artifact_dir: &Path, epoch: usize) {
    let checkpoint_dir = artifact_dir.join("checkpoint");
    fs::create_dir_all(&checkpoint_dir).unwrap();

    for e in 1..=epoch {
        fs::write(checkpoint_dir.join(format!("model-{e}.mpk")), b"fake").unwrap();
        fs::write(checkpoint_dir.join(format!("optim-{e}.mpk")), b"fake").unwrap();
        fs::write(checkpoint_dir.join(format!("scheduler-{e}.mpk")), b"fake").unwrap();
    }
}

#[test]
fn test_checkpoint_detection() {
    let dir = tempdir().unwrap();
    let artifact_dir = dir.path().join("artifacts/test-run");

    // No checkpoints yet
    assert_eq!(
        StateManager::find_checkpoint(artifact_dir.to_str().unwrap()),
        None
    );

    // Create checkpoint at epoch 3
    create_fake_checkpoints(&artifact_dir, 3);
    assert_eq!(
        StateManager::find_checkpoint(artifact_dir.to_str().unwrap()),
        Some(3)
    );

    // Add more checkpoints
    create_fake_checkpoints(&artifact_dir, 5);
    assert_eq!(
        StateManager::find_checkpoint(artifact_dir.to_str().unwrap()),
        Some(5)
    );
}

#[test]
fn test_mark_failed_preserves_checkpoint() {
    let dir = tempdir().unwrap();
    let state_file = dir.path().join("state.json");
    let artifact_dir = dir.path().join("artifacts/test-run");

    let sm = StateManager::new(state_file.to_str().unwrap());
    init_run(&sm, "test-run", artifact_dir.to_str().unwrap());

    // Mark started
    sm.mark_started("test-run", 12345).unwrap();

    // Simulate crash with checkpoint at epoch 2
    sm.mark_failed("test-run", "simulated crash", Some(2))
        .unwrap();

    // Verify state
    let state = sm.load().unwrap();
    let run = state.runs.get("test-run").unwrap();
    assert_eq!(run.status, RunStatus::Failed);
    assert_eq!(run.checkpoint_epoch, Some(2));
    assert_eq!(run.retry_count, 1);
}

#[test]
fn test_can_start_with_retries() {
    let dir = tempdir().unwrap();
    let state_file = dir.path().join("state.json");
    let artifact_dir = dir.path().join("artifacts/test-run");

    let sm = StateManager::new(state_file.to_str().unwrap());
    init_run(&sm, "test-run", artifact_dir.to_str().unwrap());

    // Initially can start
    let state = sm.load().unwrap();
    assert!(state.runs.get("test-run").unwrap().can_start(3));

    // After 1 failure, still can start (retry_count=1 < max_retries=3)
    sm.mark_started("test-run", 1).unwrap();
    sm.mark_failed("test-run", "error 1", None).unwrap();
    let state = sm.load().unwrap();
    assert!(state.runs.get("test-run").unwrap().can_start(3));

    // After 2 failures, still can start
    sm.mark_started("test-run", 2).unwrap();
    sm.mark_failed("test-run", "error 2", None).unwrap();
    let state = sm.load().unwrap();
    assert!(state.runs.get("test-run").unwrap().can_start(3));

    // After 3 failures, cannot start anymore
    sm.mark_started("test-run", 3).unwrap();
    sm.mark_failed("test-run", "error 3", None).unwrap();
    let state = sm.load().unwrap();
    assert!(!state.runs.get("test-run").unwrap().can_start(3));
}

#[test]
fn test_checkpoint_epoch_persists_across_retries() {
    let dir = tempdir().unwrap();
    let state_file = dir.path().join("state.json");
    let artifact_dir = dir.path().join("artifacts/test-run");

    let sm = StateManager::new(state_file.to_str().unwrap());
    init_run(&sm, "test-run", artifact_dir.to_str().unwrap());

    // First run fails at epoch 2
    sm.mark_started("test-run", 1).unwrap();
    sm.mark_failed("test-run", "crash 1", Some(2)).unwrap();

    let state = sm.load().unwrap();
    assert_eq!(
        state.runs.get("test-run").unwrap().checkpoint_epoch,
        Some(2)
    );

    // Second run fails at epoch 5 (progressed further)
    sm.mark_started("test-run", 2).unwrap();
    sm.mark_failed("test-run", "crash 2", Some(5)).unwrap();

    let state = sm.load().unwrap();
    assert_eq!(
        state.runs.get("test-run").unwrap().checkpoint_epoch,
        Some(5)
    );

    // Third run fails but no new checkpoint (None doesn't overwrite)
    sm.mark_started("test-run", 3).unwrap();
    sm.mark_failed("test-run", "crash 3", None).unwrap();

    let state = sm.load().unwrap();
    // Should still have epoch 5 from previous run
    assert_eq!(
        state.runs.get("test-run").unwrap().checkpoint_epoch,
        Some(5)
    );
}

/// Test that Runner passes --resume when a checkpoint exists.
/// Uses a mock bash script instead of the real ttt binary.
#[tokio::test]
#[ignore = "spawns subprocesses"]
async fn test_runner_passes_resume_flag() {
    use ttt_harness::runner::Runner;

    let dir = tempdir().unwrap();
    let state_file = dir.path().join("state.json");
    let artifact_dir = dir.path().join("artifacts/resume-run");
    let mock_bin = dir.path().join("mock_ttt");

    // Mock binary that dumps its args to a file, then exits 0
    let args_file = dir.path().join("captured_args.txt");
    let script = format!(
        "#!/usr/bin/env bash\necho \"$@\" > \"{}\"\nexit 0\n",
        args_file.display()
    );
    fs::write(&mock_bin, script).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&mock_bin, fs::Permissions::from_mode(0o755)).unwrap();
    }

    // Set up state with a checkpoint
    let sm = StateManager::new(state_file.to_str().unwrap());
    init_run(&sm, "resume-run", artifact_dir.to_str().unwrap());
    sm.mark_started("resume-run", 1).unwrap();
    sm.mark_failed("resume-run", "simulated crash", Some(3))
        .unwrap();

    // Create fake checkpoint files so find_checkpoint works
    create_fake_checkpoints(&artifact_dir, 3);

    // Load state and get the checkpoint epoch
    let state = sm.load().unwrap();
    let run_state = state.runs.get("resume-run").unwrap();
    let resume_epoch = run_state.checkpoint_epoch;
    assert_eq!(resume_epoch, Some(3));

    // Create a RunConfig
    let run_config = ttt_harness::RunConfig {
        name: "resume-run".to_string(),
        params: ttt_config::TrainParams {
            tokenizer: "gpt2".to_string(),
            ..Default::default()
        },
        out: Some(artifact_dir.to_str().unwrap().to_string()),
    };

    // Spawn with resume
    let runner = Runner::new(mock_bin.to_str().unwrap(), sm, None);
    let handle = runner.spawn(&run_config, resume_epoch).unwrap();
    let result = runner.wait(handle, None, None).await;
    assert!(result.success);

    // Verify --resume was in the args
    let captured = fs::read_to_string(&args_file).unwrap();
    assert!(
        captured.contains("--resume"),
        "Expected --resume in args, got: {captured}"
    );
    assert!(
        captured.contains(artifact_dir.to_str().unwrap()),
        "Expected artifact dir in --resume arg, got: {captured}"
    );
}

/// Test that Runner does NOT pass --resume when no checkpoint exists.
#[tokio::test]
#[ignore = "spawns subprocesses"]
async fn test_runner_no_resume_without_checkpoint() {
    use ttt_harness::runner::Runner;

    let dir = tempdir().unwrap();
    let state_file = dir.path().join("state.json");
    let artifact_dir = dir.path().join("artifacts/fresh-run");
    let mock_bin = dir.path().join("mock_ttt");

    let args_file = dir.path().join("captured_args.txt");
    let script = format!(
        "#!/usr/bin/env bash\necho \"$@\" > \"{}\"\nexit 0\n",
        args_file.display()
    );
    fs::write(&mock_bin, script).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&mock_bin, fs::Permissions::from_mode(0o755)).unwrap();
    }

    let sm = StateManager::new(state_file.to_str().unwrap());
    init_run(&sm, "fresh-run", artifact_dir.to_str().unwrap());

    let run_config = ttt_harness::RunConfig {
        name: "fresh-run".to_string(),
        params: ttt_config::TrainParams {
            tokenizer: "gpt2".to_string(),
            ..Default::default()
        },
        out: Some(artifact_dir.to_str().unwrap().to_string()),
    };

    // Spawn without resume (no checkpoint)
    let runner = Runner::new(mock_bin.to_str().unwrap(), sm, None);
    let handle = runner.spawn(&run_config, None).unwrap();
    let result = runner.wait(handle, None, None).await;
    assert!(result.success);

    // Verify --resume was NOT in the args
    let captured = fs::read_to_string(&args_file).unwrap();
    assert!(
        !captured.contains("--resume"),
        "Should NOT have --resume, got: {captured}"
    );
}
