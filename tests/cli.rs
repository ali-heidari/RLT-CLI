//! End-to-end tests for the CLI surface.
//!
//! Every test runs the built binary in its own temporary directory. That is not
//! just isolation: checkpoints are written to a relative `./models`, and the
//! library panics if its global config is initialised twice in one process
//! (see `docs/found-issues.md`, issue 6), so a subprocess per run is the only
//! way to exercise more than one configuration.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// A config small enough to train quickly, but large enough to save a model.
///
/// `interval_secs = 0` matters: inference sleeps that long between samples, so
/// any non-zero value makes these tests take minutes.
const CONFIG: &str = r#"
interval_secs = 0
batch_size = 8
total_batches = 2
input_number = 6
output_number = 3
hidden_layers = 8
reply_capacity = 128
log_interval = 1000
model_name = "test-model.json"
"#;

/// The path the library actually writes, given `model_name = "test-model.json"`.
const CHECKPOINT: &str = "models/test-model.json.test-model.json";

/// A workspace with a config file and a dataset of `rows` numeric rows.
fn workspace(rows: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("Config.toml"), CONFIG).unwrap();

    let data: String = (0..rows)
        .map(|i| {
            let base = (i % 97) as f64 / 97.0;
            format!(
                "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                base,
                1.0 - base,
                base / 2.0,
                base * 0.9,
                base * 0.3,
                1.0 - base / 2.0
            )
        })
        .collect();
    fs::write(dir.path().join("data.csv"), data).unwrap();

    dir
}

fn rlt(dir: &TempDir) -> Command {
    let mut command = Command::cargo_bin("RLT-CLI").unwrap();
    command.current_dir(dir.path());
    command
}

#[test]
fn train_writes_a_checkpoint() {
    let dir = workspace(400);

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv"])
        .assert()
        .success();

    let checkpoint = dir.path().join(CHECKPOINT);
    assert!(checkpoint.is_file(), "no checkpoint at {}", CHECKPOINT);
    assert!(
        checkpoint.metadata().unwrap().len() > 0,
        "checkpoint is empty"
    );
}

#[test]
fn train_then_infer_then_export() {
    let dir = workspace(400);

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv"])
        .assert()
        .success();

    rlt(&dir)
        .args(["infer", "--dataset", "./data.csv"])
        .assert()
        .success();

    rlt(&dir)
        .args(["export", "--output", "./exported/model.json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Model exported to"));

    let exported = dir.path().join("exported/model.json");
    assert!(exported.is_file(), "export produced no file");
    assert_eq!(
        fs::read(&exported).unwrap(),
        fs::read(dir.path().join(CHECKPOINT)).unwrap(),
        "exported file must match the checkpoint"
    );
}

#[test]
fn inferring_without_a_checkpoint_fails_instead_of_using_random_weights() {
    let dir = workspace(20);

    rlt(&dir)
        .args([
            "infer",
            "--dataset",
            "./data.csv",
            "--model-name",
            "ghost.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("checkpoint not found"));
}

#[test]
fn a_finite_dataset_terminates_training() {
    // Regression: an exhausted dataset used to yield zeros forever, so training
    // never ended. The timeout is the assertion.
    let dir = workspace(30);

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv"])
        .timeout(std::time::Duration::from_secs(30))
        .assert()
        .success();
}

#[test]
fn it_runs_without_a_config_file() {
    // Regression: the CLI defaulted --config to Config.toml and then failed
    // with a bare OS error when that file did not exist.
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("data.csv"), "1,2,3,4,5,6,7,8\n").unwrap();

    rlt(&dir)
        .args([
            "train",
            "--dataset",
            "./data.csv",
            "--model-name",
            "m.json",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("No config file found"));
}

#[test]
fn an_explicitly_requested_config_must_exist() {
    let dir = workspace(10);

    rlt(&dir)
        .args(["--config", "./missing.toml", "train"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "config file not found: ./missing.toml",
        ));
}

#[test]
fn a_malformed_config_names_the_file() {
    let dir = workspace(10);
    fs::write(dir.path().join("bad.toml"), "batch_size = \"lots\"\n").unwrap();

    rlt(&dir)
        .args(["--config", "./bad.toml", "train", "--dry-run"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("failed to parse config file"));
}

#[test]
fn config_file_values_reach_the_run() {
    // Regression: clap defaults used to mask every value in the config file.
    let dir = workspace(10);

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("batch size: 8 (Config.toml)"));
}

#[test]
fn a_flag_overrides_the_config_file() {
    let dir = workspace(10);

    rlt(&dir)
        .args([
            "train",
            "--dataset",
            "./data.csv",
            "--batch-size",
            "99",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("batch size: 99 (flag)"));
}

#[test]
fn silent_prints_nothing_at_all() {
    let dir = workspace(30);

    let output = rlt(&dir)
        .args(["--silent", "train", "--dataset", "./data.csv"])
        .output()
        .unwrap();

    assert!(
        output.stdout.is_empty() && output.stderr.is_empty(),
        "--silent leaked {} bytes of stdout and {} of stderr",
        output.stdout.len(),
        output.stderr.len()
    );
}

#[test]
fn dry_run_validates_the_configuration() {
    let dir = workspace(10);

    rlt(&dir)
        .args(["train", "--dataset", "./missing.csv", "--dry-run"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("dataset not found"));

    rlt(&dir)
        .args([
            "train",
            "--dataset",
            "./data.csv",
            "--reward-script",
            "./missing.py",
            "--dry-run",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("reward script not found"));
}

#[test]
fn a_width_mismatch_is_reported_before_training() {
    let dir = workspace(10);
    fs::write(dir.path().join("narrow.csv"), "1,2,3\n4,5,6\n").unwrap();

    rlt(&dir)
        .args(["train", "--dataset", "./narrow.csv"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("feature width mismatch"));
}

#[test]
fn a_bad_field_is_skipped_rather_than_zeroed() {
    let dir = workspace(400);
    let mut data = fs::read_to_string(dir.path().join("data.csv")).unwrap();
    // The bad row goes second: the first row is validated up front, and a
    // dataset whose very first row is unreadable is a different failure.
    data.insert_str(0, "0.1,0.2,0.3,0.4,0.5,0.6\n0.1,0.2,n/a,0.4,0.5,0.6\n");
    fs::write(dir.path().join("data.csv"), data).unwrap();

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv"])
        .assert()
        .success()
        .stderr(predicate::str::contains("'n/a' is not a number"));
}

#[test]
fn a_header_row_needs_the_flag() {
    let dir = workspace(400);
    let mut data = fs::read_to_string(dir.path().join("data.csv")).unwrap();
    data.insert_str(0, "cpu,mem,net,pow,lat,eff\n");
    fs::write(dir.path().join("headed.csv"), &data).unwrap();

    rlt(&dir)
        .args(["train", "--dataset", "./headed.csv"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("'cpu' is not a number"));

    rlt(&dir)
        .args(["train", "--dataset", "./headed.csv", "--has-header"])
        .assert()
        .success();
}

#[test]
fn export_refuses_a_checkpoint_that_holds_no_model() {
    let dir = workspace(10);
    fs::create_dir_all(dir.path().join("models")).unwrap();
    fs::write(dir.path().join(CHECKPOINT), "").unwrap();

    rlt(&dir)
        .args(["export", "--output", "./exported/model.json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("checkpoint is empty"));

    assert!(
        !Path::new(&dir.path().join("exported/model.json")).exists(),
        "a failed export must not leave an output file"
    );
}

#[test]
fn a_python_provider_serves_many_samples_from_one_interpreter() {
    let dir = workspace(10);
    fs::write(
        dir.path().join("provider.py"),
        "import json, sys\n\
         for line in sys.stdin:\n\
         \x20   print(json.dumps([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), flush=True)\n",
    )
    .unwrap();

    rlt(&dir)
        .args(["train", "--dataset", "./provider.py"])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success();

    assert!(dir.path().join(CHECKPOINT).is_file());
}

#[test]
fn a_python_provider_that_dies_is_reported() {
    let dir = workspace(10);
    fs::write(
        dir.path().join("broken.py"),
        "import sys\nsys.exit('provider failed to start')\n",
    )
    .unwrap();

    rlt(&dir)
        .args(["train", "--dataset", "./broken.py"])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .failure()
        .stderr(predicate::str::contains("without answering"));
}

#[test]
fn help_lists_the_commands_and_no_dead_flags() {
    let dir = workspace(1);

    rlt(&dir)
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("train"))
        .stdout(predicate::str::contains("infer"))
        .stdout(predicate::str::contains("export"));

    // --learning-rate was accepted and silently discarded; it must stay gone.
    rlt(&dir)
        .args(["train", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("learning-rate").not());
}
