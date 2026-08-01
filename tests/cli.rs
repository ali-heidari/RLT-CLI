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
    let mut command = Command::cargo_bin("rlt").unwrap();
    command.current_dir(dir.path());
    command
}

/// A workspace whose config says nothing about `interval_secs`, so the
/// dataset-derived default applies.
fn workspace_without_interval(rows: usize) -> TempDir {
    let dir = workspace(rows);
    let config = CONFIG.replace("interval_secs = 0\n", "");
    fs::write(dir.path().join("Config.toml"), config).unwrap();
    dir
}

/// A workspace holding a trained checkpoint and a three-row `infer.csv`.
///
/// Inference runs to the end of its dataset, so a small separate file keeps the
/// assertions on the emitted records exact.
fn trained_workspace() -> TempDir {
    let dir = workspace(400);
    fs::write(
        dir.path().join("infer.csv"),
        "0.1,0.2,0.3,0.4,0.5,0.6\n0.2,0.3,0.4,0.5,0.6,0.7\n0.3,0.4,0.5,0.6,0.7,0.8\n",
    )
    .unwrap();

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv"])
        .assert()
        .success();

    dir
}

#[test]
fn init_scaffolds_a_project_that_trains_and_evaluates() {
    // The quickstart, end to end, on a fresh empty directory. This asserts the
    // three commands work together — not that the policy wins; on the pinned
    // library revision it does not (docs/found-issues.md issue 8).
    let dir = TempDir::new().unwrap();

    rlt(&dir)
        .arg("init")
        .assert()
        .success()
        .stderr(predicate::str::contains("Config.toml"));

    for scaffolded in [
        "Config.toml",
        "scripts/reward.py",
        "scripts/heuristic.py",
        "data/train.csv",
        "data/holdout.csv",
    ] {
        assert!(
            dir.path().join(scaffolded).is_file(),
            "init did not write {}",
            scaffolded
        );
    }

    // The scaffolded config has to be self-sufficient apart from consent: it
    // names a reward script, and these tests have no terminal to confirm at.
    rlt(&dir)
        .args(["--allow-scripts", "train"])
        .timeout(std::time::Duration::from_secs(120))
        .assert()
        .success();

    rlt(&dir)
        .args([
            "--allow-scripts",
            "eval",
            "--dataset",
            "./data/holdout.csv",
            "--baseline",
            "./scripts/heuristic.py",
        ])
        .timeout(std::time::Duration::from_secs(120))
        .assert()
        .success()
        .stdout(predicate::str::contains("Evaluated 400 sample(s)"))
        .stdout(predicate::str::contains("Baseline: ./scripts/heuristic.py"));
}

#[test]
fn init_refuses_to_overwrite_without_force() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("Config.toml"), "mine\n").unwrap();

    rlt(&dir)
        .arg("init")
        .assert()
        .failure()
        .stderr(predicate::str::contains("already exist"));

    assert_eq!(
        fs::read_to_string(dir.path().join("Config.toml")).unwrap(),
        "mine\n",
        "a refused init must leave the existing file alone"
    );

    rlt(&dir).args(["init", "--force"]).assert().success();
    assert!(fs::read_to_string(dir.path().join("Config.toml"))
        .unwrap()
        .contains("dataset"));
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
        .stderr(predicate::str::contains("Model exported to"));

    let exported = dir.path().join("exported/model.json");
    assert!(exported.is_file(), "export produced no file");
    assert_eq!(
        fs::read(&exported).unwrap(),
        fs::read(dir.path().join(CHECKPOINT)).unwrap(),
        "exported file must match the checkpoint"
    );
}

#[test]
fn infer_emits_one_action_per_sample() {
    // Regression: both provider branches computed actions and then threw them
    // away. The CSV callback discarded them outright; the Python one logged
    // them at debug level, which is off by default. `infer` did its work and
    // printed nothing.
    let dir = trained_workspace();

    let stdout = rlt(&dir)
        .args(["infer", "--dataset", "./infer.csv"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout).unwrap();

    // Every line, not just the ones that look like records: stdout carries the
    // command's result and nothing else, so `rlt infer | jq` works. The banner
    // and settings block go to stderr — filtering for `{` here is exactly what
    // hid the fact that they used to share this stream.
    //
    // `EMPTY INPUT` is excluded by name because the library prints it straight
    // to stdout when a data source ends, past every filter this CLI configures
    // (docs/found-issues.md issue 5). The exclusion is deliberately narrow:
    // anything else that is not a record still fails.
    let records: Vec<&str> = stdout
        .lines()
        .filter(|line| !line.is_empty() && *line != "EMPTY INPUT")
        .collect();
    assert!(
        records.iter().all(|line| line.starts_with('{')),
        "stdout must be records only:\n{}",
        stdout
    );

    assert_eq!(records.len(), 3, "expected one record per row:\n{}", stdout);
    assert!(records[0].contains("\"row\":1"), "{}", records[0]);
    assert!(records[0].contains("\"action\":"), "{}", records[0]);
    assert!(records[2].contains("\"row\":3"), "{}", records[2]);
    assert!(
        !records.iter().any(|record| record.contains("features")),
        "features must be opt-in:\n{}",
        stdout
    );
}

#[test]
fn inference_over_a_file_does_not_poll() {
    // Regression: inference slept 10s between samples whatever the dataset, so
    // a 400-row CSV took 67 minutes. The e2e suite only passed because every
    // config it wrote set interval_secs = 0 by hand. The timeout is the
    // assertion; the settings line proves which rule applied.
    let dir = workspace_without_interval(400);

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv"])
        .assert()
        .success();

    rlt(&dir)
        .args(["infer", "--dataset", "./data.csv"])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success()
        .stderr(predicate::str::contains("interval: 0s (file dataset)"));
}

#[test]
fn infer_with_features_includes_the_input_row() {
    let dir = trained_workspace();

    rlt(&dir)
        .args(["infer", "--dataset", "./infer.csv", "--with-features"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"features\":[0.1,"));
}

#[test]
fn infer_can_write_its_actions_to_a_file() {
    let dir = trained_workspace();

    rlt(&dir)
        .args([
            "infer",
            "--dataset",
            "./infer.csv",
            "--output",
            "./actions.jsonl",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"action\":").not());

    let written = fs::read_to_string(dir.path().join("actions.jsonl")).unwrap();
    assert_eq!(written.lines().count(), 3, "{}", written);
}

#[test]
fn silent_still_writes_an_action_file_that_was_asked_for() {
    // --silent means "print nothing", not "do nothing": a file the user named
    // is the result of the command, not chatter.
    let dir = trained_workspace();

    let output = rlt(&dir)
        .args([
            "--silent",
            "infer",
            "--dataset",
            "./infer.csv",
            "--output",
            "./actions.jsonl",
        ])
        .output()
        .unwrap();

    assert_only_the_upstream_line_leaked(&output);
    assert_eq!(
        fs::read_to_string(dir.path().join("actions.jsonl"))
            .unwrap()
            .lines()
            .count(),
        3
    );
}

#[test]
fn silent_infer_without_a_destination_prints_no_actions() {
    let dir = trained_workspace();

    let output = rlt(&dir)
        .args(["--silent", "infer", "--dataset", "./infer.csv"])
        .output()
        .unwrap();

    assert_only_the_upstream_line_leaked(&output);
}

/// Assert that a silent run emitted nothing of its own.
///
/// The library prints `EMPTY INPUT` straight to stdout when the data source
/// ends, bypassing the log filter the CLI configures — `docs/found-issues.md`
/// issue 5. Every inference run ends that way, so `--silent infer` cannot be
/// byte-for-byte silent until that is fixed upstream. Asserting on the exact
/// remainder keeps the test honest: anything the CLI itself prints still fails.
fn assert_only_the_upstream_line_leaked(output: &std::process::Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        "EMPTY INPUT",
        "--silent leaked more than the known upstream line"
    );
    assert!(
        output.stderr.is_empty(),
        "--silent leaked {} bytes of stderr",
        output.stderr.len()
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
        .stderr(predicate::str::contains("No config file found"));
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
fn a_typod_config_key_warns_instead_of_being_ignored() {
    // Regression: an unknown key configured nothing and said nothing, so a
    // misspelling looked exactly like a setting that had been applied.
    let dir = workspace(10);
    let config = format!("{}bacth_size = 64\n", CONFIG);
    fs::write(dir.path().join("typo.toml"), config).unwrap();

    rlt(&dir)
        .args([
            "--config",
            "./typo.toml",
            "train",
            "--dataset",
            "./data.csv",
            "--dry-run",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("unknown key 'bacth_size'"));
}

#[test]
fn config_file_values_reach_the_run() {
    // Regression: clap defaults used to mask every value in the config file.
    let dir = workspace(10);

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv", "--dry-run"])
        .assert()
        .success()
        .stderr(predicate::str::contains("batch size: 8 (Config.toml)"));
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
        .stderr(predicate::str::contains("batch size: 99 (flag)"));
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
fn log_level_silent_silences_the_command_too() {
    // Regression: the two spellings disagreed. --silent suppressed the banner
    // and settings block; --log-level silent, documented as "no log output at
    // all", printed both.
    let dir = workspace(30);

    let output = rlt(&dir)
        .args(["--log-level", "silent", "train", "--dataset", "./data.csv"])
        .output()
        .unwrap();

    assert!(
        output.stdout.is_empty() && output.stderr.is_empty(),
        "--log-level silent leaked {} bytes of stdout and {} of stderr",
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

/// A reward script that pays for action 1 and nothing else, so a static:1
/// baseline is unbeatable and any other baseline is beatable. Deterministic, so
/// the comparison in these tests is exact.
const FIXED_REWARD: &str = "import json, sys\n\
     for line in sys.stdin:\n\
     \x20   request = json.loads(line)\n\
     \x20   good = request['action'] == 1\n\
     \x20   print(json.dumps({\"reward\": 1.0 if good else 0.0, \"success\": good}), flush=True)\n";

/// A workspace with a trained checkpoint, a held-out set, and a reward script.
fn eval_workspace() -> TempDir {
    let dir = trained_workspace();
    fs::write(dir.path().join("reward.py"), FIXED_REWARD).unwrap();
    dir
}

#[test]
fn eval_scores_a_checkpoint_against_a_baseline() {
    let dir = eval_workspace();

    let stdout = rlt(&dir)
        .args([
            "eval",
            "--dataset",
            "./infer.csv",
            "--reward-script",
            "./reward.py",
            "--baseline",
            "static:1",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout).unwrap();

    assert!(stdout.contains("Evaluated 3 sample(s)"), "{}", stdout);
    assert!(stdout.contains("Policy:"), "{}", stdout);
    assert!(stdout.contains("Baseline: static:1"), "{}", stdout);
    assert!(stdout.contains("Difference:"), "{}", stdout);
    // static:1 always earns the reward, so it cannot be beaten here.
    assert!(
        stdout.contains("The baseline beats the policy.") || stdout.contains("are level."),
        "{}",
        stdout
    );
}

#[test]
fn eval_reports_the_policy_alone_without_a_baseline() {
    let dir = eval_workspace();

    rlt(&dir)
        .args([
            "eval",
            "--dataset",
            "./infer.csv",
            "--reward-script",
            "./reward.py",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success()
        .stdout(predicate::str::contains("mean reward"))
        .stdout(predicate::str::contains("Difference:").not());
}

#[test]
fn eval_emits_a_json_summary() {
    let dir = eval_workspace();

    rlt(&dir)
        .args([
            "eval",
            "--dataset",
            "./infer.csv",
            "--reward-script",
            "./reward.py",
            "--baseline",
            "round-robin",
            "--format",
            "json",
            "--output",
            "./report.json",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success();

    // Parsed, not merely searched: --format json exists so a CI step can read
    // the result back. Written to a file rather than redirected from stdout
    // because the library prints EMPTY INPUT straight to stdout when a data
    // source ends (docs/found-issues.md issue 5), which would leave the
    // redirect holding something that is not JSON through no fault of ours.
    let written = fs::read_to_string(dir.path().join("report.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&written)
        .unwrap_or_else(|err| panic!("report is not JSON ({}):\n{}", err, written));

    assert!(parsed.get("reward_gain").is_some(), "{}", written);
    assert!(parsed["policy"]["mean_reward"].is_number(), "{}", written);
    assert!(parsed["baseline"]["label"].is_string(), "{}", written);
}

#[test]
fn eval_without_a_reward_script_is_refused() {
    // Every reward would be 0.0, so the report would be zeros presented as an
    // answer.
    let dir = eval_workspace();

    rlt(&dir)
        .args(["eval", "--dataset", "./infer.csv"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("needs a reward script"));
}

#[test]
fn eval_rejects_a_baseline_outside_the_action_space() {
    let dir = eval_workspace();

    rlt(&dir)
        .args([
            "eval",
            "--dataset",
            "./infer.csv",
            "--reward-script",
            "./reward.py",
            "--baseline",
            "static:99",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("outside the model's action space"));
}

#[test]
fn eval_accepts_a_heuristic_script_as_the_baseline() {
    let dir = eval_workspace();
    fs::write(
        dir.path().join("heuristic.py"),
        "import json, sys\n\
         for line in sys.stdin:\n\
         \x20   request = json.loads(line)\n\
         \x20   # The rule a team might already run: high first feature -> act.\n\
         \x20   print(json.dumps({\"action\": 1 if request['features'][0] > 0.15 else 0}), flush=True)\n",
    )
    .unwrap();

    rlt(&dir)
        .args([
            "eval",
            "--dataset",
            "./infer.csv",
            "--reward-script",
            "./reward.py",
            "--baseline",
            "./heuristic.py",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success()
        .stdout(predicate::str::contains("Baseline: ./heuristic.py"));
}

#[test]
fn eval_without_a_checkpoint_fails_instead_of_scoring_random_weights() {
    let dir = eval_workspace();

    rlt(&dir)
        .args([
            "eval",
            "--dataset",
            "./infer.csv",
            "--reward-script",
            "./reward.py",
            "--model-name",
            "ghost.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("checkpoint not found"));
}

#[test]
fn inspect_reports_the_architecture_and_parameter_count() {
    let dir = trained_workspace();

    rlt(&dir)
        .arg("inspect")
        .assert()
        .success()
        // Six inputs and three outputs, from the shared test config.
        .stdout(predicate::str::contains("6 -> 8 -> 3"))
        .stdout(predicate::str::contains("parameters"))
        .stdout(predicate::str::contains("Tensors"));
}

#[test]
fn inspect_emits_json_on_request() {
    let dir = trained_workspace();

    let stdout = rlt(&dir)
        .args(["inspect", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(stdout).unwrap();

    assert!(stdout.contains("\"parameters\""), "{}", stdout);
    assert!(stdout.contains("\"architecture\""), "{}", stdout);
    assert!(stdout.contains("\"tensors\""), "{}", stdout);
}

#[test]
fn inspect_rejects_something_that_is_not_a_checkpoint() {
    let dir = workspace(10);
    fs::write(dir.path().join("notes.txt"), "just some text\n").unwrap();

    rlt(&dir)
        .args(["inspect", "--checkpoint", "./notes.txt"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not valid JSON"));
}

#[test]
fn inspect_without_a_checkpoint_says_so() {
    let dir = workspace(10);

    rlt(&dir)
        .args(["inspect", "--model-name", "ghost.json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("checkpoint not found"));
}

#[test]
fn export_accepts_a_model_name() {
    // train and infer both take --model-name; export used to accept only
    // --checkpoint or whatever the config file happened to say.
    let dir = workspace(400);

    rlt(&dir)
        .args(["train", "--dataset", "./data.csv"])
        .assert()
        .success();

    rlt(&dir)
        .args([
            "export",
            "--model-name",
            "test-model.json",
            "--output",
            "./exported/by-name.json",
        ])
        .assert()
        .success();

    assert_eq!(
        fs::read(dir.path().join("exported/by-name.json")).unwrap(),
        fs::read(dir.path().join(CHECKPOINT)).unwrap(),
    );
}

#[test]
fn export_rejects_a_checkpoint_and_a_model_name_together() {
    // Silently preferring one would be the same class of bug as the settings
    // that used to be accepted and ignored.
    let dir = workspace(10);

    rlt(&dir)
        .args([
            "export",
            "--checkpoint",
            "./a.json",
            "--model-name",
            "b.json",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
}

#[test]
fn export_refuses_a_directory_at_the_checkpoint_path() {
    // Regression: a directory has a non-zero length, so it passed the
    // emptiness guard and failed later with a raw copy error.
    let dir = workspace(10);
    fs::create_dir_all(dir.path().join(CHECKPOINT)).unwrap();

    rlt(&dir)
        .args(["export", "--output", "./exported/model.json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not a file"));
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

/// A provider that answers correctly but chatters on stderr every sample.
const NOISY_PROVIDER: &str = "import json, sys\n\
     for line in sys.stdin:\n\
     \x20   print('noise on stderr', file=sys.stderr, flush=True)\n\
     \x20   print(json.dumps([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), flush=True)\n";

#[test]
fn python_stderr_reaches_the_log() {
    let dir = workspace(10);
    fs::write(dir.path().join("noisy.py"), NOISY_PROVIDER).unwrap();

    rlt(&dir)
        .args(["train", "--dataset", "./noisy.py"])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success()
        .stderr(predicate::str::contains("noise on stderr"));
}

#[test]
fn silent_hides_python_stderr() {
    // Regression: the script's stderr was inherited, so a traceback or a stray
    // print(file=sys.stderr) bypassed every log filter the CLI configures —
    // including --silent, whose contract is zero bytes on both streams.
    let dir = workspace(10);
    fs::write(dir.path().join("noisy.py"), NOISY_PROVIDER).unwrap();

    let output = rlt(&dir)
        .args(["--silent", "train", "--dataset", "./noisy.py"])
        .timeout(std::time::Duration::from_secs(60))
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
fn a_hanging_python_script_times_out_instead_of_hanging_the_cli() {
    // Regression: the CLI blocked in read_line forever, with no output and no
    // way to tell it apart from a slow run. This test's own timeout is the
    // real assertion.
    let dir = workspace(10);
    fs::write(
        dir.path().join("hang.py"),
        "import sys, time\n\
         for line in sys.stdin:\n\
         \x20   time.sleep(600)\n",
    )
    .unwrap();

    rlt(&dir)
        .args(["train", "--dataset", "./hang.py", "--script-timeout", "1"])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .failure()
        .stderr(predicate::str::contains("stopped answering"));
}

#[test]
fn a_script_named_only_by_the_config_needs_consent() {
    // The threat this guards: `cd` into a directory someone else prepared, run
    // `rlt train`, and execute whatever their Config.toml nominates. The path
    // never appears on your command line.
    let dir = workspace(400);
    fs::write(dir.path().join("reward.py"), FIXED_REWARD).unwrap();

    let config = format!("{}reward_script = \"./reward.py\"\n", CONFIG);
    fs::write(dir.path().join("Config.toml"), config).unwrap();

    // No terminal here, so there is nobody to ask and refusing is the answer.
    rlt(&dir)
        .args(["train", "--dataset", "./data.csv"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--allow-scripts"))
        .stderr(predicate::str::contains("./reward.py"));

    assert!(
        !dir.path().join(CHECKPOINT).exists(),
        "a refused run must not have trained"
    );

    rlt(&dir)
        .args(["--allow-scripts", "train", "--dataset", "./data.csv"])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success();
}

#[test]
fn a_script_named_by_a_flag_needs_no_consent() {
    // Typing the path is the informed choice the gate exists to require.
    let dir = workspace(400);
    fs::write(dir.path().join("reward.py"), FIXED_REWARD).unwrap();

    rlt(&dir)
        .args([
            "train",
            "--dataset",
            "./data.csv",
            "--reward-script",
            "./reward.py",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success();
}

#[test]
fn a_csv_named_by_the_config_needs_no_consent() {
    // Only scripts are executed; a dataset is read. Gating CSVs would be
    // friction with nothing behind it.
    let dir = workspace(400);
    let config = format!("{}dataset = \"./data.csv\"\n", CONFIG);
    fs::write(dir.path().join("Config.toml"), config).unwrap();

    rlt(&dir)
        .arg("train")
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success();
}

#[test]
fn a_working_reward_script_trains_to_completion() {
    let dir = workspace(400);
    fs::write(
        dir.path().join("reward.py"),
        "import json, sys\n\
         for line in sys.stdin:\n\
         \x20   print(json.dumps({\"reward\": 1.0, \"success\": True}), flush=True)\n",
    )
    .unwrap();

    rlt(&dir)
        .args([
            "train",
            "--dataset",
            "./data.csv",
            "--reward-script",
            "./reward.py",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success();

    assert!(dir.path().join(CHECKPOINT).is_file());
}

#[test]
fn a_dead_reward_script_fails_the_run() {
    // Regression: a reward script that died logged one error per step and let
    // training run to the end, exiting zero with a checkpoint trained entirely
    // on rewards the script never produced.
    let dir = workspace(400);
    fs::write(
        dir.path().join("dead_reward.py"),
        "import sys\nsys.exit('reward script crashed')\n",
    )
    .unwrap();

    rlt(&dir)
        .args([
            "train",
            "--dataset",
            "./data.csv",
            "--reward-script",
            "./dead_reward.py",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "reward script failed 10 times in a row",
        ));
}

#[test]
fn a_reward_script_that_answers_garbage_fails_the_run() {
    // The worker stays alive here, so this exercises the failure counter rather
    // than a closed pipe.
    let dir = workspace(400);
    fs::write(
        dir.path().join("garbage_reward.py"),
        "import sys\n\
         for line in sys.stdin:\n\
         \x20   print('not json', flush=True)\n",
    )
    .unwrap();

    rlt(&dir)
        .args([
            "train",
            "--dataset",
            "./data.csv",
            "--reward-script",
            "./garbage_reward.py",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "reward script failed 10 times in a row",
        ));
}

#[test]
fn transient_reward_failures_do_not_abort_the_run() {
    // Nine failures and then answers: the streak resets, so a flaky script does
    // not add up to an abort over a long run.
    let dir = workspace(400);
    fs::write(
        dir.path().join("flaky_reward.py"),
        "import json, sys\n\
         failures = 0\n\
         for line in sys.stdin:\n\
         \x20   if failures < 9:\n\
         \x20       failures += 1\n\
         \x20       print('not json', flush=True)\n\
         \x20   else:\n\
         \x20       print(json.dumps({\"reward\": 1.0, \"success\": True}), flush=True)\n",
    )
    .unwrap();

    rlt(&dir)
        .args([
            "train",
            "--dataset",
            "./data.csv",
            "--reward-script",
            "./flaky_reward.py",
        ])
        .timeout(std::time::Duration::from_secs(60))
        .assert()
        .success();
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
