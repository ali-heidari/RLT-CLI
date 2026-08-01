//! Scaffolding for `rlt init`.
//!
//! Zero-to-first-run should be two commands, not a documentation page. This
//! writes a config, a reward script, a heuristic to compare against, and two
//! generated datasets — everything `train` and `eval` need, with no flags.
//!
//! The generated data carries a **learnable rule**: the first feature is a
//! load figure, and the best action is which third of the range it falls in.
//! Scaffolding data with no signal in it would make the first `eval` report a
//! policy that loses to a coin flip, which is a poor introduction to a tool
//! whose whole job is deciding things.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Rows in the generated training set.
///
/// Enough to feed `batch_size` × `total_batches` below without the source
/// running dry, which would end the run early and save no model.
const TRAIN_ROWS: usize = 2400;

/// Rows in the generated held-out set, used by `eval`.
const HOLDOUT_ROWS: usize = 400;

/// Features per row. Matches `input_number` in the scaffolded config.
const FEATURES: usize = 6;

/// Actions the model chooses between. Matches `output_number`.
const ACTIONS: usize = 3;

const BATCH_SIZE: usize = 32;
const TOTAL_BATCHES: usize = 60;

/// Training reads `batch_size * (total_batches + 1)` samples before the loop
/// ends. A shorter dataset would run dry first, ending the run before a batch
/// completed and saving no model — so this is checked at compile time rather
/// than discovered by whoever ran `init` and then `train`.
const _: () = assert!(TRAIN_ROWS > BATCH_SIZE * (TOTAL_BATCHES + 1));

/// The scaffolded config.
///
/// Built from the same constants as the data generator, so the two cannot
/// disagree about how many values a row holds.
fn config() -> String {
    format!(
        r#"# Written by `rlt init`. Every key is optional; these match the generated
# sample data, so `rlt train` runs with no flags at all.

# --- Data ---------------------------------------------------------------

dataset = "./data/train.csv"
reward_script = "./scripts/reward.py"

# Values per row in the dataset.
input_number = {FEATURES}

# Actions the model chooses between.
output_number = {ACTIONS}

# --- Model --------------------------------------------------------------

model_name = "quickstart.json"
hidden_layers = 16
batch_size = {BATCH_SIZE}

# Batches to train for. `--epochs N` sets this to N * 100 instead.
total_batches = {TOTAL_BATCHES}

reply_capacity = 4096
log_interval = 1000
"#
    )
}

const REWARD_SCRIPT: &str = r#"#!/usr/bin/env python3
"""Reward for the generated sample data.

Started once and asked for a reward per step over stdin/stdout. Each request is
one JSON line holding the features and the action the model chose; each reply is
one JSON line holding the reward and whether the step counted as a success.

The rule: the first feature is a load figure in [0, 1). The right action is
which third of that range it falls in. Replace this with your own objective —
this file exists to be edited.
"""
import json
import sys


def best_action(load):
    if load < 1 / 3:
        return 0
    if load < 2 / 3:
        return 1
    return 2


for line in sys.stdin:
    request = json.loads(line)
    correct = request["action"] == best_action(request["features"][0])
    print(
        json.dumps({"reward": 1.0 if correct else 0.0, "success": correct}),
        flush=True,
    )
"#;

const HEURISTIC_SCRIPT: &str = r#"#!/usr/bin/env python3
"""A plain threshold rule, to compare the trained policy against.

    rlt eval --dataset ./data/holdout.csv --baseline ./scripts/heuristic.py

Each request is one JSON line holding the features; each reply names an action.
This is the shape of "the rule we already run today", and beating it is the
only result that justifies replacing it.

Deliberately imperfect: it splits at the halfway point and so never chooses
action 1, which leaves the middle of the range wrong. A policy that cannot beat
this has not learned anything.
"""
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    load = request["features"][0]
    print(json.dumps({"action": 0 if load < 0.5 else 2}), flush=True)
"#;

/// One scaffolded file.
pub struct Scaffolded {
    pub path: PathBuf,
    contents: String,
}

/// Every file `init` writes, relative to the target directory.
fn scaffold(dir: &Path) -> Vec<Scaffolded> {
    vec![
        Scaffolded {
            path: dir.join("Config.toml"),
            contents: config(),
        },
        Scaffolded {
            path: dir.join("scripts/reward.py"),
            contents: REWARD_SCRIPT.to_string(),
        },
        Scaffolded {
            path: dir.join("scripts/heuristic.py"),
            contents: HEURISTIC_SCRIPT.to_string(),
        },
        Scaffolded {
            path: dir.join("data/train.csv"),
            contents: dataset(TRAIN_ROWS, 0),
        },
        Scaffolded {
            path: dir.join("data/holdout.csv"),
            contents: dataset(HOLDOUT_ROWS, 7),
        },
    ]
}

/// Generate sample rows.
///
/// Deterministic, so two runs of `init` produce identical files and a
/// comparison between them means something. `offset` shifts the sequence so the
/// held-out set is not the training set over again.
fn dataset(rows: usize, offset: usize) -> String {
    let mut text = String::new();

    for row in 0..rows {
        // A coprime stride visits every value in the range before repeating,
        // so the actions stay balanced without needing a random generator.
        let load = ((row * 37 + offset * 13) % 100) as f64 / 100.0;

        let features = [
            load,
            1.0 - load,
            load * load,
            load * 0.5 + 0.25,
            // Genuine noise: uncorrelated with the answer, so the model has
            // something to learn to ignore.
            ((row * 7 + offset) % 10) as f64 / 10.0,
            (load - 0.5).abs(),
        ];
        debug_assert_eq!(features.len(), FEATURES);

        let row: Vec<String> = features
            .iter()
            .map(|value| format!("{:.4}", value))
            .collect();
        text.push_str(&row.join(","));
        text.push('\n');
    }

    text
}

/// Write the scaffold into `dir`.
///
/// Refuses to touch a file that already exists unless `force`: `init` is run in
/// directories that already hold work, and silently replacing a reward script
/// someone wrote would be unforgivable.
pub fn write(dir: &Path, force: bool) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let files = scaffold(dir);

    if !force {
        let existing: Vec<String> = files
            .iter()
            .filter(|file| file.path.exists())
            .map(|file| file.path.display().to_string())
            .collect();

        if !existing.is_empty() {
            return Err(format!(
                "these files already exist: {}. Pass --force to overwrite them.",
                existing.join(", ")
            )
            .into());
        }
    }

    let mut written = Vec::new();
    for file in files {
        if let Some(parent) = file.path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!("failed to create directory '{}': {}", parent.display(), err)
            })?;
        }
        fs::write(&file.path, &file.contents)
            .map_err(|err| format!("failed to write '{}': {}", file.path.display(), err))?;
        written.push(file.path);
    }

    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule the scaffolded reward script scores against.
    fn best_action(load: f64) -> usize {
        if load < 1.0 / 3.0 {
            0
        } else if load < 2.0 / 3.0 {
            1
        } else {
            2
        }
    }

    fn rows(text: &str) -> Vec<Vec<f64>> {
        text.lines()
            .map(|line| {
                line.split(',')
                    .map(|field| field.parse::<f64>().unwrap())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn every_generated_row_has_the_configured_width() {
        // A mismatch here fails the run at the first row, which would make the
        // scaffold useless the moment it was generated.
        let parsed = rows(&dataset(TRAIN_ROWS, 0));

        assert_eq!(parsed.len(), TRAIN_ROWS);
        assert!(parsed.iter().all(|row| row.len() == FEATURES));
        assert!(config().contains(&format!("input_number = {}", FEATURES)));
        assert!(config().contains(&format!("output_number = {}", ACTIONS)));
    }

    #[test]
    fn the_generated_data_is_deterministic() {
        assert_eq!(dataset(50, 0), dataset(50, 0));
    }

    #[test]
    fn the_held_out_set_is_not_the_training_set_again() {
        let train = rows(&dataset(HOLDOUT_ROWS, 0));
        let holdout = rows(&dataset(HOLDOUT_ROWS, 7));

        assert_ne!(train, holdout);
    }

    #[test]
    fn every_action_is_reachable_and_roughly_balanced() {
        // A scaffold where one action is always right would teach the model to
        // pick a constant, and make the static baseline unbeatable.
        let parsed = rows(&dataset(TRAIN_ROWS, 0));
        let mut counts = [0usize; ACTIONS];
        for row in &parsed {
            counts[best_action(row[0])] += 1;
        }

        let smallest = counts.iter().min().unwrap();
        assert!(
            *smallest > parsed.len() / 6,
            "actions are lopsided: {:?}",
            counts
        );
    }

    #[test]
    fn the_heuristic_is_beatable_but_not_useless() {
        // It should be a real bar to clear: right most of the time, wrong in
        // the middle of the range where it never chooses action 1.
        let parsed = rows(&dataset(TRAIN_ROWS, 0));
        let correct = parsed
            .iter()
            .filter(|row| {
                let heuristic = if row[0] < 0.5 { 0 } else { 2 };
                heuristic == best_action(row[0])
            })
            .count();

        let share = correct as f64 / parsed.len() as f64;
        assert!(
            share > 0.5,
            "the heuristic is too weak to be a bar: {}",
            share
        );
        assert!(
            share < 0.9,
            "the heuristic leaves nothing to beat: {}",
            share
        );
    }

    #[test]
    fn an_existing_file_is_not_overwritten_without_force() {
        let dir = tempfile::TempDir::new().unwrap();
        fs::write(dir.path().join("Config.toml"), "mine\n").unwrap();

        let error = write(dir.path(), false).unwrap_err().to_string();
        assert!(error.contains("already exist"), "{}", error);
        assert!(error.contains("--force"), "{}", error);
        assert_eq!(
            fs::read_to_string(dir.path().join("Config.toml")).unwrap(),
            "mine\n",
            "the existing file must be untouched"
        );

        write(dir.path(), true).unwrap();
        assert!(fs::read_to_string(dir.path().join("Config.toml"))
            .unwrap()
            .contains("dataset"));
    }

    #[test]
    fn the_scaffold_writes_every_file_it_promises() {
        let dir = tempfile::TempDir::new().unwrap();

        let written = write(dir.path(), false).unwrap();

        assert_eq!(written.len(), 5);
        assert!(written.iter().all(|path| path.is_file()));
        assert!(dir.path().join("data/train.csv").is_file());
        assert!(dir.path().join("scripts/reward.py").is_file());
    }
}
