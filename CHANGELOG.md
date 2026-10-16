# Changelog

Notable changes to `aixker-rlt-cli`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/).

## Versioning policy

Pre-1.0, so the usual caveat applies: **breaking changes may land in a minor
release**. What counts as breaking here is the CLI surface — command names,
flag names and meanings, the JSON shapes `infer` and `eval` emit, the config
file keys, and the Python script protocols. The Rust API is not a supported
interface; this crate is a binary.

## [0.2.0] - 2026-10-16

Moves to **aixker-rlt 0.2.0**, which fixes the learning bugs that collapsed
every trained policy onto one fixed action. Not tagged yet: this version also
ships everything under [Unreleased] below, none of which has been released.

**Retrain every model.** Checkpoints from aixker-rlt 0.1.0 still load, because
the file format did not change, but their weights are the collapsed ones.

### Changed

- `aixker-rlt` comes from its `develop` branch at 0.2.0. `Cargo.lock` pins the
  exact commit, so builds do not move with the branch.
- **Training learns differently.** The library now trains on normalized
  advantages with averaged, clipped gradients, a constant learning rate, an
  entropy bonus and a fixed exploration schedule. Results from earlier versions
  do not carry over.
- **Reward shaping is off.** The library used to rewrite every reward with a
  penalty of up to about 0.5, which swamped small rewards. Set
  `reward_shaping = true` or pass `--reward-shaping` to restore it.
- **`infer` and `eval` take no random actions.** They sample at
  `infer_temperature` (default `0.5`; `0` always takes the best action). Eval
  scores change as a result, and they repeat exactly with `--seed`.
- **`log_interval` now does something.** It sets how many batches pass between
  lines of training statistics at `info` level: reward mean and spread, loss,
  gradient norm, epsilon, temperature, the policy's action mix and its entropy.
  `0` turns them off. It used to be ignored.
- `rlt init` writes `log_interval = 10` instead of `1000`, so its 60-batch run
  shows statistics.
- The settings block reports every learning setting and where it came from.
- Nothing prints `EMPTY INPUT` to stdout any more, so `rlt infer | jq` and
  `rlt eval --format json > report.json` work without filtering. The library
  now logs that line instead.

### Added

- `--seed` on `train`, `infer` and `eval`, and a `seed` config key. The same
  seed, data and settings train the same weights.
- `--reward-shaping [BOOL]` and `--learning-rate RATE` on `train`.
  `--learning-rate` was removed earlier because the library ignored it; it now
  works.
- `--infer-temperature T` on `infer` and `eval`.
- Config keys `reward_shaping`, `learning_rate`, `max_grad_norm`,
  `entropy_coef`, `epsilon_start`, `epsilon_end`, `temperature_start`,
  `temperature_end`, `exploration_batches` and `infer_temperature`. Every one is
  optional and defaults to the library's value.
- Out-of-range learning settings fail before the run starts, for example a
  learning rate of `0` or an epsilon above `1`.
- `inspect` warns when a checkpoint was trained by aixker-rlt 0.1.0, and
  `infer`, `eval` and `export` log the same warning. They still run.

### Fixed

- A training run that completed no batch exited `0`. Under 0.2.0 the library
  always saves when it stops, so the run left a checkpoint of untrained weights
  that later passed for a model. `train` now exits non-zero, and removes the
  checkpoint if the run created it.

## [Unreleased]

Everything before 0.2.0. There has been no tagged release yet.

### Added

- `rlt init` — scaffolds a config, a reward script, a heuristic baseline and
  generated train/holdout datasets, so a first run needs nothing from you.
- `rlt eval` — scores a checkpoint against held-out data and against a baseline
  policy (`static:N`, `round-robin`, `random`, or a `.py` heuristic), reporting
  mean reward, success rate, a per-action breakdown and the difference. Both
  policies are scored on the same rows in one pass. `--format json` emits
  `reward_gain` for CI to gate on.
- `rlt inspect` — reports a checkpoint's architecture, parameter count,
  per-tensor statistics and embedded training snapshot, and **warns when the
  weights have diverged or gone non-finite**. Such a checkpoint loads and
  answers without complaint; it just answers badly.
- `infer` now emits one JSON line per decision to stdout or `--output PATH`,
  with `--with-features` to include the input that produced each one.
- `--script-timeout` (and `script_timeout_secs`): a deadline on Python script
  responses, default 30s, `0` waits forever.
- `--interval-secs` on `infer`.
- Graceful Ctrl-C: a run winds down as though the data had ended, keeping the
  last checkpoint and exiting `0`. A second Ctrl-C exits `130`.
- Continuous integration: fmt, clippy (`-D warnings`), tests and a release build
  across Linux, macOS and Windows.

### Changed

- **Renamed**: the crate is `aixker-rlt-cli` and the binary is `rlt`. It was
  `RLT-CLI` for both.
- Inference over a **file** no longer sleeps between samples. The default now
  depends on the dataset — `0` for a file, `10` for a live `.py` provider — so a
  400-row CSV takes seconds rather than 67 minutes.
- Python script stderr is forwarded to the log instead of inherited, so it obeys
  `--log-level` (including `--silent`).
- `--log-level silent` now silences command output, matching `--silent`.
- Unrecognised config keys are reported as warnings instead of ignored.
- Checkpoint paths are printed absolutely, because `./models` resolves against
  the working directory.
- The reward protocol's `action` field is `u32`; it was `u8`, which truncated
  above 255 actions.
- `export` accepts `--model-name`, like `train` and `infer`.
- `aixker-rlt` is pinned to `rev = "b0603e0"` and `Cargo.lock` is committed, so
  builds are reproducible.

### Fixed

- `infer` computed an action for every sample and discarded all of them.
- A dead or garbage-answering reward script let training run to completion on
  fabricated zero rewards, exit `0`, and write a checkpoint that looked
  legitimate. Ten consecutive failures now end the run non-zero.
- A Python script that read a request and never answered hung the CLI forever.
- A directory at the checkpoint path passed the "is it empty" guard.
- `.py` detection was case-sensitive, so `provider.PY` was read as CSV.
- On Windows, the Microsoft Store `python3` alias stub defeated the interpreter
  fallback. The `py` launcher is now tried first, and an interpreter that exits
  before answering is abandoned for the next candidate. **Untested** — no CI had
  ever run on Windows before this release.
- A repeated reward-script failure is logged once rather than once per step.

### Known issues

- **A trained policy decided no better than chance** on the `aixker-rlt`
  revision pinned here. Fixed by moving to aixker-rlt 0.2.0; see [0.2.0].
- Checkpoint filenames are doubled (`m.json` becomes `models/m.json.m.json`),
  also upstream. The CLI reports the real path so the file can be found.
