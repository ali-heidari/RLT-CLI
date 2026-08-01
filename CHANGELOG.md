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

## [Unreleased]

Everything so far. There has been no tagged release yet.

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

- **A trained policy currently decides no better than chance.** This is a
  limitation of the pinned `aixker-rlt` revision, not of this CLI: training
  drives the policy toward a constant action rather than an input-dependent
  one. Selecting the best action instead of sampling was measured and scores
  worse, so it is the learning rather than the action selection. `rlt eval`
  reports the result honestly, which is what it is for. See
  [docs/found-issues.md](docs/found-issues.md) issue 8.
- Checkpoint filenames are doubled (`m.json` becomes `models/m.json.m.json`),
  also upstream. The CLI reports the real path so the file can be found.
