# Contributing

Thanks for looking. This is a small project and the bar for a useful
contribution is low — a corrected error message counts.

## Setting up

Requires Rust **1.92** or newer. That floor comes from `wgpu`, pulled in by the
GPU backend, not from this crate.

```bash
git clone https://github.com/ali-heidari/RLT-CLI
cd RLT-CLI
cargo build
cargo test
```

Python 3 must be on `PATH` for the tests that exercise the data provider and the
reward factory. Everything else runs without it.

To try the tool end to end, scaffold a throwaway project:

```bash
mkdir /tmp/rlt-scratch && cd /tmp/rlt-scratch
/path/to/RLT-CLI/target/debug/rlt init
/path/to/RLT-CLI/target/debug/rlt train
```

## Before you open a pull request

These are exactly what CI runs, so running them locally saves a round trip:

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
```

Warnings are errors in CI. If clippy objects to something you believe is right,
say so in the pull request rather than reaching for `#[allow]` — the lint is
usually correct, and when it is not that is worth a comment explaining why.

## What good looks like here

**Explain why, not what.** The code says what it does. Comments earn their place
by recording the reason — the constraint, the failure that motivated it, the
thing that looks wrong but is not. Compare:

```rust
// Increment the counter.                          // adds nothing
let failures = self.consecutive_errors.fetch_add(1, Ordering::Relaxed) + 1;

// Returning (0.0, false) forever is how a run used to "succeed" having
// trained entirely on rewards the script never produced.
let failures = self.consecutive_errors.fetch_add(1, Ordering::Relaxed) + 1;
```

**Every behaviour change gets a test.** Bug fixes get a test that fails before
the fix; the test comment should name the regression it guards, so whoever reads
it in a year knows why it is there.

**Test names are sentences.** `a_dead_reward_script_fails_the_run`, not
`test_reward_2`.

**Fail loudly, never silently.** The recurring theme of this codebase's bug
history is commands that appeared to succeed: actions computed and discarded, a
dead reward script that trained on fabricated zeros, a policy loaded from a
checkpoint that was never written. When something goes wrong, say so and exit
non-zero.

**Update the docs in the same change.** `docs/` is part of the repository, and
`docs/index.md` must link every file in it. A behaviour change that leaves the
documentation describing the old behaviour is not finished.

## Commits

`type(scope): short summary`, a blank line, the body, per
[`.agent/base/commit-conventions.md`](.agent/base/commit-conventions.md). Types:
`feat`, `fix`, `chore`, `docs`, `ci`, `style`, `refactor`, `test`.

Use the body to explain the problem the change solves. "Fix bug" tells the next
reader nothing; "a reward script that died let training run to completion on
fabricated zero rewards" tells them everything.

Add `Refs: #<issue>` when there is an issue.

## Where to start

- [`docs/roadmap.md`](docs/roadmap.md) is the prioritised backlog.
- [`docs/found-issues.md`](docs/found-issues.md) records problems that live in
  the upstream `aixker-rlt` library rather than here. **Read issue 8 before
  spending time on model quality** — a trained policy currently decides at
  chance, and the cause is upstream.
- Anything labelled `good-first-issue`.

## Project layout

| Path | What lives there |
| --- | --- |
| `src/main.rs` | Command handlers and configuration resolution |
| `src/cli/` | Argument definitions |
| `src/providers/` | CSV reader, Python provider, the long-lived Python worker |
| `src/eval.rs`, `src/inspect.rs`, `src/init.rs` | The `eval`, `inspect` and `init` commands |
| `src/reward_factory.rs` | Reward-script integration |
| `src/stop_signal.rs` | How a run is ended early, by failure or by Ctrl-C |
| `tests/cli.rs` | End-to-end tests driving the built binary |
| `docs/` | Everything user-facing |

Unit tests live beside the code they cover, in a `#[cfg(test)] mod tests` at the
foot of the file.
