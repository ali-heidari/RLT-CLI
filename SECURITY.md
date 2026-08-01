# Security Policy

## Supported versions

The project is pre-1.0 and there are no releases yet. Only the tip of `develop`
is supported; fixes land there.

## Reporting a vulnerability

Report privately through
[GitHub's security advisories](https://github.com/ali-heidari/RLT-CLI/security/advisories/new),
or by email to the address on the maintainer's GitHub profile. Please do not
open a public issue for a vulnerability.

Include what you did, what happened, and what you expected. A reproduction — a
config file, a dataset shape, a script — is worth more than a description.

Expect an acknowledgement within a week. This is a small project maintained in
spare time, so please be patient with the fix itself.

## What is in scope

The interesting surface is small. Worth reporting:

- A crafted **dataset** that causes something worse than a parse error: a panic,
  unbounded memory growth, a hang.
- A crafted **checkpoint** that causes a panic or unbounded allocation in
  `rlt inspect` or when loading a model.
- A crafted **Python script response** that escapes the JSON-line protocol and
  affects the CLI beyond that one step.
- Anything that lets a **config file** cause execution the user did not ask for,
  beyond the documented behaviour below.

## What is by design

**`rlt` executes the Python scripts it is pointed at.** `--dataset provider.py`,
`--reward-script reward.py` and `--baseline heuristic.py` all start an
interpreter and run the file. That is the feature — but where the path came
from decides whether you are asked first.

**A path you typed is an informed choice** and runs without ceremony.

**A path named only by a config file is not.** Someone who clones a repository
and runs `rlt train` in it has asked to train a model, not to execute whatever
that directory nominates. So when a script path comes from `Config.toml` rather
than the command line:

- With a terminal, `rlt` lists the scripts and asks before running them:

  ```text
  Config.toml asks to execute:
    ./scripts/reward.py
  These run as you, with your privileges.
  Run them? [y/N]
  ```

- Without one — CI, a pipeline, a cron job — it refuses and exits non-zero,
  naming the scripts. Pass `--allow-scripts` to answer in advance.

Only `.py` paths are gated. A CSV dataset is read, not executed.

This narrows the exposure rather than removing it: `--allow-scripts` is a real
decision, and a script you approved once can be edited afterwards. A checksum
pin would close that gap and is not built.

Also by design, and so not vulnerabilities:

- Scripts inherit the environment and run with your privileges.
- A script's stderr is forwarded to the log.
- `rlt init --force` overwrites files, as documented; without `--force` it
  refuses.

## Dependencies

The `aixker-rlt` dependency is pinned to a specific revision and `Cargo.lock` is
committed, so builds are reproducible and an upstream change cannot arrive
unannounced. Vulnerabilities in the library itself belong in
[its repository](https://github.com/ali-heidari/Aixker-RLT).
