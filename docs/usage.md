# Usage

`rlt` is the command installed by `aixker-rlt-cli`: a Rust command-line
interface for Aixker-RLT model training, evaluation, inference, and export.

## Build

```bash
cargo build --release
```

The binary is produced at `target/release/rlt`. Windows builds work the same
way (`target\release\rlt.exe`); see the
[README "Build (Windows)" section](../README.md#build-windows) for toolchain
setup, Python-on-PATH notes, and cross-compilation from Linux.

## Where settings come from

Every setting is resolved in this order, first match winning:

1. **command-line flag**
2. **config file** — `--config FILE`, or `Config.toml` in the working directory
3. **built-in default**

The settings block printed at the start of each run names the origin of every
value, so there is no guessing:

```text
Starting training with the following settings:
  dataset: ./your-data.csv (flag)
  model name: my-model.json (Config.toml)
  batch size: 32 (default)
```

A config file passed with `--config` must exist. The default `Config.toml` is
optional — without one, built-in defaults apply.

## Run

Train from a CSV dataset (supply your own data):

```bash
rlt train --dataset ./your-data.csv --epochs 20 --batch-size 64 \
  --model-name my-model.json
```

Train from a Python data provider with a Python reward script:

```bash
rlt train --dataset ./scripts/system_metrics.py --model-name my-model.json \
  --reward-script ./scripts/reward_script.py
```

Run inference:

```bash
rlt infer --dataset ./your-data.csv --model-name my-model.json
```

Export a trained checkpoint:

```bash
rlt export --output ./exported/model.json
```

## Interrupting a run

Ctrl-C ends `train`, `infer` and `eval` **cleanly** rather than killing them.
The run stops asking for data, winds down the way it would at the end of the
dataset, reports its totals, and exits `0`:

```text
WARN  interrupted: finishing the current step and saving what is done
INFO  read 10489 row(s) from the data source, skipped 0
INFO  checkpoint written to ./models/my-model.json.my-model.json (3150 bytes)
```

So the last completed batch's checkpoint survives, and `infer --output` keeps
the decisions written so far, instead of the process dying part-way through a
write.

A **second** Ctrl-C exits immediately with status `130`, without saving. The
wind-down can itself be slow — waiting on a Python script mid-request, for
instance — and handling the first signal would otherwise leave no way out.

## Where checkpoints live

Checkpoints are written under `./models/` **relative to the working directory**,
so the same command run from two directories reads and writes different files.
Messages naming a checkpoint print the absolute path for that reason:

```text
INFO  checkpoint written to /home/you/project/models/my-model.json.my-model.json (3150 bytes)
```

The file name currently repeats the model name — `--model-name my-model.json`
produces `./models/my-model.json.my-model.json`. That doubling comes from the
library and is recorded in [found-issues.md](found-issues.md) (issue 1); the CLI
reports the real path in its messages so you can always find the file.

## Parameters

### Global options

- `--log-level [silent|error|warn|info|debug|trace]`
  - Logging verbosity. Default: `info`. `--log-level silent` is exactly
    equivalent to `--silent`.
- `-v`, `--verbose`
  - Debug-level logging. Overrides `--log-level`.
- `-q`, `--silent` (alias `--quiet`)
  - Suppresses **all** CLI output, including the settings block, the inference
    records, and anything a Python script writes to stderr. A run that *fails*
    still reports why on stderr and exits non-zero — silencing the reason for a
    failure would be worse than useless.
- `--errors-only`
  - Only error logs. Overrides `--log-level`.
- The three convenience flags are mutually exclusive.
- `--config FILE`
  - Path to a TOML config file. Defaults to `Config.toml` if present.

### `init`

Scaffolds a config, a reward script, a heuristic baseline, and generated
train/holdout datasets. See [init.md](init.md).

- `[DIR]` — where to scaffold. Default: the current directory.
- `--force` — overwrite files that already exist. Without it, `init` refuses
  rather than replacing a reward script you wrote.

### `train`

- `--dataset PATH`
  - Data source. `.py` selects the Python script provider, anything else is read
    as CSV. See [data_providers.md](data_providers.md).
  - Required, via flag or `dataset` in the config file.
- `--epochs N`
  - Training length. **One epoch is 100 training batches**, so `--epochs 20`
    means 2000 batches. Default: `10`. A config file can set `total_batches`
    directly instead; `--epochs` takes precedence over it.
- `--batch-size N`
  - Batch size. Default: `32`.
- `--dry-run`
  - Resolve and validate the configuration, print it, and exit without training.
    Fails if the dataset or reward script is missing.
- `--model-name NAME`
  - Checkpoint name. Required, via flag or `model_name` in the config file.
- `--reward-script PATH`
  - Python script computing reward and success per step. See
    [reward_factory.md](reward_factory.md).
- `--script-timeout SECS`
  - How long a Python script may take to answer one request, for both the data
    provider and the reward script. Default: `30`. `0` waits forever, which is
    what the CLI did before the deadline existed. Reported in the settings block
    only when a script is actually in use.
- `--backend [cpu|gpu]`
  - Compute backend. Default: `cpu`. `gpu` uses wgpu compute shaders
    (Vulkan/Metal/DX12 — NVIDIA, AMD, Intel, Apple Silicon; no CUDA required)
    and falls back to CPU with a warning when no adapter is found.
- `--has-header [true|false]`
  - Skip the first CSV row instead of parsing it as data. Default: `false`.
- `--delimiter CHAR`
  - CSV field separator; a single ASCII character. Default: `,`.

### `infer`

- `--dataset PATH` — as for `train`. Required.
- `--model-name NAME` — checkpoint to load. Required.
- `--backend [cpu|gpu]` — as for `train`.
- `--has-header [true|false]`, `--delimiter CHAR` — as for `train`.
- `--script-timeout SECS` — as for `train`.
- `--interval-secs SECS`
  - Seconds slept between samples. Default: `0` for a file dataset, `10` for a
    `.py` provider. See [Polling interval](#polling-interval) below.
- `--output PATH`
  - Where the decisions go. `-` is stdout. Default: `-`. Parent directories are
    created as needed.
- `--with-features`
  - Include the input features in each record. Default: off.

#### Output

One JSON object per sample, one per line:

```console
$ rlt infer --dataset ./infer.csv --model-name my-model.json
{"row":1,"action":2}
{"row":2,"action":1}
{"row":3,"action":2}
```

`--with-features` adds the input that produced each decision:

```console
$ rlt infer --dataset ./infer.csv --model-name my-model.json --with-features
{"row":1,"features":[0.1,0.2,0.3,0.4,0.5,0.6],"action":0}
```

`row` is the number of the row **in the data source**, so a row that was skipped
as unparseable leaves a gap rather than shifting every number after it.

Records go to stdout and logs go to stderr, so the stream is machine-readable
without any flags:

```bash
rlt infer --dataset ./infer.csv --model-name my-model.json 2>/dev/null | jq -c
```

`--silent` prints nothing, so it suppresses the records too — unless you also
pass `--output PATH`, in which case the file is still written. A file you asked
for is the result of the command, not chatter.

#### Polling interval

Inference sleeps between samples, because one of its uses is a polling decision
loop against a live provider. The default depends on the dataset:

| Dataset | Default interval | Why |
| --- | --- | --- |
| A file (CSV) | `0` | The rows are already there; there is nothing to wait for |
| A `.py` provider | `10` | It is sampling a live source |

`--interval-secs SECS`, or `interval_secs` in the config file, overrides both —
the rule replaces the default, not your configuration. The settings block names
which applied:

```text
  interval: 0s (file dataset)
  interval: 10s (default)
  interval: 5s (flag)
```

Training never sleeps: the library only polls in inference mode, so
`interval_secs` has no effect on `train`.

The checkpoint must exist and be non-empty; otherwise the command fails rather
than running an untrained model.

### `eval`

Scores a checkpoint against a held-out dataset, optionally next to a baseline
policy. Full details in [eval.md](eval.md).

- `--dataset PATH` — the held-out data. Required.
- `--model-name NAME` — checkpoint to score. Required.
- `--reward-script PATH` — **required.** Without it every reward is `0.0` and
  the report would be zeros presented as an answer.
- `--baseline SPEC` — `static:N`, `round-robin`, `random`, or a `.py` heuristic.
  Optional; without it only the policy's own numbers are reported.
- `--format [text|json]` — human-readable report, or one JSON object for CI.
  Default: `text`.
- `--backend`, `--has-header`, `--delimiter`, `--script-timeout` — as for
  `train`.

Evaluation never sleeps between samples: it is a batch pass over a held-out
file, so `interval_secs` does not apply.

### `inspect`

Reports what a checkpoint actually holds. Checkpoints are JSON, but that only
makes them technically readable — four tensors of bare numbers and a snapshot.

- `--checkpoint PATH` — the checkpoint, as a path.
- `--model-name NAME` — resolved under `./models/`. Mutually exclusive with
  `--checkpoint`.
- `--format [text|json]` — default `text`.

```console
$ rlt inspect --model-name my-model.json
Checkpoint: ./models/my-model.json.my-model.json (4169 bytes)
Node id:    my-model.json

Architecture
  12 -> 16 -> 3
  259 parameters

Tensors
  b1   [16]           16 values   min      -0.0000  max    3446.6870  mean     659.1670
  w1   [12, 16]      192 values   min      -0.0982  max  887513.5000  mean   40460.4545

Warnings
  w1 reaches 8.8751e5: the run probably diverged rather than converged
```

The **Warnings** section is the reason to reach for this command. A model whose
weights have run away to `1e6`, or that holds a non-finite value, still loads
and still answers — it just answers badly, which is indistinguishable from a
model that merely trained poorly. Warnings also go to the log, so they show up
in CI output that only captures logs.

The last training snapshot is printed too: the logits, probabilities, reward and
action counts from the final batch. Logits like `[0.0, -50.0, -50.0]` mean the
policy has collapsed onto one action.

### `export`

- `--checkpoint PATH`
  - Checkpoint to export, as a path. Defaults to the configured model under
    `./models/`.
- `--model-name NAME`
  - Checkpoint to export, resolved under `./models/` the same way `train` and
    `infer` resolve it. Mutually exclusive with `--checkpoint`; passing both is
    an error rather than one silently winning.
- `--output PATH`
  - Destination. Default: `./exported/model.json`. Parent directories are
    created as needed.
- `--format json`
  - The only format. Checkpoints are already JSON, so **the file is copied
    verbatim; no conversion is performed.** Real conversion targets
    (safetensors, onnx) are tracked in [roadmap.md](roadmap.md).

## Config file keys

All keys are optional. See [Config.sample.toml](../Config.sample.toml).

| Key | Flag | Default |
| --- | --- | --- |
| `dataset` | `--dataset` | — (required) |
| `model_name` | `--model-name` | — (required) |
| `reward_script` | `--reward-script` | none |
| `batch_size` | `--batch-size` | `32` |
| `total_batches` | `--epochs` × 100 | `1000` |
| `input_number` | — | `8` |
| `output_number` | — | `3` |
| `hidden_layers` | — | `16` |
| `reply_capacity` | — | `4096` |
| `interval_secs` | `--interval-secs` | `0` for a file, `10` for a `.py` provider |
| `log_interval` | — | `64` |
| `backend` | `--backend` | `"Cpu"` |
| `has_header` | `--has-header` | `false` |
| `delimiter` | `--delimiter` | `","` |
| `script_timeout_secs` | `--script-timeout` | `30` |

`input_number` must match the number of values in each row of your data; a
mismatch is reported before training starts.

An unrecognised key is reported as a warning and otherwise ignored:

```text
WARN unknown key 'bacth_size' in Config.toml has no effect.
```

`mode` and `debug` are the exceptions: they are accepted silently because they
appear in configs written for earlier versions. Neither does anything — the
subcommand decides whether the run trains or infers, and `--log-level` controls
logging.

## Tests

```bash
cargo test
```

Unit tests cover CSV parsing, the data-source boundary, and config precedence.
End-to-end tests drive the built binary in a temporary directory.

## Notes

This project follows `ai-agent-standards` conventions and keeps human-readable
documentation in [docs/](index.md).
