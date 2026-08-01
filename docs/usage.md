# RLT-CLI Usage

`RLT-CLI` is a Rust command-line interface for Aixker-RLT model training,
inference, and export.

## Build

```bash
cargo build --release
```

The binary is produced at `target/release/RLT-CLI`. Windows builds work the same
way (`target\release\RLT-CLI.exe`); see the
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
RLT-CLI train --dataset ./your-data.csv --epochs 20 --batch-size 64 \
  --model-name my-model.json
```

Train from a Python data provider with a Python reward script:

```bash
RLT-CLI train --dataset ./scripts/system_metrics.py --model-name my-model.json \
  --reward-script ./scripts/reward_script.py
```

Run inference:

```bash
RLT-CLI infer --dataset ./your-data.csv --model-name my-model.json
```

Export a trained checkpoint:

```bash
RLT-CLI export --output ./exported/model.json
```

## Where checkpoints live

Checkpoints are written under `./models/` relative to the working directory.
The file name currently repeats the model name — `--model-name my-model.json`
produces `./models/my-model.json.my-model.json`. That doubling comes from the
library and is recorded in [found-issues.md](found-issues.md) (issue 1); the CLI
reports the real path in its messages so you can always find the file.

## Parameters

### Global options

- `--log-level [silent|error|warn|info|debug|trace]`
  - Logging verbosity. Default: `info`.
- `-v`, `--verbose`
  - Debug-level logging. Overrides `--log-level`.
- `-q`, `--silent` (alias `--quiet`)
  - Suppresses **all** CLI output, including the settings block.
- `--errors-only`
  - Only error logs. Overrides `--log-level`.
- The three convenience flags are mutually exclusive.
- `--config FILE`
  - Path to a TOML config file. Defaults to `Config.toml` if present.

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

Inference sleeps `interval_secs` between samples (default `10`), because its
intended use is a polling decision loop against a live provider. Set
`interval_secs = 0` in the config file when running inference over a file.

The checkpoint must exist and be non-empty; otherwise the command fails rather
than running an untrained model.

### `export`

- `--checkpoint PATH`
  - Checkpoint to export. Defaults to the configured model under `./models/`.
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
| `interval_secs` | — | `10` |
| `log_interval` | — | `64` |
| `backend` | `--backend` | `"Cpu"` |
| `has_header` | `--has-header` | `false` |
| `delimiter` | `--delimiter` | `","` |

`input_number` must match the number of values in each row of your data; a
mismatch is reported before training starts.

Unknown keys are ignored. `mode` in particular has no effect — the subcommand
decides whether the run trains or infers.

## Tests

```bash
cargo test
```

Unit tests cover CSV parsing, the data-source boundary, and config precedence.
End-to-end tests drive the built binary in a temporary directory.

## Notes

This project follows `ai-agent-standards` conventions and keeps human-readable
documentation in [docs/](index.md).
