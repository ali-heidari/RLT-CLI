# RLT-CLI Usage

`RLT-CLI` is a Rust command-line interface for Aixker-RLT model training and export.

## Build

```bash
cargo build --release
```

## Run

```bash
cargo run -- train --dataset ./data/train --epochs 20 --batch-size 64 --learning-rate 0.001
```

```bash
cargo run -- train --dataset ./data/train --epochs 20 --batch-size 64 --learning-rate 0.001 --model-name my-model.json
```

```bash
cargo run -- train --dataset ./data/train --epochs 20 --batch-size 64 --learning-rate 0.001 --dry-run --model-name my-model.json
```

```bash
cargo run -- infer --dataset ./data/train.csv --model-name my-model.json
```

```bash
cargo run -- infer --dataset ./system_metrics.py --model-name my-model.json
```

Train or infer on the GPU instead of the CPU (default):

```bash
cargo run -- train --dataset ./data/train.csv --model-name my-model.json --backend gpu
```

```bash
cargo run -- export --checkpoint ./checkpoints/latest.pt --output ./exported/model.json --format json
```

## Parameters

### Global Options
- `--log-level [error|warn|info|debug|trace]`
  - Sets the CLI logging verbosity.
- `--config FILE`
  - Path to a TOML configuration file to override default settings. Command-line flags take precedence over config file values. See `Config.toml` for a sample configuration.

### Train Subcommand
- `--dataset PATH`
  - Path to the training dataset file. Can be either a CSV file (`*.csv`) or a Python script (`*.py`).
    - **CSV**: Each line is parsed as comma-separated floats representing feature vectors.
    - **Python**: The script is called repeatedly, and its output is parsed as JSON array or comma-separated floats.
  - Default: `./data/train`.
- `--epochs N`
  - Number of training epochs. Default: `10`.
- `--batch-size N`
  - Batch size used during training. Default: `32`.
- `--learning-rate FLOAT`
  - Initial learning rate for the training algorithm. Default: `0.001`.
- `--dry-run`
  - Validate and print the resolved training settings without actually running model training.
- `--model-name NAME`
  - Checkpoint file name for the trained model. Default: `model.json`.
- `--reward-script PATH`
  - Optional Python script used to compute reward and success for each training sample.
    The script receives a JSON request on stdin and returns a JSON response on stdout.
- `--backend [cpu|gpu]`
  - Compute backend for the neural network math. Default: `cpu` (pure-Rust `ndarray`).
    `gpu` runs training on wgpu compute shaders (Vulkan/Metal/DX12 — NVIDIA, AMD, Intel,
    Apple Silicon; no CUDA required) and falls back to CPU with a warning if no
    compatible adapter is found. Can also be set with `backend = "Gpu"` in the config
    file; the flag takes precedence.

### Infer Subcommand
- `--dataset PATH`
  - **Required**. Path to the data source (CSV file or Python script) to fetch features for inference.
    - Can be a CSV file (`*.csv`) or Python script (`*.py`).
    - Features are read from the data source and passed to the trained model for inference.
  - Must be set in `Config.toml` or via `--dataset`.
- `--model-name NAME`
  - Checkpoint file name for the trained model to load. Default: `model.json`.
- `--backend [cpu|gpu]`
  - Compute backend for inference. Default: `cpu`. See the train subcommand for details.

### Export Subcommand
- `--checkpoint PATH`
  - Input checkpoint path for the trained model. Default: `./checkpoints/latest.json`.
- `--output PATH`
  - Output path for the exported model file. Default: `./exported/model.json`.
- `--format json`
  - Export format for the model. Currently only `json` is supported.

## Notes

This project follows `ai-agent-standards` conventions and includes a `docs/` folder with Markdown documentation.
