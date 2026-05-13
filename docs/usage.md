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
cargo run -- infer --model-name my-model.json --features "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"
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
  - Path to the training dataset or environment configuration. Default: `./data/train`.
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

### Infer Subcommand
- `--model-name NAME`
  - Checkpoint file name for the trained model to load. Default: `model.json`.
- `--features FEATURES`
  - Comma-separated list of input features as floats (e.g., "1.0,2.0,3.0"). If not provided, uses dummy values.

### Export Subcommand
- `--checkpoint PATH`
  - Input checkpoint path for the trained model. Default: `./checkpoints/latest.json`.
- `--output PATH`
  - Output path for the exported model file. Default: `./exported/model.json`.
- `--format json`
  - Export format for the model. Currently only `json` is supported.

## Notes

This project follows `ai-agent-standards` conventions and includes a `docs/` folder with Markdown documentation.
