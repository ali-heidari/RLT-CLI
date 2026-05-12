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
cargo run -- export --checkpoint ./checkpoints/latest.pt --output ./exported/model.json --format json
```

## Parameters

### Global Options
- `--log-level [error|warn|info|debug|trace]`
  - Sets the CLI logging verbosity.
- `--config FILE`
  - Path to a JSON configuration file to override default settings.

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

### Export Subcommand
- `--checkpoint PATH`
  - Input checkpoint path for the trained model. Default: `./checkpoints/latest.json`.
- `--output PATH`
  - Output path for the exported model file. Default: `./exported/model.json`.
- `--format json`
  - Export format for the model. Currently only `json` is supported.

## Notes

This project follows `ai-agent-standards` conventions and includes a `docs/` folder with Markdown documentation.
