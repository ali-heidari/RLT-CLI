# RLT-CLI

A lightweight Rust command-line interface for [@ali-heidari/Aixker-RLT](https://github.com/ali-heidari/Aixker-RLT) model training, inference, and export workflows.

## Overview

`RLT-CLI` provides a simple CLI wrapper around AIXKER-RLT concepts for training reinforcement learning models, running inference, and exporting trained artifacts.

It supports two Python integration points:

1. **Python reward factory**: Training can call a user-provided Python script for reward computation. The script receives feature and action data on stdin and returns JSON containing `reward`.

2. **Python data providers**: Training and inference can fetch feature vectors from a Python script on demand, enabling integration with system metrics, sensors, simulations, or other dynamic data sources.

## CLI workflow

The `RLT-CLI` workflow is:

```mermaid
flowchart TD
    A[Start CLI] --> B{Command}
    B --> |train| C[Load config and dataset]
    B --> |infer| D[Load model and infer]
    B --> |export| E[Copy checkpoint file]
    C --> F{Reward script?}
    F --> |yes| G[Validate script path]
    G --> H[Run Python reward script per sample]
    F --> |no| I[Use default reward]
    H --> J[Start training node]
    I --> J
    D --> K[Infer actions]
    J --> L[Background training worker]
    K --> M[Print action]
    E --> N[Export output]
```

See [docs/cli_workflow.md](docs/cli_workflow.md) for a detailed diagram and explanation.

## Features

- `train` subcommand for starting model training
- `infer` subcommand for running inference against a trained model
- `export` subcommand for exporting trained models
- Python reward factory and Python data provider integration
- Global options for configuration files and logging level
- Built in Rust with `clap` for command parsing

## Getting Started

### Build

```bash
cargo build --release
```

### Run

Train from a CSV dataset:

```bash
cargo run -- train --dataset ./sample-data/vmCloud_data.csv --epochs 20 --batch-size 64 --learning-rate 0.001 --model-name my-model.json
```

Train using a Python data provider and a Python reward script:

```bash
cargo run -- train --dataset ./scripts/system_metrics.py --epochs 20 --batch-size 64 --model-name my-model.json --reward-script ./scripts/reward_script.py
```

Run inference:

```bash
cargo run -- infer --dataset ./sample-data/vmCloud_data.csv --model-name my-model.json
```

Export a trained model:

```bash
cargo run -- export --checkpoint ./checkpoints/latest.pt --output ./exported/model.json --format json
```

### Dry Run

Use `--dry-run` with `train` to validate CLI arguments and show the configured settings without actually starting training. This is useful for confirming your dataset path, hyperparameters, and model name before committing to a full run.

### Configuration

You can provide a TOML configuration file using `--config Config.toml` to set default values. Command-line flags override config file settings. See [Config.sample.toml](Config.sample.toml) for a sample configuration.

## Project Structure

- `Cargo.toml` — Rust package manifest
- `src/main.rs` — CLI entry point
- `src/cli/` — command definitions and argument parsing
- `src/providers/` — CSV and Python-script data providers
- `src/reward_factory.rs` — Python reward script integration
- `scripts/` — sample Python reward and data-provider scripts
- `sample-data/` — sample CSV datasets
- `docs/` — human-readable project documentation
- `.agent/` — AI agent instructions and shared standards
- `LICENSE` — project license

## Documentation

For detailed usage instructions, troubleshooting, and feature guides, see the [documentation index](docs/index.md):

- [usage.md](docs/usage.md)
- [cli_workflow.md](docs/cli_workflow.md)
- [reward_factory.md](docs/reward_factory.md)
- [data_providers.md](docs/data_providers.md)
- [troubleshooting.md](docs/troubleshooting.md)

## Standards

This repository follows the `ai-agent-standards` conventions. The AI agent guidance is documented in [.agent/agent-instructions.md](.agent/agent-instructions.md), and the shared standard files are available at:

- [ali-heidari/ai-agent-standards](https://github.com/ali-heidari/ai-agent-standards)

## License

Apache-2.0
