# RLT-CLI

A lightweight Rust command-line interface for Aixker-RLT model training and export workflows.

## Overview

`RLT-CLI` provides a simple CLI wrapper around AIXKER-RLT concepts for training reinforcement learning models and exporting trained artifacts.

## Features

- `train` subcommand for starting model training
- `export` subcommand for exporting trained models
- Global options for configuration files and logging level
- Built in Rust with `clap` for command parsing

## Getting Started

### Build

```bash
cargo build --release
```

### Run

```bash
cargo run -- train --dataset ./data/train --epochs 20 --batch-size 64 --learning-rate 0.001 --model-name my-model.json
```

```bash
cargo run -- export --checkpoint ./checkpoints/latest.pt --output ./exported/model.json --format json
```

### Dry Run

Use `--dry-run` with `train` to validate CLI arguments and show the configured settings without actually starting training. This is useful for confirming your dataset path, hyperparameters, and model name before running a long job.

## Project Structure

- `Cargo.toml` — Rust package manifest
- `src/main.rs` — CLI entry point and command definitions
- `copilot-instructions.md` — agent usage instructions and standards reference
- `docs/` — human-readable project documentation
- `LICENSE` — project license

## Documentation

This repository includes a `docs/` folder containing Markdown documentation for users and maintainers.

## Standards

This repository follows the `ai-agent-standards` conventions. The AI agent guidance is documented in `copilot-instructions.md`, and the shared standard files are available at:

- `https://github.com/ali-heidari/ai-agent-standards`

## License

Apache-2.0

This repository follows the `ai-agent-standards` instruction conventions. The main AI agent guidance is documented in `copilot-instructions.md`, and the shared standard files are available from the `ai-agent-standards` repository.

For more usage information, see the docs index:
- `docs/index.md`

See also:
- `copilot-instructions.md`
- `https://github.com/ali-heidari/ai-agent-standards`
