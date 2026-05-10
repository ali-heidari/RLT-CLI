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
cargo run -- export --checkpoint ./checkpoints/latest.pt --output ./exported/model.json --format json
```

## Notes

This project follows `ai-agent-standards` conventions and includes a `docs/` folder with Markdown documentation.
