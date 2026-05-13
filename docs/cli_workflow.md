# CLI Workflow

This document explains the `RLT-CLI` command flow and how the training, inference, and export paths operate.

## Workflow Diagram

```mermaid
flowchart TD
    A[Start CLI] --> B{Command}
    B --> |train| C[Load config and dataset]
    B --> |infer| D[Load config and model]
    B --> |export| E[Copy checkpoint file]
    C --> F{File extension}
    F --> |.csv| G[Open CSV data provider]
    F --> |.py| H[Open Python script data provider]
    G --> I{Reward script configured?}
    H --> I
    I --> |yes| J[Validate reward script path]
    I --> |no| K[Use default reward callback]
    J --> L[Run Python reward script per sample]
    L --> M[Supply reward and success to node]
    K --> M
    M --> N[Start training node]
    N --> O[Training worker processes batches]
    D --> P[Run inference on input features]
    P --> Q[Print predicted action]
    E --> R[Export model artifact]
```

## Training path

1. `cargo run -- train` starts the CLI in training mode.
2. Configuration values are loaded from the TOML config file and command-line overrides.
3. The data source is identified by file extension:
   - `.csv`: Reads comma-separated feature vectors from a text file
   - `.py`: Calls a Python script to fetch feature vectors on demand
4. The first row/sample is validated.
5. If `reward_script` is configured, the path is validated.
6. The reward factory executes the Python reward script for each sample (if configured) and receives `reward` and `success`.
7. The training node is started with the configured model name.
8. A background training worker processes batches and updates the model.

## Inference path

1. `cargo run -- infer` loads the model and optionally parses input features.
2. The model produces actions from the given feature vector.
3. The CLI outputs the selected action.

## Export path

1. `cargo run -- export` copies a checkpoint file to the requested output location.
2. This path is used for packaging trained models or converting artifacts.

## Notes

- The data provider type is automatically selected based on the dataset file extension (`.csv` or `.py`).
- The reward script is optional. If absent, the CLI uses a default reward callback.
- If the reward script path is invalid, the CLI exits with an error before training starts.
- If a Python data script fails or produces invalid output, the CLI exits with an error.
- The `train` path additionally supports `--reward-script` and `reward_script` in `Config.toml`.
