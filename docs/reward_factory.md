# Reward Factory

`RLT-CLI` now supports user-provided Python reward scripts for training.

## How it works

During training, the CLI can invoke a Python script to compute reward and success values for each experience. The Rust reward factory sends a JSON request to the script on stdin and expects a JSON response on stdout.

### Request format

The Python script receives an object like:

```json
{
  "features": [0.1, 0.2, 0.3],
  "action": 1
}
```

### Response format

The script must print a JSON object containing `reward` and `success`:

```json
{
  "reward": 0.5,
  "success": true
}
```

## CLI usage

Use `--reward-script` with the `train` subcommand:

```bash
cargo run -- train --dataset ./data/train.csv --model-name my-model.json --reward-script ./reward_script.py
```

You can also set `reward_script` in `Config.toml`:

```toml
reward_script = "./reward_script.py"
```

## Sample script

A sample script is included as `reward_script.py` in the repository root.

The script demonstrates a simple reward rule:
- reward is positive when the selected action matches a heuristic condition
- `success` is `true` when reward is positive

## Notes

- Python must be installed on the machine as `python3` or `python`.
- If the reward script fails or returns invalid JSON, the CLI falls back to a default reward of `0.0` and `success: false`.
