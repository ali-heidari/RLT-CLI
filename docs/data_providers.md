# Data Providers

`RLT-CLI` supports multiple data sources for training. The data provider type is determined by the file extension of the `--dataset` argument.

## CSV Data Provider

The CSV provider reads comma-separated feature vectors from a text file, one per line.

### Format

Each line contains comma-separated floats:

```
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85
...
```

### Usage

```bash
cargo run -- train --dataset ./data/train.csv --model-name my-model.json
```

## Python Script Data Provider

The Python script provider calls a Python script to fetch feature vectors on demand. This is useful for dynamic data sources like system metrics, real-time sensors, or simulations.

### How it works

1. RLT-CLI runs the Python script each time new features are needed.
2. The script outputs feature data on stdout.
3. The output is parsed as either JSON array or comma-separated floats.
4. If parsing fails, training stops.

### Output format

The Python script should output either:

**JSON array:**
```json
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
```

**Comma-separated floats:**
```
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
```

### Usage

```bash
cargo run -- train --dataset ./system_metrics.py --model-name my-model.json
```

### Sample script

A sample Python script is included as `system_metrics.py` that outputs system metrics:
- CPU usage percentage
- Memory usage percentage
- Disk usage percentage
- Load averages (1, 5, 15 minute)
- Number of processes
- Uptime in days

The script requires `psutil` for full functionality. Install it with:

```bash
pip install psutil
```

### Notes

- Python must be installed as `python3` or `python`.
- If the script fails or produces invalid output, training stops.
- The script is called in a subprocess with no stdin. Ensure the script is self-contained.
- For data providers that require state, maintain state in the script using file-based persistence or in-memory state.
