# Troubleshooting

Errors below are quoted as `rlt` actually prints them. All errors go to
stderr and exit with status 1.

## Configuration

### `config file not found: ./my-config.toml`

A file passed with `--config` does not exist. Check the path, remembering that
relative paths resolve from the current working directory.

A missing default `Config.toml` is **not** an error — the CLI then reports
`No config file found; using built-in defaults.` and runs with the defaults
listed in [usage.md](usage.md).

### `failed to parse config file 'Config.toml': TOML parse error at line 1, column 14`

The message includes the offending line, column, and expected type:

```text
Error: failed to parse config file './Config.toml': TOML parse error at line 1, column 14
  |
1 | batch_size = "not a number"
  |              ^^^^^^^^^^^^^^
invalid type: string "not a number", expected u32
```

Fix the value's type. Unknown keys are ignored rather than rejected, so a typo'd
key name silently does nothing — check spelling against the table in
[usage.md](usage.md) if a setting seems to have no effect.

### A config file value seems to be ignored

Command-line flags win over the config file. The settings block names the origin
of every value:

```text
  batch size: 99 (flag)          <- a flag overrode the file
  batch size: 8 (Config.toml)    <- the file was used
  batch size: 32 (default)       <- neither set it
```

`mode` in the config file is ignored by design: the subcommand decides whether
the run trains or infers.

### `dataset must be set in the config file or via --dataset`

Neither `--dataset` nor a `dataset` key was provided. Same for
`model_name must be set ...`.

## Datasets

### `dataset not found: ./data.csv`

The path does not exist or is not a file. This is checked before training, and
also under `--dry-run`.

### `feature width mismatch at row 1: input_number is 12 but the row has 6 value(s)`

`input_number` does not match your data. Set it to the number of columns your
rows actually have — the message tells you which value to use. A mismatch found
mid-file names the offending row and stops the run.

### `failed to read the first row: line 1, column 1: 'cpu' is not a number`

The first line is a header. Pass `--has-header` (or set `has_header = true`) to
skip it.

The first row is validated up front, so an unreadable first line is fatal while
later bad rows are merely skipped. That asymmetry is deliberate: a bad first row
usually means the whole file is being read wrongly.

### `line 42, column 3: 'n/a' is not a number` / `line 42, column 3: empty field`

A field is not numeric, or is blank. The row is skipped and counted; the run
continues and reports `read 1841 row(s) from the data source, skipped 3` at the
end.

Values are never silently replaced with `0.0`. If your data legitimately
contains blanks or sentinels, preprocess them before training — an imputed value
is a modelling decision the CLI will not make for you.

### `data source failed 10 times in a row, last error: ...`

Ten consecutive failures end the run, on the basis that the source is broken
rather than imperfect. Check the preceding `skipping row` warnings for the
pattern.

### `data source produced no rows`

The file is empty, or every line is blank.

### Non-numeric columns (ids, timestamps, categories)

There is no column selection yet, so every column must be numeric. Preprocess
identifier, timestamp, and categorical columns into numbers first, or drop them.
Tracked in [roadmap.md](roadmap.md) (§3, dataset handling).

## Python scripts

### The run hangs and nothing happens

Almost certainly an old-style reward script using `json.load(sys.stdin)`, which
waits for end-of-input while the CLI waits for a response.

Scripts are now started **once** and must loop:

```python
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps(response), flush=True)
```

See [reward_factory.md](reward_factory.md) and
[data_providers.md](data_providers.md).

### `Python script './provider.py' exited (exit status: 0): failed to send a request: Broken pipe`

The script printed one line and exited — the old single-shot style. Wrap its
body in the stdin loop above.

### `Python script './provider.py': the script closed its output without answering`

The script died, usually at startup. Its stderr is inherited, so the traceback
appears in your terminal just above this message.

### `no Python interpreter found on PATH (tried python3, python)`

Install Python, or make it available on `PATH`. On Windows, tick *"Add python.exe
to PATH"* in the installer.

### `reward script failed: invalid JSON reward response '...'`

The script printed something other than `{"reward": <number>, "success": <bool>}`.
That step falls back to `(0.0, false)` and the run continues — repeated
occurrences mean your objective is not being applied at all. Check for stray
`print()` calls writing to stdout; diagnostics belong on stderr.

## Models and checkpoints

### `checkpoint not found: ./models/m.json.m.json`

Nothing has been trained under that name yet, or `--model-name` is misspelled.

The doubled file name is expected: the library builds it from the model name
twice (see [found-issues.md](found-issues.md), issue 1). The CLI always reports
the real path.

### `checkpoint is empty: ./models/m.json.m.json. The model was never saved`

Training ran but never completed a batch, so no weights were written. You will
have seen this warning during training:

```text
WARN no model was saved to ./models/m.json.m.json: the run ended before a full
     batch completed. Check that the dataset holds enough rows for batch_size
     and total_batches.
```

The dataset needs enough rows to fill at least one batch. Reduce `batch_size`,
or use more data.

### Inference returns implausible actions

Confirm the checkpoint is the one you trained: the CLI logs
`loading checkpoint ./models/...` at info level. A missing or empty checkpoint is
now rejected rather than being replaced with random weights, so this should no
longer happen silently.

## Performance

### Inference is extremely slow

`infer` sleeps `interval_secs` (default `10`) between samples, because it is
built for polling a live provider. Over a file that is 10 seconds per row. Set
`interval_secs = 0` in the config file for batch inference.

### Training with Python scripts is slow

It should not be: interpreters are started once and kept alive, worth roughly
130× on a run using both a provider and a reward script. If a run is still slow,
the cost is inside your script — `scripts/system_metrics.py`, for instance,
sleeps 0.1 s per sample to measure CPU usage.

### Training is CPU-bound

Try `--backend gpu`, a smaller `hidden_layers`, or a larger `batch_size`.

## Getting help

Re-run with `-v` for debug-level logging, which includes per-sample detail.
When reporting a problem, include the settings block from the start of the run —
it shows every effective value and where it came from.
