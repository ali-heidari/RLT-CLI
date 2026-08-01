# Data Providers

`RLT-CLI` selects a data provider from the file extension of `--dataset`:
`.py` uses the Python script provider, anything else is read as CSV.

You supply your own data. No dataset ships with this repository.

## CSV Data Provider

Reads one feature vector per line. Parsing uses the `csv` crate, so quoted
fields, embedded delimiters and escapes are handled correctly.

### Format

Every field must be a number, and every row must hold exactly `input_number`
of them:

```csv
0.1,0.2,0.3,0.4,0.5,0.6
0.15,0.25,0.35,0.45,0.55,0.65
```

### Options

| Flag | Config key | Default | Meaning |
| --- | --- | --- | --- |
| `--has-header [true\|false]` | `has_header` | `false` | Skip the first row instead of parsing it as data |
| `--delimiter CHAR` | `delimiter` | `,` | Field separator; must be a single ASCII character |

```bash
RLT-CLI train --dataset ./your-data.csv --model-name my-model.json --has-header
RLT-CLI train --dataset ./your-data.tsv --model-name my-model.json --delimiter $'\t'
```

### How bad data is handled

Values are never silently substituted. A field that is not a number, and an
empty field, are both errors naming the line and column:

```text
skipping row 41: line 42, column 3: 'n/a' is not a number
skipping row 58: line 59, column 5: empty field
```

- **A bad row is skipped**, counted, and reported at the end
  (`read 1841 row(s) from the data source, skipped 3`). One bad line does not
  end a long run.
- **Ten consecutive failures end the run**, on the basis that the source is
  broken rather than merely imperfect.
- **The first row must be valid.** It is checked before training starts, so a
  file whose first line is a header (or otherwise unreadable) fails immediately
  rather than after an hour.
- **Every row must have `input_number` fields.** A mismatch stops the run and
  names the offending row.

An empty field is an error rather than `0.0` on purpose: a missing measurement
is not a measurement of zero, and treating it as one trains the model on data
that was never observed.

### Columns must all be numeric

There is no column selection yet. A CSV containing identifiers, timestamps or
categorical text must be preprocessed into numeric columns before training.
Column selection is tracked in [roadmap.md](roadmap.md) (§3, dataset handling).

## Python Script Data Provider

Fetches feature vectors from a Python script on demand — useful for live system
metrics, sensors, or simulations.

### Protocol

**The script is started once and kept running for the whole run.** For each
sample `RLT-CLI` writes one request line to the script's stdin and reads one
response line from its stdout.

- **Request:** a JSON object, currently always `{}`. It is an object rather than
  a bare newline so fields can be added later without breaking existing scripts;
  ignore its contents.
- **Response:** one line holding either a JSON array of numbers or
  comma-separated floats.

```python
#!/usr/bin/env python3
import json
import sys

for _request in sys.stdin:
    print(json.dumps(collect_features()), flush=True)
```

> **Breaking change.** Earlier versions ran `python3 script.py` once per sample,
> so a script that printed one line and exited was correct. Such a script now
> fails after its first sample with `exited (exit status: 0): failed to send a
> request: Broken pipe`. Wrap the body in `for _request in sys.stdin:` as above.

The interpreter is started with `-u` (unbuffered), so a missing `flush=True`
will not stall the run. Keeping `flush=True` is still good practice.

### Why it changed

Starting an interpreter per sample cost 20–50 ms of pure overhead per step, and
two of them per step when a reward script was also in use. On a measured
416-sample run with both a provider and a reward script:

| | wall time |
| --- | --- |
| interpreter per sample | 26.24 s |
| one interpreter, kept alive | 0.20 s |

### Failures

- If the script exits, the CLI reports the exit status and the last error rather
  than hanging.
- **A script that reads a request and never answers times out** after
  `--script-timeout` seconds (default `30`):

  ```text
  Python script './provider.py': it stopped answering after 30s. Raise
  --script-timeout if the script is legitimately slow, or set it to 0 to
  wait forever.
  ```

  A worker that has missed one deadline is not used again, even if the script
  recovers: a late answer would be paired with the *next* request and quietly
  mismatch every feature vector after it.
- Stderr is inherited, so tracebacks appear in your terminal as they happen.
- A response that is not numeric is an error naming the offending value.
- Provider failures follow the same skip/count/abort rules as CSV rows above.

### Timeout

| Flag | Config key | Default | Meaning |
| --- | --- | --- | --- |
| `--script-timeout SECS` | `script_timeout_secs` | `30` | Seconds to wait for one response; `0` waits forever |

The deadline applies to both the data provider and the
[reward script](reward_factory.md). Raise it for a script that calls a slow API;
set it to `0` only if you would rather have the CLI wait indefinitely than fail.

### Usage

```bash
RLT-CLI train --dataset ./scripts/system_metrics.py --model-name my-model.json
```

### Sample script

[`scripts/system_metrics.py`](../scripts/system_metrics.py) reports twelve
normalised metrics read from `/proc` and `shutil.disk_usage` — CPU, memory,
load, process count, uptime and derived values. It needs no third-party
packages, and it is Linux-specific because it reads `/proc` directly.

Set `input_number = 12` to use it.

### Notes

- Python must be on `PATH` as `python3` or `python`.
- The script may keep state between samples in ordinary local variables, since
  the process is no longer restarted.
- Scripts are executed as given. Only point `--dataset` at a script you trust.
