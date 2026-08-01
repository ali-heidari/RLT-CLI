# Reward Factory

During training `RLT-CLI` can delegate reward computation to a Python script of
your own, so the objective is defined where your domain knowledge lives.

Without a reward script the CLI returns a reward of `0.0` and `success: true`
for every step, which trains nothing useful — a reward script is what makes
training meaningful.

## Protocol

**The script is started once and kept running for the whole run.** For each
environment step `RLT-CLI` writes one request line to the script's stdin and
reads one response line from its stdout.

**Request** — one JSON object per line:

```json
{"features": [0.1, 0.2, 0.3], "action": 1}
```

`action` is a whole number in `0..output_number`.

**Response** — one JSON object per line, with both fields required:

```json
{"reward": 0.5, "success": true}
```

`reward` is a number, `success` a boolean.

## Minimal script

```python
#!/usr/bin/env python3
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    request = json.loads(line)
    reward, success = evaluate(request["features"], request["action"])
    print(json.dumps({"reward": reward, "success": success}), flush=True)
```

> **Breaking change.** Earlier versions ran `python3 script.py` once per reward,
> so `request = json.load(sys.stdin)` was correct. That form now **hangs**,
> because `json.load` waits for end-of-input while the CLI holds the pipe open
> waiting for a response. Convert the script to the loop shown above.

The interpreter is started with `-u` (unbuffered), so a missing `flush=True`
will not stall the run. Keeping `flush=True` is still good practice.

## Why it changed

Starting an interpreter per reward cost 20–50 ms per environment step. On a
measured 416-sample run using both a data provider and a reward script, moving
both to long-lived workers took the run from **26.24 s to 0.20 s**.

## Usage

```bash
RLT-CLI train --dataset ./your-data.csv --model-name my-model.json \
  --reward-script ./scripts/reward_script.py
```

Or in the config file:

```toml
reward_script = "./scripts/reward_script.py"
```

The flag takes precedence over the config file. The path is validated before
training starts, including under `--dry-run`.

## Sample script

[`scripts/reward_script.py`](../scripts/reward_script.py) implements a simple
rule: reward is positive when the chosen action matches a heuristic on the first
feature, and `success` mirrors the sign of the reward. It is a starting point,
not a useful objective.

## Failures

- **A script that will not start** fails the run immediately, with its stderr
  shown.
- **A script that dies mid-run** is reported with its exit status rather than
  hanging the CLI.
- **A script that never answers** times out after `--script-timeout` seconds
  (default `30`, `0` waits forever) and counts as a failure like any other. The
  first timeout costs the full wait; after that the worker is not used again, so
  the abort below arrives promptly instead of ten deadlines apart. See
  [data_providers.md](data_providers.md#timeout).
- **Invalid JSON, or a missing field**, is logged at error level and that step
  falls back to `(0.0, false)`. A single bad answer does not end the run: a
  script with an occasional hiccup should not kill a long one.
- **Ten consecutive failures end the run**, with a non-zero exit and the last
  error as the message:

  ```text
  Error: reward script failed 10 times in a row, last error: invalid JSON
  reward response 'not json': expected ident at line 1 column 2.
  ```

  This mirrors the data provider's threshold. Without it, a script that died on
  the first step let training run to completion on rewards it never produced,
  exiting zero and leaving a checkpoint that looked legitimate — the worst kind
  of failure for a decision engine, because nothing about it looks wrong.

  Any successful answer resets the count, so the ten have to be consecutive.
- Stderr is forwarded to the log at error level, tagged with the script path, so
  `print(..., file=sys.stderr)` reaches your terminal and obeys `--log-level`.
  See [data_providers.md](data_providers.md#failures).

## Notes

- Python must be on `PATH`; see
  [data_providers.md](data_providers.md#notes) for the candidate order and the
  Windows Store-alias caveat.
- The script may hold state between steps in ordinary local variables, since the
  process is no longer restarted per call.
- Scripts are executed as given. Only point `--reward-script` at a script you
  trust, particularly when the path comes from a config file rather than the
  command line.
