# Evaluation

`rlt eval` answers the question a trained checkpoint otherwise leaves open: **is
this model any good?** — and, more usefully, **is it better than the heuristic
we already run?**

```bash
rlt eval --dataset ./holdout.csv --model-name my-model.json \
  --reward-script ./scripts/reward_script.py --baseline static:1
```

```text
Evaluated 400 sample(s) from ./holdout.csv

Policy: my-model.json
  mean reward   0.6234
  success rate  87.2%
  actions       0: 120 (30.0%)  mean reward 0.5100  success 71.7%
                1: 190 (47.5%)  mean reward 0.7200  success 96.8%
                2: 90 (22.5%)   mean reward 0.5800  success 84.4%

Baseline: static:1
  mean reward   0.4100
  success rate  62.0%
  actions       1: 400 (100.0%)  mean reward 0.4100  success 62.0%

Difference: +0.2134 mean reward, +25.2pp success rate
The policy beats the baseline.
```

## What it needs

| Setting | Flag | Required |
| --- | --- | --- |
| Held-out dataset | `--dataset PATH` | Yes |
| Checkpoint | `--model-name NAME` | Yes |
| Reward script | `--reward-script PATH` | **Yes** |
| Baseline | `--baseline SPEC` | No |

The reward script is required, and the command refuses to run without one. It
is what defines "good": without it every reward is `0.0`, and the report would
be a page of zeros presented as an answer. See
[reward_factory.md](reward_factory.md) for the protocol.

Use a dataset the model was **not** trained on. Nothing enforces that — the CLI
cannot know what you trained on — but scoring a model against its training data
measures memory, not usefulness.

## Baselines

A number on its own is hard to act on. A number next to the rule you run today
is a decision.

| `--baseline` | Meaning |
| --- | --- |
| `static:N` | Always action `N`. The "is your AI even needed?" baseline, and the one skeptics ask for first |
| `round-robin` | Cycle through the actions in order. The dumb baseline |
| `random` | Uniform choice, seeded so two runs agree |
| `path/to/heuristic.py` | The rule your team already runs |

`static:N` and a script are the two that matter. `round-robin` and `random`
establish the floor: a policy that cannot beat them has learned nothing.

An action outside the model's action space is rejected up front, whether it came
from `static:N` or from a script — comparing against an action the model could
never have chosen is not a comparison.

### Heuristic script protocol

Same shape as the other Python integration points: started once, one JSON line
in, one JSON line out.

**Request:**

```json
{"features": [0.1, 0.2, 0.3]}
```

**Response:**

```json
{"action": 1}
```

```python
#!/usr/bin/env python3
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    cpu = request["features"][0]
    # Whatever rule you run today.
    print(json.dumps({"action": 1 if cpu > 0.8 else 0}), flush=True)
```

The script obeys `--script-timeout` like every other; see
[data_providers.md](data_providers.md#timeout).

## Both policies see the same rows

The policy and the baseline are scored **in a single pass, on identical rows**.
Running two separate evaluations would compare two different samples of the data
and report the difference as a result. Each row costs two reward-script calls,
one per policy.

## Machine-readable output

`--format json` emits one object, for a CI regression check or a plot:

```bash
rlt eval --dataset ./holdout.csv --model-name my-model.json \
  --reward-script ./reward.py --baseline static:1 --format json
```

```json
{
  "dataset": "./holdout.csv",
  "samples": 400,
  "policy": {
    "label": "my-model.json",
    "mean_reward": 0.6234,
    "success_rate": 0.872,
    "actions": { "0": { "count": 120, "successes": 86 } }
  },
  "baseline": { "label": "static:1", "mean_reward": 0.41, "success_rate": 0.62 },
  "reward_gain": 0.2134
}
```

`reward_gain` is the policy's mean reward minus the baseline's. Negative means
the baseline won; the run also logs a warning in that case, so it shows up in CI
output that only captures logs.

## Exit status

`eval` exits non-zero when it **could not evaluate** — a missing checkpoint, an
unreadable dataset, a dead reward script. A policy that loses to its baseline is
a successful evaluation with a disappointing result, so it exits `0`. Gate on
`reward_gain` from the JSON output if you want CI to fail on a regression.

## What it does not do yet

- No train/validation split (`--split`); bring your own held-out file.
- No confidence intervals. With a small held-out set, treat a small difference
  as noise.
- No per-row output. Use [`infer --output`](usage.md#infer) for that.
