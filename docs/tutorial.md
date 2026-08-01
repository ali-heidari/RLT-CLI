# Tutorial: from logs to a decision you can check

One complete story, end to end: shape a dataset, write a reward function, train
a policy, find out whether it beats the rule you already run, and export it.

Everything here runs on data you generate locally — no dataset ships with this
repository, and the generator is short enough to read.

> **Read this first.** On the `aixker-rlt` revision this CLI pins, a trained
> policy decides no better than chance. This tutorial ends with `eval` reporting
> that honestly, because that is what it is for and because a tutorial that
> pretended otherwise would waste your afternoon. The cause is upstream and
> documented in [found-issues.md](found-issues.md) issue 8. Everything else here
> — the data shaping, the reward function, the comparison method — is the part
> you will still need when it is fixed.

## The problem

A service takes requests. For each one it can:

| Action | Meaning |
| --- | --- |
| `0` | Serve locally |
| `1` | Redirect to a peer |
| `2` | Shed — return 503 immediately |

Serving locally is best when there is capacity. Redirecting costs a round trip
but avoids a queue. Shedding is the least bad option when the alternative is
timing out everyone already waiting.

Today you decide this with a CPU threshold. The question is whether a learned
policy does better.

## Step 1: shape the data

The model takes a row of numbers per decision. Every value must be numeric and
every row the same width — no identifiers, no timestamps, no categorical text.
See [data_providers.md](data_providers.md#columns-must-all-be-numeric).

Six features, each normalised to roughly `[0, 1]`:

| Column | Feature | Where it comes from |
| --- | --- | --- |
| 1 | CPU utilisation | Your metrics store |
| 2 | Memory utilisation | Your metrics store |
| 3 | Queue depth / capacity | Load balancer or app metric |
| 4 | p99 latency / SLO | Your metrics store |
| 5 | Error rate | Your metrics store |
| 6 | Peer load | The peer you would redirect to |

Normalising matters. A raw p99 in milliseconds sits in the hundreds while error
rate sits near zero, and the model has to learn to undo that scale difference
before it can learn anything useful. Divide each by a sensible ceiling — latency
by your SLO, queue depth by capacity — so "1.0" means "at the limit".

### Generating a sample

Save this as `make_data.py` and run it. It produces data with a rule worth
learning: load builds, and the right action escalates with it.

```python
#!/usr/bin/env python3
"""Synthesise request-routing snapshots with a learnable rule."""
import random

random.seed(7)  # Deterministic, so two runs are comparable.


def row():
    cpu = random.random()
    # Everything else correlates with load, with noise on top.
    noise = lambda: random.uniform(-0.08, 0.08)
    return [
        cpu,
        min(1.0, max(0.0, cpu * 0.8 + 0.1 + noise())),   # memory
        min(1.0, max(0.0, cpu ** 2 + noise())),          # queue depth
        min(1.0, max(0.0, cpu ** 1.5 + noise())),        # p99 / SLO
        min(1.0, max(0.0, (cpu - 0.7) * 2 + noise())),   # error rate
        random.random(),                                  # peer load: independent
    ]


for name, count in [("train.csv", 4000), ("holdout.csv", 800)]:
    with open(name, "w") as handle:
        for _ in range(count):
            handle.write(",".join(f"{value:.4f}" for value in row()) + "\n")
    print(f"wrote {name}")
```

```bash
python3 make_data.py
```

Two files, and the held-out set is **not** the training set. Scoring a model on
data it trained on measures memory, not usefulness.

### Coming from real logs

The same shape, from whatever you already have. A Prometheus range query, a
`SELECT` over your request log, an export from your APM — anything that gives
one row per decision point. The only rules are: all numeric, fixed width,
normalised, and one row per moment you would have made this decision.

## Step 2: write the reward

The reward function is where your judgement lives. It answers "was that the
right call?", and the model optimises exactly what you write here — including
the parts you did not think about.

Save as `reward.py`:

```python
#!/usr/bin/env python3
"""What a good routing decision looks like."""
import json
import sys


def score(features, action):
    cpu, _memory, queue, latency, errors, peer = features
    pressure = max(cpu, queue, latency)

    if action == 0:                      # serve locally
        if pressure < 0.6:
            return 1.0, True             # exactly right
        if pressure < 0.8:
            return -0.2, False           # risky
        return -1.0, False               # this is how queues collapse
    if action == 1:                      # redirect to a peer
        if pressure >= 0.6 and peer < 0.7:
            return 0.8, True             # good: we had somewhere to send it
        if peer >= 0.7:
            return -0.5, False           # sent it into another fire
        return -0.3, False               # unnecessary round trip
    # action == 2: shed
    if pressure >= 0.9 or errors > 0.5:
        return 0.5, True                 # least bad
    return -1.0, False                   # threw away a request we could serve


for line in sys.stdin:
    request = json.loads(line)
    reward, success = score(request["features"], request["action"])
    print(json.dumps({"reward": reward, "success": success}), flush=True)
```

Three things worth copying from this:

**Redirect scores lower than serving locally, even when correct.** It is right,
but it costs a round trip. If both paid `1.0` the model would have no reason to
prefer the cheaper one.

**Shedding is rewarded, but only at the top.** Reward it too readily and you
train a policy that sheds under mild load — technically optimal against a
careless reward function, useless in production.

**`success` is not "reward was positive".** It is your own definition of an
acceptable outcome, and `eval` reports it separately as a success rate. Here a
risky-but-served request is a negative reward and also not a success.

The script must loop over stdin and flush — it is started once and kept running.
See [reward_factory.md](reward_factory.md).

## Step 3: write the rule you run today

This is the number to beat. Save as `heuristic.py`:

```python
#!/usr/bin/env python3
"""The CPU threshold currently in production."""
import json
import sys

for line in sys.stdin:
    cpu = json.loads(line)["features"][0]
    print(json.dumps({"action": 0 if cpu < 0.75 else 1}), flush=True)
```

It only looks at CPU, and never sheds. That is realistic: most thresholds in
production are one number someone picked in an incident.

## Step 4: configure

`Config.toml`:

```toml
dataset = "./train.csv"
reward_script = "./reward.py"

input_number = 6       # six features per row
output_number = 3      # serve, redirect, shed

model_name = "router.json"
hidden_layers = 16
batch_size = 32
total_batches = 120    # 121 * 32 = 3872 samples, under the 4000 we generated
log_interval = 1000
```

`input_number` must match your row width exactly; a mismatch is reported before
training starts rather than a thousand rows in.

`total_batches × batch_size` must not exceed your row count, or the source runs
dry and the run ends early.

## Step 5: train

```bash
rlt train
```

No flags — the config has them. Because the config *names a script* rather than
you typing its path, `rlt` asks before running it:

```text
Config.toml asks to execute:
  ./reward.py
These run as you, with your privileges.
Run them? [y/N] y
```

Answer `y`, or pass `--allow-scripts` to answer in advance — which you will need
in CI, where there is no terminal to ask.

The settings block prints where every value came from, so there is no guessing:

```text
Starting training with the following settings:
  dataset: ./train.csv (Config.toml)
  model name: router.json (Config.toml)
  batch size: 32 (Config.toml)
  total batches: 120 (Config.toml)
  reward script: ./reward.py (Config.toml)
  script timeout: 30s (default)
```

Ctrl-C is safe: the run winds down, keeps the last checkpoint, and exits `0`.

## Step 6: the step that matters

```bash
rlt eval --dataset ./holdout.csv --baseline ./heuristic.py
```

This is real output from the steps above, not an illustration. **Your figures
will differ, and so will two runs of your own** — inference samples from the
action distribution rather than taking the best action, so the policy's numbers
move between evaluations even on an identical checkpoint and dataset. That is
the second half of [found-issues.md](found-issues.md) issue 8. The baseline,
being deterministic, reproduces exactly:

```text
Evaluated 800 sample(s) from ./holdout.csv

Policy: router.json
  mean reward   -0.1829
  success rate  32.5%
  actions       0: 268 (33.5%)  mean reward 0.4030  success 62.7%
                1: 263 (32.9%)  mean reward -0.0639  success 27.0%
                2: 269 (33.6%)  mean reward -0.8829  success 7.8%

Baseline: ./heuristic.py
  mean reward   0.7059
  success rate  79.9%
  actions       0: 615 (76.9%)  mean reward 0.7854  success 82.1%
                1: 185 (23.1%)  mean reward 0.4416  success 72.4%

Difference: -0.8888 mean reward, -47.4pp success rate
The baseline beats the policy.
```

**Read the per-action breakdown before the headline number.** Compare the two
distributions above:

- The policy split its choices `33.5% / 32.9% / 33.6%`. Three actions at a third
  each is the signature of a policy that is not looking at its input.
- The baseline split `76.9% / 23.1%`, which is roughly the share of rows where
  CPU is under `0.75`. Its distribution **reflects the data**.

That contrast is the tell, and it is more informative than the mean reward. Note
also that the policy's action 0 scored `+0.40` — serving locally is right most of
the time, so choosing it at random still earns something. Its action 2 scored
`-0.88` at a 7.8% success rate: it is shedding requests it could have served,
which is the specific behaviour the reward function punishes hardest.

That is what you are seeing here, and it is [found-issues.md](found-issues.md)
issue 8 rather than a mistake in your reward function. To confirm it is not your
setup, check the model itself:

```bash
rlt inspect
```

Weights reaching `1e5` or higher, or a snapshot showing logits like
`[0.0, -50.0, -50.0]`, mean the run diverged rather than converged.

### If your policy did win

Then read the breakdown anyway, because a good mean reward can hide a bad
policy:

- **One action at ~100%** — it found a constant that scores acceptably. Your
  reward function probably does not punish the wrong call harshly enough.
- **Success rate high, mean reward low** — it is scraping acceptable outcomes
  expensively. Look at whether your cheap action is rewarded enough.
- **A tiny margin** — on 800 samples, a few hundredths is noise. Get more
  held-out data before believing it.

## Step 7: gate it in CI

```bash
rlt eval --dataset ./holdout.csv --baseline ./heuristic.py \
  --format json --output result.json

python3 -c "
import json, sys
gain = json.load(open('result.json'))['reward_gain']
print(f'reward gain: {gain:+.4f}')
sys.exit(0 if gain > 0 else 1)
"
```

Use `--output` rather than redirecting stdout. The library still prints
`EMPTY INPUT` directly to stdout when a data source ends
([found-issues.md](found-issues.md) issue 5), so a redirect would capture that
line too and the file would not parse. `--output` writes only the report.

`eval` itself exits `0` whenever it managed to evaluate — a policy losing is a
successful evaluation with a disappointing result, not a failure. Gate on
`reward_gain` when you want CI to fail on a regression.

## Step 8: export

```bash
rlt export --output ./exported/router.json
```

The checkpoint is copied verbatim; it is already JSON and nothing is converted.
`safetensors` and `onnx` are on the [roadmap](roadmap.md).

## What you have

```text
train.csv  holdout.csv     data, shaped and split
reward.py                  your objective, written down
heuristic.py               the incumbent, as a comparable policy
models/router.json...      the trained checkpoint
result.json                the comparison, machine-readable
```

The parts that survive the upstream fix are the parts that took judgement: the
feature set, the reward function, and the decision to hold data back. The
training command was one word.

## Next

- [eval.md](eval.md) — baselines and the report in full
- [recipes.md](recipes.md) — the same method applied to autoscaling, cache
  admission, queue prioritisation and retry budgets
- [usage.md](usage.md) — every command and flag
