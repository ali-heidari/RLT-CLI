# Tutorial: from logs to a decision you can check

One complete story, end to end: shape a dataset, write a reward function, train
a policy, find out whether it beats the rule you already run, and export it.

Everything here runs on data you generate locally — no dataset ships with this
repository, and the generator is short enough to read.

> **Read this first.** The first training run here is small on purpose, and
> `eval` reports that it loses to the heuristic. Step 6 shows how to read *why*
> from the report, and Step 7 trains on more data until it wins. A tutorial that
> only showed the winning run would teach you less.

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
log_interval = 10      # one line of training statistics every 10 batches
seed = 1               # so your figures match the ones below
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
  backend: cpu (default)
  reward script: ./reward.py (Config.toml)
  dry run: false (default)
  seed: 1 (Config.toml)
  learning rate: 0.05 (default)
  max grad norm: 1 (default)
  entropy coef: 0.1 (default)
  exploration: epsilon 0.3 -> 0.05, temperature 1.5 -> 1 over 500 batches (default)
  reward shaping: false (default)
  stats log: every 10 batches (Config.toml)
  csv header row: false (default)
  csv delimiter: ',' (default)
  script timeout: 30s (default)
```

Every 10 batches the library logs what training is doing. The two values to
watch are `policy`, the average chance of each action, and `entropy`, how
spread out those chances are. For three actions, 1.0986 means "picks evenly";
it falls as the policy commits:

```text
batch 60: ... | policy=[...] entropy=0.9780 | ...
batch 120: ... | policy=[...] entropy=0.7404 | ...
```

Ctrl-C is safe: the run winds down, keeps the last checkpoint, and exits `0`.

## Step 6: the step that matters

```bash
rlt eval --dataset ./holdout.csv --baseline ./heuristic.py
```

This is real output from the steps above, not an illustration. With
`seed = 1` and the same data you should see the same figures; without a seed,
the policy's numbers move between runs, because `eval` samples its actions. The
baseline is deterministic and always reproduces:

```text
Evaluated 800 sample(s) from ./holdout.csv

Policy: router.json
  mean reward   0.4067
  success rate  62.5%
  actions       0: 735 (91.9%)  mean reward 0.4422  success 64.5%
                1: 56 (7.0%)  mean reward 0.1411  success 44.6%
                2: 9 (1.1%)  mean reward -0.8333  success 11.1%

Baseline: ./heuristic.py
  mean reward   0.7059
  success rate  79.9%
  actions       0: 615 (76.9%)  mean reward 0.7854  success 82.1%
                1: 185 (23.1%)  mean reward 0.4416  success 72.4%

Difference: -0.2991 mean reward, -17.4pp success rate
The baseline beats the policy.
```

**Read the per-action breakdown before the headline number.** Compare the two
distributions above:

- The policy serves locally `91.9%` of the time. Serving locally is right for
  most rows, so it learned the common case first.
- The baseline redirects `23.1%` of the time, roughly the share of rows where
  CPU is over `0.75`. The policy redirects only `7.0%`, so it has not yet
  learned *when* to redirect.

That contrast is the tell, and it says more than the mean reward. The policy is
not broken; it has not seen enough examples of the rarer case. Two other checks
confirm a run is healthy rather than diverged:

```bash
rlt inspect
```

Weights reaching `1e5` or higher, or a snapshot showing logits like
`[0.0, -50.0, -50.0]`, mean the run diverged rather than converged. And the
settings block above shows exploration scheduled over 500 batches, while this
run stopped at 120 — it ended while still taking random actions a quarter of
the time.

## Step 7: train on more

120 batches is about all that 4,000 rows can feed. Generate ten times the
training data by changing one line in `make_data.py`:

```python
for name, count in [("train.csv", 40000), ("holdout.csv", 800)]:
```

and train for ten times as long in `Config.toml`:

```toml
total_batches = 1200
```

Then run `python3 make_data.py`, `rlt train` and the same `eval` again. The
held-out rows change too, because the generator draws them after the larger
training set:

```text
Policy: router.json
  mean reward   0.7193
  success rate  83.0%
  actions       0: 530 (66.2%)  mean reward 0.8642  success 88.7%
                1: 270 (33.8%)  mean reward 0.4348  success 71.9%

Baseline: ./heuristic.py
  mean reward   0.6583
  success rate  76.4%
  actions       0: 598 (74.8%)  mean reward 0.7452  success 78.8%
                1: 202 (25.2%)  mean reward 0.4010  success 69.3%

Difference: +0.0610 mean reward, +6.6pp success rate
The policy beats the baseline.
```

Seeds 2 and 3 scored 84.6% and 82.8%, so this is not one lucky run. The policy
now redirects a third of the time, and each action scores better than the
baseline's version of it. Like the heuristic, it never sheds: under this reward,
shedding only pays at the very top of the load range.

### If your policy did win

Then read the breakdown anyway, because a good mean reward can hide a bad
policy:

- **One action at ~100%** — it found a constant that scores acceptably. Your
  reward function probably does not punish the wrong call harshly enough.
- **Success rate high, mean reward low** — it is scraping acceptable outcomes
  expensively. Look at whether your cheap action is rewarded enough.
- **A tiny margin** — on 800 samples, a few hundredths is noise. Get more
  held-out data before believing it.

## Step 8: gate it in CI

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

`--output` writes only the report. A stdout redirect works too: logs go to
stderr.

`eval` itself exits `0` whenever it managed to evaluate — a policy losing is a
successful evaluation with a disappointing result, not a failure. Gate on
`reward_gain` when you want CI to fail on a regression.

## Step 9: export

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

The parts that took judgement are the feature set, the reward function, and the
decision to hold data back. The training command was one word.

## Next

- [eval.md](eval.md) — baselines and the report in full
- [recipes.md](recipes.md) — the same method applied to autoscaling, cache
  admission, queue prioritisation and retry budgets
- [usage.md](usage.md) — every command and flag
