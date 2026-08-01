# Getting started with `rlt init`

`rlt init` scaffolds everything a first run needs, so getting to a trained
model is two commands rather than a documentation page.

```bash
rlt init
rlt train
```

## What it writes

| File | Purpose |
| --- | --- |
| `Config.toml` | Matches the generated data, so `train` needs no flags |
| `scripts/reward.py` | Defines what "good" means. **This is the file to edit** |
| `scripts/heuristic.py` | A plain threshold rule to measure the policy against |
| `data/train.csv` | 2400 generated rows |
| `data/holdout.csv` | 400 rows the model does not train on, for `eval` |

`init` refuses to overwrite anything that already exists. Pass `--force` if you
mean it. Scaffold somewhere other than the current directory with
`rlt init path/to/dir`.

## The generated data

Six numeric features per row. The first is a load figure in `[0, 1)`, and the
rule the reward script scores against is simply which third of that range it
falls in:

| Load | Best action |
| --- | --- |
| `< 0.33` | 0 |
| `< 0.66` | 1 |
| otherwise | 2 |

The other five features are derived from the load, except one that is pure
noise — something for the model to learn to ignore. Generation is deterministic,
so two runs of `init` produce identical files and comparisons between runs mean
something.

The rule is deliberately trivial. The point of a quickstart is to exercise the
workflow on data whose right answer you already know, so that anything the tool
reports can be checked by hand.

## Then evaluate it

```bash
rlt eval --dataset ./data/holdout.csv --baseline ./scripts/heuristic.py
```

The heuristic splits at the halfway point and so never chooses action 1, which
makes it wrong through the middle of the range. It scores **67%** — a real bar,
but one a model that has learned the rule should clear comfortably.

## What you will actually see

**The scaffolded policy does not beat the heuristic.** On the library revision
this CLI pins, it scores about **33%** — chance, for three actions:

```text
Policy: quickstart.json          mean reward 0.3325   success 33.2%
Baseline: ./scripts/heuristic.py mean reward 0.6700   success 67.0%

Difference: -0.3375 mean reward, -33.8pp success rate
The baseline beats the policy.
```

That is not a mistake in the scaffold, and training longer does not fix it —
30× the batches gives 32.5%. It is a limitation in the `aixker-rlt` library,
recorded in detail as [found-issues.md](found-issues.md) issue 8: learning is
far too weak for the hardcoded learning rate, and inference *samples* from the
action distribution rather than taking the best action, so even a well-ranked
model would be crippled at decision time.

It is documented here rather than hidden because a quickstart that quietly
showed a losing model would be worse, and because this is exactly the job
`eval` exists to do. A tool that tells you your model is no good is working
correctly; the alternative is shipping it.

Once the upstream issue is fixed, the same three commands become the real
workflow — train, check it beats the heuristic, ship the checkpoint.

## Making it yours

1. Replace `data/train.csv` with your own rows, all numeric, one sample per
   line. Set `input_number` to match the width.
2. Rewrite `scripts/reward.py` to score what you actually care about. This is
   the file that decides what the model optimises for.
3. Point `scripts/heuristic.py` at whatever rule you run today — that is the
   number worth beating.

See [usage.md](usage.md) for every flag, [reward_factory.md](reward_factory.md)
for the reward protocol, and [eval.md](eval.md) for the comparison in full.
