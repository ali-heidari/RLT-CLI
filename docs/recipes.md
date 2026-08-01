# Recipes

Five infrastructure decisions, each with the three things you need to try it: a
**dataset shape**, a **reward function**, and the **incumbent rule** to measure
against.

These are starting points, not solutions. The reward function is where your
judgement goes, and nobody else's version of "good" will match yours.

> **Before you invest a day in one of these.** On the `aixker-rlt` revision this
> CLI pins, a trained policy decides no better than chance — see
> [found-issues.md](found-issues.md) issue 8. The shapes and reward functions
> below are the durable part and are worth writing down now; expect `eval` to
> report the heuristic winning until that is fixed. Nothing here has been
> demonstrated to beat its baseline.

Every recipe follows the method in [tutorial.md](tutorial.md). Read that first —
it explains normalisation, holding data back, and how to read the report.

---

## 1. Autoscaling

**Decide:** scale out, hold, or scale in.

**Why it suits a policy.** Threshold autoscaling is famously bad at two things:
it flaps when load oscillates around the threshold, and it reacts to the past
rather than the shape of the curve. Both are visible in a window of history,
which is exactly what a feature vector is.

| Column | Feature | Ceiling to divide by |
| --- | --- | --- |
| 1–3 | CPU now, 1 min ago, 5 min ago | 100% |
| 4 | Request rate | Peak capacity |
| 5 | Queue depth | Queue capacity |
| 6 | Current replica count | Max replicas |
| 7 | Seconds since the last scaling action | Cooldown period |
| 8 | Time of day | 24 |

Columns 1–3 are the trick: three points give the model the *trend*, which is the
information a single threshold throws away.

```python
def score(features, action):
    cpu, cpu_1m, cpu_5m, rate, queue, replicas, since_action, _hour = features
    trend = cpu - cpu_5m

    if action == 0:                       # scale out
        if cpu > 0.7 or (cpu > 0.5 and trend > 0.15):
            return 1.0, True              # ahead of the curve
        return -0.6 - replicas, False     # paying for idle capacity
    if action == 1:                       # hold
        if 0.3 < cpu < 0.7:
            return 1.0, True
        return -0.8, False                # should have acted
    # action == 2: scale in
    if cpu < 0.3 and trend < 0 and since_action > 1.0:
        return 0.8, True                  # calm, cooled down, shrink
    if since_action < 1.0:
        return -1.0, False                # this is flapping
    return -0.5, False
```

**Note the `-0.6 - replicas`**: scaling out when you did not need to costs more
when you are already large. A flat penalty teaches the model that over-scaling
is equally bad at 3 replicas and 30.

**Baseline:** `--baseline static:1` (never scale — surprisingly hard to beat on
cost) and a threshold script mirroring your current HPA.

**Better looks like:** the same p99 with fewer replica-hours, or fewer scaling
actions for the same p99. Count actions in the per-action breakdown.

---

## 2. Load balancer weighting

**Decide:** which of N backends gets the request.

| Column | Feature | Ceiling |
| --- | --- | --- |
| 1–N | Each backend's in-flight requests | Its connection limit |
| N+1–2N | Each backend's recent p99 | SLO |
| 2N+1 | Request size or cost estimate | Largest expected |

Set `output_number = N`. This is the one recipe where the action space grows
with your fleet, so keep N small — under about 8 — or the model spends its
capacity learning which backends exist.

```python
def score(features, action, backends=4):
    inflight = features[:backends]
    latency = features[backends:backends * 2]

    chosen_load = inflight[action]
    chosen_p99 = latency[action]

    if chosen_p99 > 1.0:
        return -1.0, False                       # sent into an SLO breach
    if chosen_load >= 0.95:
        return -0.8, False                       # sent into a full queue
    # Reward picking the genuinely least-loaded backend, not just a tolerable one.
    return 1.0 - (chosen_load - min(inflight)), chosen_load <= min(inflight) + 0.1
```

**Baseline:** `--baseline round-robin` — this is the one case where round-robin
is a serious incumbent rather than a strawman, because it is what most load
balancers actually do.

**Better looks like:** lower p99 at the same throughput. Least-connections is a
strong baseline; beating it needs the latency columns to carry real signal.

---

## 3. Cache admission

**Decide:** cache this object, or do not.

**Why it suits a policy.** Admission is where cache hit rates are won, and LRU
admits everything by default — one scan of large cold objects evicts your
working set.

| Column | Feature | Ceiling |
| --- | --- | --- |
| 1 | Object size | Largest cacheable |
| 2 | Times seen in the last hour | Cap at, say, 20 |
| 3 | Seconds since last seen | Window length |
| 4 | Cost to recompute or refetch | Slowest expected |
| 5 | Current cache utilisation | Capacity |
| 6 | Current hit rate | 1.0 |

Set `output_number = 2`.

```python
def score(features, action):
    size, frequency, recency, cost, utilisation, _hit_rate = features
    value = frequency * cost                     # worth keeping
    price = size * utilisation                   # what it displaces

    if action == 1:                              # admit
        if value > price:
            return 1.0, True
        return -(price - value), False           # evicted something better
    # action == 0: reject
    if value > price:
        return -0.8, False                       # a hit you gave away
    return 0.4, True                             # correct, but unexciting
```

**Rejecting correctly earns less than admitting correctly** on purpose. A policy
that rejects everything has a perfect record against a symmetric reward function
and an empty cache.

**Baseline:** `--baseline static:1` — admit everything, which is what plain LRU
does.

**Better looks like:** a higher hit rate at the same cache size. Replay a
production trace and compare.

---

## 4. Queue prioritisation

**Decide:** run this job now, defer it, or drop it.

| Column | Feature | Ceiling |
| --- | --- | --- |
| 1 | Estimated runtime | Longest expected |
| 2 | Time already waited | SLA |
| 3 | Declared priority | Highest tier |
| 4 | Retry count | Retry limit |
| 5 | Queue depth | Capacity |
| 6 | Free worker capacity | Total |
| 7 | Deadline proximity | SLA |

```python
def score(features, action):
    runtime, waited, priority, retries, depth, capacity, deadline = features
    urgent = deadline > 0.8 or waited > 0.9

    if action == 0:                              # run now
        if capacity < 0.1:
            return -1.0, False                   # nowhere to run it
        if urgent or priority > 0.7:
            return 1.0, True
        if depth > 0.8:
            return -0.4, False                   # jumped the queue under pressure
        return 0.5, True
    if action == 1:                              # defer
        if urgent:
            return -1.0, False                   # missed the deadline
        return 0.6, True
    # action == 2: drop
    if retries > 0.8 and priority < 0.3:
        return 0.3, True                         # genuinely not worth it
    return -1.0, False                           # you threw away someone's work
```

**Dropping is punished hard except in one narrow case.** Left symmetric, a
policy learns that an empty queue scores well.

**Baseline:** `--baseline static:0` — FIFO, run everything in order.

**Better looks like:** fewer SLA breaches at the same throughput, especially at
the tail. Check that the drop action stays rare in the breakdown.

---

## 5. Retry budgets

**Decide:** retry now, retry after a delay, or give up.

**Why it suits a policy.** Retry storms are a classic self-inflicted outage:
every client retrying a struggling dependency is what converts a slow service
into a dead one. The right answer depends on *why* it failed, which a fixed
backoff cannot see.

| Column | Feature | Ceiling |
| --- | --- | --- |
| 1 | Attempts so far | Max attempts |
| 2 | Time since first attempt | Timeout budget |
| 3 | Dependency error rate | 1.0 |
| 4 | Dependency p99 | SLO |
| 5 | Was it a timeout or an error code | 0 or 1 |
| 6 | Retry budget remaining | Budget |
| 7 | Caller's remaining deadline | Original |

```python
def score(features, action):
    attempts, elapsed, error_rate, p99, was_timeout, budget, deadline = features
    struggling = error_rate > 0.3 or p99 > 1.0

    if deadline < 0.1:                           # nothing left to spend
        return (0.5, True) if action == 2 else (-1.0, False)

    if action == 0:                              # retry immediately
        if struggling:
            return -1.0, False                   # this is how storms start
        if was_timeout > 0.5:
            return -0.5, False                   # it was slow, not broken
        return 0.8, True                         # a blip; try again
    if action == 1:                              # retry after backoff
        if struggling and budget > 0.2:
            return 1.0, True                     # right call under pressure
        return 0.2, True                         # safe, slightly slow
    # action == 2: give up
    if budget < 0.1 or attempts > 0.8:
        return 0.6, True
    return -0.7, False                           # gave up with budget to spare
```

**Baseline:** a fixed exponential-backoff script — `--baseline ./backoff.py`
returning action 1 whenever attempts remain.

**Better looks like:** fewer total retries during a dependency incident, without
more caller-visible failures. That trade-off is the whole point, and a single
mean reward hides it — look at the per-action counts.

---

## Common mistakes

**Rewarding the outcome you can measure instead of the one you want.** Latency
is easy to log; "the customer was served" is what matters. If they diverge, the
model will optimise the one you wrote.

**Symmetric rewards for asymmetric mistakes.** Scaling out unnecessarily costs
money; scaling in unnecessarily costs an outage. Those are not the same number.

**Features the model cannot act on.** A timestamp is not a feature unless
behaviour genuinely varies by hour. Every useless column is capacity spent
learning to ignore it.

**Scoring on training data.** Always hold a slice back, and prefer a *later*
time window rather than a random split — production drifts, and a random split
quietly lets the model see the future.

**Believing a small margin.** A few hundredths on a few hundred rows is noise.

## Next

- [tutorial.md](tutorial.md) — one of these worked through end to end
- [eval.md](eval.md) — baselines and the report
- [reward_factory.md](reward_factory.md) — the reward script protocol
