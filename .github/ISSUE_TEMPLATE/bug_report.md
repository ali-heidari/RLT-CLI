---
name: Bug report
about: Something behaved differently from how it is documented
title: ''
labels: bug
assignees: ''
---

## What happened

<!-- Include the command you ran and its output. The settings block printed at
the start of a run names where every value came from, which usually answers the
first three questions anyone would ask — please paste it. -->

```console
$ rlt ...
```

## What you expected

## Environment

- `rlt --version`:
- OS:
- `rustc --version` (if you built it yourself):
- `python3 --version` (if a Python script is involved):

## Configuration

<!-- Your Config.toml, or the flags you passed. Redact paths if you need to. -->

```toml
```

## Anything else

<!-- If a model is involved, `rlt inspect` output helps: it reports the
architecture and flags weights that have diverged.

Note that a trained policy currently deciding no better than chance is a known
upstream problem, not a bug in this CLI — see docs/found-issues.md issue 8
before filing that one. -->
