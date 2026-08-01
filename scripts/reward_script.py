#!/usr/bin/env python3
"""
Sample reward script for RLT-CLI.

RLT-CLI starts this script once and keeps it running. For every environment
step it writes one JSON line to stdin:

    {"features": [0.1, 0.2, ...], "action": 1}

and expects exactly one JSON line back on stdout:

    {"reward": 1.0, "success": true}

`flush=True` is required: Python block-buffers stdout when it is a pipe, so
without it the CLI would wait forever for a response sitting in that buffer.
"""

import json
import sys


def evaluate_reward(features, action):
    # Example reward logic: prefer action 1 when the first feature is positive.
    baseline = features[0] if features else 0.0
    if action == 1 and baseline > 0.0:
        reward = 1.0
    elif action == 1:
        reward = -0.5
    else:
        reward = 0.1 if baseline <= 0.0 else -0.1

    return reward, reward > 0.0


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        features = request.get("features", [])
        action = int(request.get("action", 0))

        reward, success = evaluate_reward(features, action)
        print(json.dumps({"reward": reward, "success": success}), flush=True)


if __name__ == "__main__":
    main()
