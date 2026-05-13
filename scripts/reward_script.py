#!/usr/bin/env python3
"""
Sample reward script for RLT-CLI.

This script reads a JSON object from stdin with `features` and `action`.
It writes a JSON response with `reward` and `success`.
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
    request = json.load(sys.stdin)
    features = request.get("features", [])
    action = int(request.get("action", 0))

    reward, success = evaluate_reward(features, action)
    print(json.dumps({"reward": reward, "success": success}))


if __name__ == "__main__":
    main()
