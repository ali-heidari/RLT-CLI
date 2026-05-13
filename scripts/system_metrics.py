#!/usr/bin/env python3
"""
Sample system metrics provider for RLT-CLI.

This script outputs system metrics as a JSON array.
Useful for training RL models that respond to system state.
"""

import json
import os
import shutil
import sys
import time


def _read_proc_stat():
    with open("/proc/stat", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("cpu "):
                parts = line.split()
                values = [float(p) for p in parts[1:]]
                return values
    raise RuntimeError("Could not read /proc/stat")


def _cpu_percent(interval=0.1):
    first = _read_proc_stat()
    time.sleep(interval)
    second = _read_proc_stat()

    idle1 = first[3] + first[4]
    idle2 = second[3] + second[4]
    total1 = sum(first)
    total2 = sum(second)

    total_delta = total2 - total1
    idle_delta = idle2 - idle1
    if total_delta <= 0:
        return 0.0
    usage = (1.0 - idle_delta / total_delta) * 100.0
    return max(0.0, min(100.0, usage))


def _memory_percent():
    meminfo = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            key, value = line.split(":", 1)
            meminfo[key.strip()] = float(value.split()[0])

    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable") or meminfo.get("MemFree")
    if not total or not available:
        return 0.0
    used = total - available
    return max(0.0, min(100.0, used / total * 100.0))


def _disk_percent(path="/"):
    usage = shutil.disk_usage(path)
    if usage.total == 0:
        return 0.0
    return max(0.0, min(100.0, usage.used / usage.total * 100.0))


def _process_count():
    try:
        proc_entries = os.listdir("/proc")
        return float(sum(1 for name in proc_entries if name.isdigit()))
    except Exception:
        return 0.0


def _uptime_days():
    with open("/proc/uptime", "r", encoding="utf-8") as f:
        uptime_seconds = float(f.readline().split()[0])
    return uptime_seconds / 86400.0


def get_system_metrics():
    """
    Collect system metrics and return as a list of floats.

    Metrics returned (in order):
    1. CPU usage percentage (0-100)
    2. Memory usage percentage (0-100)
    3. Disk usage percentage (0-100)
    4. Load average (1-minute)
    5. Load average (5-minute)
    6. Load average (15-minute)
    7. Number of processes
    8. Uptime in days
    """
    cpu_percent = _cpu_percent(interval=0.1)
    memory_percent = _memory_percent()
    disk_percent = _disk_percent("/")
    load_avg = os.getloadavg()
    process_count = _process_count()
    uptime_days = _uptime_days()

    return [
        cpu_percent,
        memory_percent,
        disk_percent,
        float(load_avg[0]),
        float(load_avg[1]),
        float(load_avg[2]),
        process_count,
        uptime_days,
    ]


def main():
    try:
        metrics = get_system_metrics()
        print(json.dumps(metrics))
    except Exception as e:
        print(json.dumps([0.0] * 8), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
