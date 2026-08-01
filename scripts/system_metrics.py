#!/usr/bin/env python3
"""
Sample system metrics provider for Aixker-RLT CLI.

`rlt` starts this script once and keeps it running. For every sample it
writes one JSON request line to stdin (currently the empty object `{}`) and
expects exactly one line back on stdout holding the feature vector, either as
a JSON array or as comma-separated floats:

    [0.12, 0.48, 0.03, ...]

`flush=True` is required: Python block-buffers stdout when it is a pipe, so
without it the CLI would wait forever for a response sitting in that buffer.
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
    Collect system metrics and return as a list of floats in 0.0-1.0 range.

    All metrics are normalized to [0.0, 1.0] range:
    1. CPU usage (0-100% → 0.0-1.0)
    2. Memory usage (0-100% → 0.0-1.0)
    3. Network traffic (0-300 → 0.0-1.0)
    4. Power consumption (0-400 → 0.0-1.0)
    5. Num executed instructions (0-8000 → 0.0-1.0)
    6. Execution time (0-80 → 0.0-1.0)
    7. Energy efficiency (already 0.0-1.0)
    8. Task type encoded (0-2 → 0.0-1.0)
    9. Task priority encoded (0-2 → 0.0-1.0)
    10. Task status encoded (1-2 → 0.0-1.0)
    11. VM ID hash (already 0.0-1.0)
    12. Timestamp normalized (already 0.0-1.0)
    """
    cpu_percent = _cpu_percent(interval=0.1)
    memory_percent = _memory_percent()
    disk_percent = _disk_percent("/")
    load_avg = os.getloadavg()
    process_count = _process_count()
    uptime_days = _uptime_days()

    # Normalize features to match training data ranges first, then scale to 0.0-1.0
    network_traffic = min(300.0, load_avg[0] * 50.0)  # Load * 50, capped at 300
    power_consumption = min(400.0, cpu_percent * 3.0 + memory_percent * 2.0)  # CPU*3 + Mem*2, capped at 400
    num_instructions = min(8000.0, process_count * 30.0)  # Process count * 30, capped at 8000
    execution_time = min(80.0, uptime_days * 10.0)  # Uptime * 10, capped at 80
    energy_efficiency = max(0.0, min(1.0, (100.0 - cpu_percent - memory_percent) / 100.0))  # Inverse of resource usage

    # Encode categorical features (using simple rules)
    task_type = 0 if load_avg[0] < 1.0 else (1 if load_avg[0] < 3.0 else 2)  # 0=network, 1=io, 2=compute
    task_priority = 0 if cpu_percent < 30.0 else (1 if cpu_percent < 70.0 else 2)  # 0=low, 1=medium, 2=high
    task_status = 1 if cpu_percent < 90.0 else 2  # 1=completed, 2=failed (based on CPU stress)

    # VM ID hash (simple hash of hostname)
    import hashlib
    vm_id_hash = int(hashlib.md5(os.uname().nodename.encode()).hexdigest()[:8], 16) / 2**32

    # Timestamp normalized to 0-1 range
    timestamp_norm = (time.time() % 86400) / 86400.0  # Daily cycle normalized

    # Normalize all values directly to 0.0-1.0 range
    return [
        cpu_percent / 100.0,           # 1. CPU usage % → 0.0-1.0
        memory_percent / 100.0,        # 2. Memory usage % → 0.0-1.0
        min(1.0, load_avg[0] / 6.0),   # 3. Network traffic (load avg 0-6 → 0.0-1.0)
        min(1.0, (cpu_percent * 3.0 + memory_percent * 2.0) / 400.0),  # 4. Power consumption → 0.0-1.0
        min(1.0, process_count / 8000.0),  # 5. Num executed instructions → 0.0-1.0
        min(1.0, uptime_days / 80.0),  # 6. Execution time (days) → 0.0-1.0
        energy_efficiency,              # 7. Energy efficiency (already 0.0-1.0)
        float(task_type) / 2.0,        # 8. Task type encoded → 0.0-1.0
        float(task_priority) / 2.0,    # 9. Task priority encoded → 0.0-1.0
        (float(task_status) - 1.0),    # 10. Task status encoded → 0.0-1.0
        vm_id_hash,                     # 11. VM ID hash (already 0.0-1.0)
        timestamp_norm,                 # 12. Timestamp normalized (already 0.0-1.0)
    ]


def main():
    # One response per request line, for as long as `rlt` keeps asking.
    for _request in sys.stdin:
        try:
            metrics = get_system_metrics()
        except Exception as exc:
            # Exit rather than emit fabricated zeros: the CLI reports the dead
            # worker, which is honest, where a zero vector would be trained on.
            print(f"failed to collect system metrics: {exc}", file=sys.stderr)
            sys.exit(1)

        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
