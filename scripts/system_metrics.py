#!/usr/bin/env python3
"""
Sample system metrics provider for RLT-CLI.

This script outputs system metrics as JSON array or comma-separated floats.
Useful for training RL models that respond to system state.
"""

import json
import sys


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
    try:
        import psutil
    except ImportError:
        # Fallback without psutil
        return [0.0] * 8

    metrics = []

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    metrics.append(cpu_percent)

    # Memory usage
    memory = psutil.virtual_memory()
    metrics.append(memory.percent)

    # Disk usage
    disk = psutil.disk_usage("/")
    metrics.append(disk.percent)

    # Load average
    load_avg = psutil.getloadavg()
    metrics.extend(load_avg)

    # Number of processes
    process_count = len(psutil.pids())
    metrics.append(float(process_count))

    # Uptime in days
    boot_time = psutil.boot_time()
    import time

    current_time = time.time()
    uptime_days = (current_time - boot_time) / (24 * 3600)
    metrics.append(uptime_days)

    return metrics


def main():
    try:
        metrics = get_system_metrics()
        metrics.insert(0, 0.0)  # Prepend a dummy feature for compatibility
        metrics.insert(0, 0.0)  # Prepend a dummy feature for compatibility
        metrics.insert(0, 0.0)  # Prepend a dummy feature for compatibility
        metrics.insert(0, 0.0)  # Prepend a dummy feature for compatibility
        # Output as JSON array
        print(json.dumps(metrics))
    except Exception as e:
        print(json.dumps([0.0] * 12), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
