"""
Extract timing and peak epoch data from nnU-Net training logs.

Scans training log files for epoch timestamps, best EMA pseudo Dice
checkpoints, and computes wall-clock training time.

Usage:
    python extract_timing.py \
        --results-dir $nnUNet_results

Expects the standard nnU-Net results directory structure:
    Dataset*/nnUNetTrainer150__nnUNetPlans__2d/fold_0/training_log_*.txt
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path


SIZES = [25, 50, 100, 200, 300, 400, 500, 625]
DATASET_IDS = {25: 2, 50: 3, 100: 4, 200: 5, 300: 6, 400: 7, 500: 8, 625: 1}

TS_PAT = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)')
EPOCH_PAT = re.compile(r'Epoch (\d+)$')
BEST_PAT = re.compile(r'Yayy! New best EMA pseudo Dice: ([\d.]+)')


def parse_timestamp(ts_str):
    return datetime.strptime(ts_str[:26], '%Y-%m-%d %H:%M:%S.%f')


def find_training_log(results_dir, dataset_id, size):
    """Find the training log for a given dataset size."""
    if size == 625:
        dataset_name = f"Dataset{dataset_id:03d}_LiverUS"
    else:
        dataset_name = f"Dataset{dataset_id:03d}_LiverUS_{size}"

    fold_dir = os.path.join(
        results_dir, dataset_name,
        "nnUNetTrainer150__nnUNetPlans__2d", "fold_0"
    )

    if not os.path.exists(fold_dir):
        return None

    # Find the most recent training log
    logs = sorted([
        f for f in os.listdir(fold_dir)
        if f.startswith("training_log_") and f.endswith(".txt")
    ])

    if not logs:
        return None

    return os.path.join(fold_dir, logs[-1])


def parse_log(log_path):
    """Parse a training log and return timing + peak epoch data."""
    with open(log_path) as f:
        lines = f.readlines()

    train_start = None
    epoch_times = {}
    current_epoch = -1
    best_epoch = 0
    best_dice = 0.0

    for line in lines:
        line = line.strip()

        # Track epoch start times
        m_epoch = EPOCH_PAT.search(line)
        if m_epoch:
            m_ts = TS_PAT.match(line)
            if m_ts:
                ep = int(m_epoch.group(1))
                ts = parse_timestamp(m_ts.group(1))
                epoch_times[ep] = ts
                current_epoch = ep
                if train_start is None:
                    train_start = ts

        # Track best checkpoints
        m_best = BEST_PAT.search(line)
        if m_best:
            dice = float(m_best.group(1))
            if dice > best_dice:
                best_dice = dice
                best_epoch = current_epoch

    # Training end = "Training done" timestamp or last epoch start
    train_end = None
    for line in reversed(lines):
        line = line.strip()
        if 'Training done' in line:
            m_ts = TS_PAT.match(line)
            if m_ts:
                train_end = parse_timestamp(m_ts.group(1))
                break

    if train_end is None and epoch_times:
        train_end = max(epoch_times.values())

    # Time to peak = end of best epoch (approximated by start of next epoch)
    peak_end = epoch_times.get(best_epoch + 1, epoch_times.get(best_epoch))

    total_seconds = (train_end - train_start).total_seconds() if train_end else 0
    peak_seconds = (peak_end - train_start).total_seconds() if peak_end else 0

    return {
        "best_epoch": best_epoch,
        "best_ema_dice": best_dice,
        "time_to_peak_min": peak_seconds / 60,
        "total_time_min": total_seconds / 60,
        "train_start": train_start,
        "train_end": train_end,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Path to nnUNet_results")
    args = parser.parse_args()

    print(f"{'Size':>6} | {'Peak epoch':>10} | {'Best EMA Dice':>13} | "
          f"{'Time to peak':>12} | {'Total time':>10}")
    print("-" * 70)

    for size in SIZES:
        dataset_id = DATASET_IDS[size]
        log_path = find_training_log(args.results_dir, dataset_id, size)

        if log_path is None:
            print(f"{size:>6} | {'not found':>10} |")
            continue

        data = parse_log(log_path)
        print(f"{size:>6} | {data['best_epoch']:>10} | "
              f"{data['best_ema_dice']:>13.4f} | "
              f"{data['time_to_peak_min']:>10.0f} min | "
              f"{data['total_time_min']:>8.0f} min")


if __name__ == "__main__":
    main()
