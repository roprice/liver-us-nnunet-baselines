"""
Evaluate data efficiency study results.

Computes liver and mass Dice scores for each training set size
and prints a summary table.

Usage:
    python evaluate_efficiency.py \
        --predictions-base $nnUNet_results \
        --labels-dir $nnUNet_raw/Dataset001_LiverUS/labelsTs
"""

import argparse
import os
import numpy as np
from PIL import Image


SIZES = [25, 50, 100, 200, 300, 400, 500, 625]


def dice_score(pred_mask, label_mask):
    intersection = np.sum(pred_mask & label_mask)
    denom = np.sum(pred_mask) + np.sum(label_mask)
    if denom == 0:
        return 1.0
    return 2 * intersection / denom


def evaluate_predictions(pred_dir, labels_dir):
    liver_scores = []
    mass_scores = []

    for f in sorted(os.listdir(pred_dir)):
        if not f.endswith(".png"):
            continue

        pred = np.array(Image.open(os.path.join(pred_dir, f)))
        label_path = os.path.join(labels_dir, f)
        if not os.path.exists(label_path):
            continue
        label = np.array(Image.open(label_path))

        liver_scores.append(dice_score(pred >= 1, label >= 1))

        if np.any(label == 2):
            mass_scores.append(dice_score(pred == 2, label == 2))

    return liver_scores, mass_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-base", required=True,
                        help="Directory containing predictions_25, predictions_50, etc.")
    parser.add_argument("--labels-dir", required=True,
                        help="Path to labelsTs from Dataset001")
    args = parser.parse_args()

    print(f"{'Size':>6} | {'Liver Dice':>12} | {'Liver Std':>10} | "
          f"{'Mass Dice':>11} | {'Mass Std':>9} | {'N mass':>6}")
    print("-" * 72)

    for size in SIZES:
        pred_dir = os.path.join(args.predictions_base, f"predictions_{size}")

        if not os.path.exists(pred_dir):
            print(f"{size:>6} | {'not found':>12} |")
            continue

        liver_scores, mass_scores = evaluate_predictions(pred_dir, args.labels_dir)

        if liver_scores:
            print(f"{size:>6} | {np.mean(liver_scores):>12.4f} | "
                  f"{np.std(liver_scores):>10.4f} | "
                  f"{np.mean(mass_scores):>11.4f} | "
                  f"{np.std(mass_scores):>9.4f} | "
                  f"{len(mass_scores):>6}")


if __name__ == "__main__":
    main()
