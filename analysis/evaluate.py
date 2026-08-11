"""
Compute per-class Dice scores on held-out test predictions.

Usage:
    python evaluate.py \
        --predictions-dir predictions \
        --labels-dir $nnUNet_raw/Dataset001_LiverUS/labelsTs
"""

import argparse
import os
import numpy as np
from PIL import Image


def dice_score(pred_mask, label_mask):
    intersection = np.sum(pred_mask & label_mask)
    denom = np.sum(pred_mask) + np.sum(label_mask)
    if denom == 0:
        return 1.0
    return 2 * intersection / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    args = parser.parse_args()

    liver_scores = []
    mass_scores = []

    for f in sorted(os.listdir(args.predictions_dir)):
        if not f.endswith(".png"):
            continue

        pred = np.array(Image.open(os.path.join(args.predictions_dir, f)))
        label_path = os.path.join(args.labels_dir, f)
        if not os.path.exists(label_path):
            print(f"Warning: no ground truth for {f}, skipping")
            continue
        label = np.array(Image.open(label_path))

        # Liver: classes 1 and 2 both count as liver region
        liver_scores.append(dice_score(pred >= 1, label >= 1))

        # Mass: class 2 only, skip if no mass in ground truth
        if np.any(label == 2):
            mass_scores.append(dice_score(pred == 2, label == 2))

    print(f"Test set results (n={len(liver_scores)}):")
    print(f"  Liver Dice: {np.mean(liver_scores):.4f} +/- {np.std(liver_scores):.4f}")
    print(f"  Mass Dice:  {np.mean(mass_scores):.4f} +/- {np.std(mass_scores):.4f} "
          f"(n={len(mass_scores)} cases with mass)")


if __name__ == "__main__":
    main()
