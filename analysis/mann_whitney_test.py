"""Mann-Whitney U test: benign vs malignant mass Dice at 625 training images.

Tests whether the difference in mass segmentation performance between
benign and malignant cases is statistically significant.

Usage:
    python analysis/mann_whitney_test.py
"""

import json
import os
import numpy as np
from scipy import stats
from PIL import Image


def dice_score(pred_mask, label_mask):
    intersection = np.sum(pred_mask & label_mask)
    denom = np.sum(pred_mask) + np.sum(label_mask)
    if denom == 0:
        return 1.0
    return 2 * intersection / denom


def main():
    base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
    pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")
    labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")

    with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
        mapping = json.load(f)

    test_categories = {m["case_name"]: m["category"] for m in mapping if m["split"] == "test"}

    benign_scores = []
    malignant_scores = []

    for f_name in sorted(os.listdir(pred_dir)):
        if not f_name.endswith(".png"):
            continue
        case_name = f_name.replace(".png", "")
        category = test_categories.get(case_name, "Unknown")

        pred = np.array(Image.open(os.path.join(pred_dir, f_name)))
        label = np.array(Image.open(os.path.join(labels_dir, f_name)))

        if not np.any(label == 2):
            continue

        d = dice_score(pred == 2, label == 2)
        if category == "Benign":
            benign_scores.append(d)
        elif category == "Malignant":
            malignant_scores.append(d)

    benign = np.array(benign_scores)
    malignant = np.array(malignant_scores)

    # Mann-Whitney U test (two-sided)
    u_stat, p_value = stats.mannwhitneyu(malignant, benign, alternative="two-sided")

    print("Benign vs Malignant Mass Dice (625 training images)")
    print("=" * 55)
    print(f"Benign:    n={len(benign):>3}, mean={np.mean(benign):.3f}, "
          f"std={np.std(benign):.3f}, median={np.median(benign):.3f}")
    print(f"Malignant: n={len(malignant):>3}, mean={np.mean(malignant):.3f}, "
          f"std={np.std(malignant):.3f}, median={np.median(malignant):.3f}")
    print()
    print(f"Mann-Whitney U statistic: {u_stat:.1f}")
    print(f"p-value: {p_value:.2e}")
    print()
    if p_value < 0.001:
        print("Result: p < 0.001 — the difference is statistically significant.")
    elif p_value < 0.05:
        print(f"Result: p = {p_value:.4f} — the difference is statistically significant.")
    else:
        print(f"Result: p = {p_value:.4f} — the difference is not statistically significant.")


if __name__ == "__main__":
    main()
