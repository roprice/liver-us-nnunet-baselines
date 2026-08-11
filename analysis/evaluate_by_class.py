"""Evaluate mass Dice broken down by Benign vs Malignant."""

import json
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

def main():
    base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
    predictions_base = os.path.join(base, "results/nnUNet_results")
    labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")

    with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
        mapping = json.load(f)

    test_categories = {m["case_name"]: m["category"] for m in mapping if m["split"] == "test"}

    print(f"{'Size':>6} | {'Benign Mass':>12} | {'B Std':>7} | {'N ben':>5} | "
          f"{'Malign Mass':>12} | {'M Std':>7} | {'N mal':>5} | "
          f"{'Normal':>8}")
    print("-" * 85)

    for size in SIZES:
        pred_dir = os.path.join(predictions_base, f"predictions_{size}")
        benign_scores, malignant_scores, normal_count = [], [], 0

        for f_name in sorted(os.listdir(pred_dir)):
            if not f_name.endswith(".png"):
                continue
            case_name = f_name.replace(".png", "")
            category = test_categories.get(case_name, "Unknown")

            pred = np.array(Image.open(os.path.join(pred_dir, f_name)))
            label = np.array(Image.open(os.path.join(labels_dir, f_name)))

            if category == "Normal":
                normal_count += 1
            elif category == "Benign" and np.any(label == 2):
                benign_scores.append(dice_score(pred == 2, label == 2))
            elif category == "Malignant" and np.any(label == 2):
                malignant_scores.append(dice_score(pred == 2, label == 2))

        b_mean = np.mean(benign_scores) if benign_scores else 0
        b_std = np.std(benign_scores) if benign_scores else 0
        m_mean = np.mean(malignant_scores) if malignant_scores else 0
        m_std = np.std(malignant_scores) if malignant_scores else 0

        print(f"{size:>6} | {b_mean:>12.4f} | {b_std:>7.4f} | {len(benign_scores):>5} | "
              f"{m_mean:>12.4f} | {m_std:>7.4f} | {len(malignant_scores):>5} | "
              f"{normal_count:>8}")

if __name__ == "__main__":
    main()
