"""Measure intensity contrast between mass and surrounding liver parenchyma,
and correlate contrast with segmentation performance.

For each test case with a mass, computes:
  - Mean pixel intensity inside the mass (label == 2)
  - Mean pixel intensity in surrounding liver (label == 1)
  - Contrast ratio: abs(mass - liver) / liver
  - Mass Dice score from the 625-image model predictions

Reports:
  - Per-case details for benign and malignant
  - Summary statistics by pathology type
  - Mann-Whitney U test for contrast difference
  - Pearson correlation between contrast and Dice
  - Dice performance binned by contrast level

Usage:
    python analysis/intensity_contrast.py
"""

import json
import os
import numpy as np
from PIL import Image
from scipy import stats


def dice_score(pred_mask, label_mask):
    intersection = np.sum(pred_mask & label_mask)
    denom = np.sum(pred_mask) + np.sum(label_mask)
    if denom == 0:
        return 1.0
    return 2 * intersection / denom


def main():
    base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
    images_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/imagesTs")
    labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
    pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")

    with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
        mapping = json.load(f)

    test_categories = {m["case_name"]: m["category"] for m in mapping if m["split"] == "test"}

    benign_results = []
    malignant_results = []

    for case_name, category in sorted(test_categories.items()):
        if category == "Normal":
            continue

        img = np.array(Image.open(os.path.join(images_dir, f"{case_name}_0000.png")), dtype=np.float64)
        label = np.array(Image.open(os.path.join(labels_dir, f"{case_name}.png")))
        pred = np.array(Image.open(os.path.join(pred_dir, f"{case_name}.png")))

        mass_mask = label == 2
        liver_mask = label == 1  # liver parenchyma only (excludes mass)

        if not np.any(mass_mask) or not np.any(liver_mask):
            continue

        mass_mean = np.mean(img[mass_mask])
        liver_mean = np.mean(img[liver_mask])
        contrast = abs(mass_mean - liver_mean) / liver_mean if liver_mean > 0 else 0
        d = dice_score(pred == 2, label == 2)

        result = {
            "case": case_name,
            "mass_mean": mass_mean,
            "liver_mean": liver_mean,
            "contrast": contrast,
            "dice": d,
        }

        if category == "Benign":
            benign_results.append(result)
        elif category == "Malignant":
            malignant_results.append(result)

    # Per-case details
    print("BENIGN CASES")
    print(f"{'Case':<14} {'Mass':>8} {'Liver':>8} {'Contrast':>10} {'Dice':>8}")
    print("-" * 52)
    for r in benign_results:
        print(f"{r['case']:<14} {r['mass_mean']:>8.1f} {r['liver_mean']:>8.1f} {r['contrast']:>10.3f} {r['dice']:>8.3f}")

    print(f"\nMALIGNANT CASES")
    print(f"{'Case':<14} {'Mass':>8} {'Liver':>8} {'Contrast':>10} {'Dice':>8}")
    print("-" * 52)
    for r in malignant_results:
        print(f"{r['case']:<14} {r['mass_mean']:>8.1f} {r['liver_mean']:>8.1f} {r['contrast']:>10.3f} {r['dice']:>8.3f}")

    # Summary statistics
    b_contrasts = [r["contrast"] for r in benign_results]
    m_contrasts = [r["contrast"] for r in malignant_results]

    print(f"\n{'='*55}")
    print(f"SUMMARY: CONTRAST BY PATHOLOGY TYPE")
    print(f"{'='*55}")
    print(f"{'':>20} {'Benign':>10} {'Malignant':>10}")
    print(f"{'-'*42}")
    print(f"{'n':>20} {len(b_contrasts):>10} {len(m_contrasts):>10}")
    print(f"{'Mean contrast':>20} {np.mean(b_contrasts):>10.3f} {np.mean(m_contrasts):>10.3f}")
    print(f"{'Median contrast':>20} {np.median(b_contrasts):>10.3f} {np.median(m_contrasts):>10.3f}")
    print(f"{'Std contrast':>20} {np.std(b_contrasts):>10.3f} {np.std(m_contrasts):>10.3f}")
    print(f"{'Mean mass intensity':>20} {np.mean([r['mass_mean'] for r in benign_results]):>10.1f} {np.mean([r['mass_mean'] for r in malignant_results]):>10.1f}")
    print(f"{'Mean liver intensity':>20} {np.mean([r['liver_mean'] for r in benign_results]):>10.1f} {np.mean([r['liver_mean'] for r in malignant_results]):>10.1f}")

    # Statistical test: is benign contrast lower than malignant?
    u_stat, p_value = stats.mannwhitneyu(b_contrasts, m_contrasts, alternative="less")
    print(f"\nMann-Whitney U (benign contrast < malignant contrast):")
    print(f"  U = {u_stat:.1f}, p = {p_value:.4f}")

    # Correlation between contrast and Dice
    all_contrasts = b_contrasts + m_contrasts
    all_dices = [r["dice"] for r in benign_results] + [r["dice"] for r in malignant_results]
    r, p = stats.pearsonr(all_contrasts, all_dices)

    print(f"\n{'='*55}")
    print(f"CONTRAST VS DICE PERFORMANCE")
    print(f"{'='*55}")
    print(f"Pearson correlation: r = {r:.3f}, p = {p:.6f}")

    # Dice binned by contrast level
    all_rows = [(r["contrast"], r["dice"], "B") for r in benign_results] + \
               [(r["contrast"], r["dice"], "M") for r in malignant_results]

    print(f"\n{'Contrast bin':>15} | {'Mean Dice':>9} | {'n':>3} | {'Benign':>6} | {'Malignant':>9}")
    print("-" * 55)
    for lo, hi, label in [(0, 0.1, "0-10%"), (0.1, 0.2, "10-20%"), (0.2, 0.3, "20-30%"), (0.3, 1.0, "30%+")]:
        bin_rows = [(c, d, t) for c, d, t in all_rows if lo <= c < hi]
        if bin_rows:
            mean_d = np.mean([d for _, d, _ in bin_rows])
            n_b = sum(1 for _, _, t in bin_rows if t == "B")
            n_m = sum(1 for _, _, t in bin_rows if t == "M")
            print(f"{label:>15} | {mean_d:>9.3f} | {len(bin_rows):>3} | {n_b:>6} | {n_m:>9}")


if __name__ == "__main__":
    main()
