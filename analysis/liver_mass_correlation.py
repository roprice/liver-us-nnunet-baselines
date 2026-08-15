"""Compute correlation between liver Dice and mass Dice at the per-image level.

Tests whether liver segmentation quality predicts mass detection quality.

Usage:
    python analysis/liver_mass_correlation.py
"""

import os, json
import numpy as np
from PIL import Image
from scipy import stats

base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")

with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
    mapping = json.load(f)
test_map = {m["case_name"]: m["category"] for m in mapping if m["split"] == "test"}

def dice(p, l):
    i = np.sum(p & l)
    d = np.sum(p) + np.sum(l)
    return 2*i/d if d > 0 else 1.0

liver_dices, mass_dices = [], []
for f_name in sorted(os.listdir(pred_dir)):
    if not f_name.endswith(".png"):
        continue
    case = f_name.replace(".png", "")
    cat = test_map.get(case)
    if cat != "Malignant":
        continue
    label = np.array(Image.open(os.path.join(labels_dir, f_name)))
    pred = np.array(Image.open(os.path.join(pred_dir, f_name)))
    if not np.any(label == 2):
        continue
    liver_dices.append(dice(pred >= 1, label >= 1))
    mass_dices.append(dice(pred == 2, label == 2))

r, p = stats.pearsonr(liver_dices, mass_dices)
print(f"Liver vs Mass Dice (malignant, n={len(liver_dices)}): r={r:.3f}, p={p:.4f}")
