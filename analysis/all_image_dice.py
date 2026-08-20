import os, json
import numpy as np
from PIL import Image

base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")

with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
    mapping = json.load(f)
test_map = {m["case_name"]: m for m in mapping if m["split"] == "test"}

def dice(pred_mask, label_mask):
    intersection = np.sum(pred_mask & label_mask)
    denom = np.sum(pred_mask) + np.sum(label_mask)
    if denom == 0:
        return 1.0
    return 2 * intersection / denom

rows = []
for f_name in sorted(os.listdir(pred_dir)):
    if not f_name.endswith(".png"):
        continue
    case_name = f_name.replace(".png", "")
    info = test_map.get(case_name)
    if not info:
        continue
    cat = info["category"]
    orig = f"{cat}/image/{info['original_file']}"
    pred = np.array(Image.open(os.path.join(pred_dir, f_name)))
    label = np.array(Image.open(os.path.join(labels_dir, f_name)))
    liver_d = dice(pred >= 1, label >= 1)
    mass_d = dice(pred == 2, label == 2) if np.any(label == 2) else None
    rows.append((case_name, cat, liver_d, mass_d, orig))

print("=== ALL IMAGES SORTED BY MASS DICE (low to high) ===")
print(f"{'Case':>12} | {'Type':>10} | {'Liver':>6} | {'Mass':>6} | Original")
print("-" * 80)
mass_rows = [(c, t, l, m, o) for c, t, l, m, o in rows if m is not None]
mass_rows.sort(key=lambda x: x[3])
for case, cat, liver_d, mass_d, orig in mass_rows:
    print(f"{case:>12} | {cat:>10} | {liver_d:>6.3f} | {mass_d:>6.3f} | {orig}")

print(f"\nTotal mass cases: {len(mass_rows)}")
print(f"Mean mass Dice: {np.mean([r[3] for r in mass_rows]):.4f}")
print(f"Median mass Dice: {np.median([r[3] for r in mass_rows]):.4f}")
