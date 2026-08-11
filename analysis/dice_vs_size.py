import os, json
import numpy as np
from PIL import Image

base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")

with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
    mapping = json.load(f)
test_categories = {m["case_name"]: m["category"] for m in mapping if m["split"] == "test"}

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
    cat = test_categories.get(case_name)
    if cat not in ("Benign", "Malignant"):
        continue

    label = np.array(Image.open(os.path.join(labels_dir, f_name)))
    pred = np.array(Image.open(os.path.join(pred_dir, f_name)))

    if not np.any(label == 2):
        continue

    mass_pixels = np.sum(label == 2)
    total_pixels = label.size
    mass_fraction = mass_pixels / total_pixels * 100
    d = dice(pred == 2, label == 2)
    rows.append((case_name, cat, mass_pixels, mass_fraction, d))

rows.sort(key=lambda x: x[2])

print(f"{'Case':>12} | {'Type':>10} | {'Mass px':>8} | {'% image':>7} | {'Dice':>6}")
print("-" * 55)
for case, cat, px, frac, d in rows:
    print(f"{case:>12} | {cat:>10} | {px:>8} | {frac:>6.2f}% | {d:>6.3f}")

# Summary stats
from scipy import stats
sizes = [r[2] for r in rows]
dices = [r[4] for r in rows]
r, p = stats.pearsonr(sizes, dices)
print(f"\nPearson correlation: r={r:.3f}, p={p:.6f}")

# By size quartile
quartiles = np.percentile(sizes, [25, 50, 75])
print(f"\nBy size quartile (pixel count cutoffs: {quartiles[0]:.0f}, {quartiles[1]:.0f}, {quartiles[2]:.0f}):")
for i, label in enumerate(["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"]):
    if i == 0:
        q_rows = [r for r in rows if r[2] <= quartiles[0]]
    elif i == 3:
        q_rows = [r for r in rows if r[2] > quartiles[2]]
    else:
        q_rows = [r for r in rows if quartiles[i-1] < r[2] <= quartiles[i]]
    q_dices = [r[4] for r in q_rows]
    n_benign = sum(1 for r in q_rows if r[1] == "Benign")
    n_malig = sum(1 for r in q_rows if r[1] == "Malignant")
    print(f"  {label}: mean Dice {np.mean(q_dices):.3f}, std {np.std(q_dices):.3f}, "
          f"n={len(q_rows)} ({n_benign}B/{n_malig}M)")
