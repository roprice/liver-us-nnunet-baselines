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
    d = dice(pred == 2, label == 2)
    rows.append((case_name, cat, mass_pixels, d))

sizes = [r[2] for r in rows]
quartiles = np.percentile(sizes, [25, 50, 75])
q_labels = ["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"]

print(f"{'Quartile':>15} | {'Benign Dice':>12} | {'n_B':>4} | {'Malign Dice':>12} | {'n_M':>4}")
print("-" * 60)
for i, ql in enumerate(q_labels):
    if i == 0:
        q_rows = [r for r in rows if r[2] <= quartiles[0]]
    elif i == 3:
        q_rows = [r for r in rows if r[2] > quartiles[2]]
    else:
        q_rows = [r for r in rows if quartiles[i-1] < r[2] <= quartiles[i]]
    b_dices = [r[3] for r in q_rows if r[1] == "Benign"]
    m_dices = [r[3] for r in q_rows if r[1] == "Malignant"]
    b_mean = f"{np.mean(b_dices):.3f}" if b_dices else "n/a"
    m_mean = f"{np.mean(m_dices):.3f}" if m_dices else "n/a"
    print(f"{ql:>15} | {b_mean:>12} | {len(b_dices):>4} | {m_mean:>12} | {len(m_dices):>4}")
