import os, numpy as np
from PIL import Image

base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")

def dice(p, l):
    i = np.sum(p & l)
    d = np.sum(p) + np.sum(l)
    return 2*i/d if d > 0 else 1.0

rows = []
for f in sorted(os.listdir(pred_dir)):
    if not f.endswith(".png"): continue
    label = np.array(Image.open(os.path.join(labels_dir, f)))
    pred = np.array(Image.open(os.path.join(pred_dir, f)))
    liver = np.sum(label >= 1)
    mass = np.sum(label == 2)
    if mass == 0 or liver == 0: continue
    rows.append((mass / liver, dice(pred == 2, label == 2)))

for threshold in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
    below = [d for r, d in rows if r < threshold]
    above = [d for r, d in rows if r >= threshold]
    if below and above:
        print(f"Threshold {threshold*100:5.1f}%: below mean {np.mean(below):.3f} (n={len(below)}), above mean {np.mean(above):.3f} (n={len(above)})")
