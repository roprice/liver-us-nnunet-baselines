import os, json
import numpy as np
from PIL import Image

base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")

with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
    mapping = json.load(f)

test_categories = {m["case_name"]: m["category"] for m in mapping if m["split"] == "test"}

print("Normal cases - false mass predictions at 625:")
for f_name in sorted(os.listdir(pred_dir)):
    if not f_name.endswith(".png"):
        continue
    case_name = f_name.replace(".png", "")
    if test_categories.get(case_name) != "Normal":
        continue
    pred = np.array(Image.open(os.path.join(pred_dir, f_name)))
    label = np.array(Image.open(os.path.join(labels_dir, f_name)))
    false_mass_pixels = np.sum(pred == 2)
    total_pixels = pred.size
    print(f"  {case_name}: {false_mass_pixels} mass pixels predicted ({false_mass_pixels/total_pixels*100:.3f}%)")

print("\nMass cases - does model ever predict NO mass?")
for f_name in sorted(os.listdir(pred_dir)):
    if not f_name.endswith(".png"):
        continue
    case_name = f_name.replace(".png", "")
    cat = test_categories.get(case_name)
    if cat not in ("Benign", "Malignant"):
        continue
    pred = np.array(Image.open(os.path.join(pred_dir, f_name)))
    label = np.array(Image.open(os.path.join(labels_dir, f_name)))
    if np.any(label == 2) and not np.any(pred == 2):
        print(f"  {case_name} ({cat}): mass in ground truth but model predicted NONE")
