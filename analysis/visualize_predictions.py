import os, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
images_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/imagesTs")
pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")
out_dir = os.path.join(base, "results/visualizations")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
    mapping = json.load(f)
test_categories = {m["case_name"]: m["category"] for m in mapping if m["split"] == "test"}

def dice(pred_mask, label_mask):
    intersection = np.sum(pred_mask & label_mask)
    denom = np.sum(pred_mask) + np.sum(label_mask)
    if denom == 0:
        return 1.0
    return 2 * intersection / denom

def make_overlay(case_name):
    img = np.array(Image.open(os.path.join(images_dir, f"{case_name}_0000.png")))
    pred = np.array(Image.open(os.path.join(pred_dir, f"{case_name}.png")))
    label = np.array(Image.open(os.path.join(labels_dir, f"{case_name}.png")))

    rgb = np.stack([img, img, img], axis=-1).astype(np.float64)

    # Ground truth outline: green for liver, yellow for mass
    from scipy import ndimage
    for val, color in [(1, [0, 255, 0]), (2, [255, 255, 0])]:
        mask = label >= val if val == 1 else label == val
        if not np.any(mask):
            continue
        border = ndimage.binary_dilation(mask, iterations=2) ^ mask
        rgb[border] = color

    # Prediction: red tint for liver, blue tint for mass
    pred_liver = (pred >= 1)
    pred_mass = (pred == 2)
    rgb[pred_liver, 0] = np.clip(rgb[pred_liver, 0] * 0.7 + 50, 0, 255)
    rgb[pred_mass, 2] = np.clip(rgb[pred_mass, 2] * 0.7 + 100, 0, 255)

    cat = test_categories.get(case_name, "?")
    mass_dice = dice(pred == 2, label == 2) if np.any(label == 2) else None
    liver_dice = dice(pred >= 1, label >= 1)

    result = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(result)
    text = f"{case_name} ({cat}) | Liver: {liver_dice:.3f}"
    if mass_dice is not None:
        text += f" | Mass: {mass_dice:.3f}"
    draw.text((10, 10), text, fill=(255, 255, 255))

    result.save(os.path.join(out_dir, f"{case_name}.png"))
    print(f"Saved {case_name} ({cat}) - Liver: {liver_dice:.3f}, Mass: {mass_dice}")

# Missed masses
for c in ["liver_0628", "liver_0639", "liver_0702"]:
    make_overlay(c)

# Sample good and bad from each category
for f_name in sorted(os.listdir(pred_dir)):
    if not f_name.endswith(".png"):
        continue
    case_name = f_name.replace(".png", "")
    cat = test_categories.get(case_name)
    if cat in ("Benign", "Malignant"):
        label = np.array(Image.open(os.path.join(labels_dir, f_name)))
        pred = np.array(Image.open(os.path.join(pred_dir, f_name)))
        if np.any(label == 2):
            d = dice(pred == 2, label == 2)
            # Save best and worst from each category
            if d > 0.85 or d < 0.15:
                make_overlay(case_name)
