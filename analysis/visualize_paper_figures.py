"""Generate prediction overlay figures with unified color palette.

Colors match the learning curves chart:
  - Blue (#2a78d6) for liver
  - Orange (#eb6834) for malignant mass
  - Gray (#898781) for benign mass

Ground truth shown as outlines, predictions as semi-transparent fills.

Usage:
    python analysis/visualize_paper_figures.py
"""

import os, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

BLUE = np.array([42, 120, 214])    # liver
ORANGE = np.array([235, 104, 52])  # malignant mass
GRAY = np.array([137, 135, 129])   # benign mass

def dice(p, l):
    i = np.sum(p & l)
    d = np.sum(p) + np.sum(l)
    return 2*i/d if d > 0 else 1.0

def make_overlay(case_name, category, images_dir, labels_dir, pred_dir, out_dir):
    img = np.array(Image.open(os.path.join(images_dir, f"{case_name}_0000.png")))
    pred = np.array(Image.open(os.path.join(pred_dir, f"{case_name}.png")))
    label = np.array(Image.open(os.path.join(labels_dir, f"{case_name}.png")))

    rgb = np.stack([img, img, img], axis=-1).astype(np.float64)

    mass_color = ORANGE if category == "Malignant" else GRAY

    # Prediction fills (semi-transparent)
    pred_liver = (pred >= 1) & (pred != 2)
    pred_mass = (pred == 2)

    alpha_liver = 0.25
    alpha_mass = 0.35

    for c in range(3):
        rgb[pred_liver, c] = rgb[pred_liver, c] * (1 - alpha_liver) + BLUE[c] * alpha_liver
        rgb[pred_mass, c] = rgb[pred_mass, c] * (1 - alpha_mass) + mass_color[c] * alpha_mass

    # Ground truth outlines
    liver_mask = label >= 1
    mass_mask = label == 2

    if np.any(liver_mask):
        border = ndimage.binary_dilation(liver_mask, iterations=2) ^ liver_mask
        rgb[border] = BLUE

    if np.any(mass_mask):
        border = ndimage.binary_dilation(mass_mask, iterations=2) ^ mass_mask
        rgb[border] = mass_color

    liver_d = dice(pred >= 1, label >= 1)
    mass_d = dice(pred == 2, label == 2) if np.any(label == 2) else None

    result = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(result)

    label_text = f"{case_name} ({category}) | Liver: {liver_d:.3f}"
    if mass_d is not None:
        label_text += f" | Mass: {mass_d:.3f}"
    draw.text((10, 10), label_text, fill=(255, 255, 255))

    out_path = os.path.join(out_dir, f"{case_name}_paper.png")
    result.save(out_path)
    print(f"Saved {out_path}")

def main():
    base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
    images_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/imagesTs")
    labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
    pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")
    out_dir = os.path.join(base, "results/visualizations")

    figures = [
        ("liver_0689", "Malignant"),
        ("liver_0633", "Benign"),
        ("liver_0645", "Benign"),
    ]

    for case_name, category in figures:
        make_overlay(case_name, category, images_dir, labels_dir, pred_dir, out_dir)

if __name__ == "__main__":
    main()
