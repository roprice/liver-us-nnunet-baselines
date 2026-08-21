"""Generate prediction overlay visualizations for all test cases.

Uses unified color palette:
  - Blue (#2a78d6) for liver (outline and fill)
  - Orange (#eb6834) for mass (outline and fill), regardless of pathology type

The model predicts a single mass class and does not distinguish
between benign and malignant. All mass predictions use the same color.

Usage:
    python analysis/visualize_all_predictions.py
"""

import os, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

BLUE = np.array([42, 120, 214])
ORANGE = np.array([235, 104, 52])


def dice(p, l):
    i = np.sum(p & l)
    d = np.sum(p) + np.sum(l)
    return 2 * i / d if d > 0 else 1.0


def make_overlay(case_name, category, images_dir, labels_dir, pred_dir, out_dir):
    img = np.array(Image.open(os.path.join(images_dir, f"{case_name}_0000.png")))
    pred = np.array(Image.open(os.path.join(pred_dir, f"{case_name}.png")))
    label = np.array(Image.open(os.path.join(labels_dir, f"{case_name}.png")))

    rgb = np.stack([img, img, img], axis=-1).astype(np.float64)

    # Prediction fills
    pred_liver = (pred >= 1) & (pred != 2)
    pred_mass = (pred == 2)

    for c in range(3):
        rgb[pred_liver, c] = rgb[pred_liver, c] * 0.75 + BLUE[c] * 0.25
        rgb[pred_mass, c] = rgb[pred_mass, c] * 0.65 + ORANGE[c] * 0.35

    # Ground truth outlines
    liver_mask = label >= 1
    mass_mask = label == 2

    if np.any(liver_mask):
        border = ndimage.binary_dilation(liver_mask, iterations=2) ^ liver_mask
        rgb[border] = BLUE

    if np.any(mass_mask):
        border = ndimage.binary_dilation(mass_mask, iterations=2) ^ mass_mask
        rgb[border] = ORANGE

    liver_d = dice(pred >= 1, label >= 1)
    mass_d = dice(pred == 2, label == 2) if np.any(label == 2) else None

    result = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(result)

    text = f"{case_name} ({category}) | Liver: {liver_d:.3f}"
    if mass_d is not None:
        text += f" | Mass: {mass_d:.3f}"
    draw.text((10, 10), text, fill=(255, 255, 255))

    result.save(os.path.join(out_dir, f"{case_name}.png"))
    return liver_d, mass_d


def main():
    base = os.path.expanduser("~/Projects/liver-us-nnunet-baselines")
    images_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/imagesTs")
    labels_dir = os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/labelsTs")
    pred_dir = os.path.join(base, "results/nnUNet_results/predictions_625")
    out_dir = os.path.join(base, "results/visualizations")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(base, "nnUNet_raw/Dataset001_LiverUS/case_mapping.json")) as f:
        mapping = json.load(f)
    test_map = {m["case_name"]: m["category"] for m in mapping if m["split"] == "test"}

    count = 0
    for f_name in sorted(os.listdir(pred_dir)):
        if not f_name.endswith(".png"):
            continue
        case_name = f_name.replace(".png", "")
        category = test_map.get(case_name, "Unknown")
        liver_d, mass_d = make_overlay(
            case_name, category, images_dir, labels_dir, pred_dir, out_dir
        )
        mass_str = f"{mass_d:.3f}" if mass_d is not None else "n/a"
        print(f"  {case_name} ({category}) | Liver: {liver_d:.3f} | Mass: {mass_str}")
        count += 1

    print(f"\nGenerated {count} overlays in {out_dir}")

    # Also generate paper-specific copies
    paper_dir = os.path.join(base, "paper")
    paper_cases = ["liver_0689", "liver_0633", "liver_0645"]
    for case_name in paper_cases:
        src = os.path.join(out_dir, f"{case_name}.png")
        dst = os.path.join(paper_dir, f"{case_name}_paper.png")
        if os.path.exists(src):
            Image.open(src).save(dst)
            print(f"  Copied {case_name} to paper/")


if __name__ == "__main__":
    main()
