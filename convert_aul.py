"""
Convert AUL dataset to nnU-Net V2 format.

Downloads from: https://zenodo.org/records/7272660
Expected input structure:
    AUL/
        Benign/image/*.jpg, Benign/segmentation/{liver,mass,outline}/*.json
        Malignant/image/*.jpg, Malignant/segmentation/{liver,mass,outline}/*.json
        Normal/image/*.jpg, Normal/segmentation/{liver,outline}/*.json

Outputs nnU-Net V2 Dataset001_LiverUS with stratified train/test split.
Labels: 0=background, 1=liver, 2=mass

The AUL dataset provides outline polygons marking the ultrasound scan
sector boundary. Pixels outside the scan sector are non-diagnostic
(black padding, UI overlays). This script:
  1. Zeros out image pixels outside the outline polygon, so nnU-Net's
     automatic nonzero-mask normalization excludes non-scan-sector
     regions (including UI elements like the body position icon).
  2. Saves outline masks separately in outlinesTr/outlinesTs/ for
     potential use in future experiments (POCUS simulation, custom
     normalization).
"""

import os
import json
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


CATEGORIES = ["Benign", "Malignant", "Normal"]
RANDOM_SEED = 42
TEST_FRACTION = 0.15


def load_polygon(json_path):
    with open(json_path) as f:
        points = json.load(f)
    return [(p[0], p[1]) for p in points]


def render_binary_mask(polygon, image_size):
    """Render a binary mask (0/1) from a polygon."""
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=1)
    return np.array(mask, dtype=np.uint8)


def render_mask(polygons_by_label, image_size):
    """Render segmentation mask from polygons.

    Polygons are drawn in label order (1=liver first, 2=mass second)
    so that mass pixels overwrite liver pixels where they overlap.
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for label_val in sorted(polygons_by_label.keys()):
        poly = polygons_by_label[label_val]
        if poly is not None:
            draw.polygon(poly, fill=label_val)
    return np.array(mask, dtype=np.uint8)


def gather_cases(raw_data_dir):
    cases = []
    for category in CATEGORIES:
        img_dir = raw_data_dir / category / "image"
        liver_dir = raw_data_dir / category / "segmentation" / "liver"
        mass_dir = raw_data_dir / category / "segmentation" / "mass"
        outline_dir = raw_data_dir / category / "segmentation" / "outline"

        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = Path(img_file).stem
            cases.append({
                "category": category,
                "original_file": img_file,
                "image": img_dir / img_file,
                "liver_json": liver_dir / f"{stem}.json",
                "mass_json": mass_dir / f"{stem}.json",
                "outline_json": outline_dir / f"{stem}.json",
            })
    return cases


def main():
    parser = argparse.ArgumentParser(description="Convert AUL to nnU-Net format")
    parser.add_argument("--raw-data-dir", type=Path, required=True,
                        help="Path to AUL raw data (contains Benign/, Malignant/, Normal/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="nnU-Net raw dataset output path")
    args = parser.parse_args()

    cases = gather_cases(args.raw_data_dir)
    print(f"Total cases found: {len(cases)}")

    random.seed(RANDOM_SEED)
    by_cat = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c)

    train_cases, test_cases = [], []
    mapping = []

    for cat, cat_cases in sorted(by_cat.items()):
        random.shuffle(cat_cases)
        n_test = max(1, int(len(cat_cases) * TEST_FRACTION))
        test_cases.extend(cat_cases[:n_test])
        train_cases.extend(cat_cases[n_test:])
        print(f"  {cat}: {len(cat_cases)} total -> "
              f"{len(cat_cases) - n_test} train, {n_test} test")

    for subdir in ["imagesTr", "labelsTr", "imagesTs", "labelsTs",
                    "outlinesTr", "outlinesTs"]:
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    outline_missing_count = 0

    def write_cases(case_list, img_subdir, lbl_subdir, outline_subdir,
                    start_id, split_name):
        nonlocal outline_missing_count
        case_id = start_id
        for case in case_list:
            case_name = f"liver_{case_id:04d}"

            img = Image.open(case["image"])
            img_gray = np.array(img.convert("L"), dtype=np.uint8)

            # Apply outline mask: zero out pixels outside scan sector
            if case["outline_json"].exists():
                outline_poly = load_polygon(case["outline_json"])
                outline_mask = render_binary_mask(outline_poly, img.size)
                img_gray = img_gray * outline_mask

                # Save outline mask for future use
                Image.fromarray(outline_mask * 255).save(
                    args.output_dir / outline_subdir / f"{case_name}.png")
            else:
                outline_missing_count += 1

            Image.fromarray(img_gray).save(
                args.output_dir / img_subdir / f"{case_name}_0000.png")

            polygons = {}
            if case["liver_json"].exists():
                polygons[1] = load_polygon(case["liver_json"])
            if case["mass_json"].exists():
                polygons[2] = load_polygon(case["mass_json"])

            label = render_mask(polygons, img.size)
            Image.fromarray(label).save(
                args.output_dir / lbl_subdir / f"{case_name}.png")

            mapping.append({
                "case_name": case_name,
                "split": split_name,
                "category": case["category"],
                "original_file": case["original_file"],
            })

            case_id += 1
        return case_id

    next_id = write_cases(train_cases, "imagesTr", "labelsTr", "outlinesTr",
                          1, "train")
    write_cases(test_cases, "imagesTs", "labelsTs", "outlinesTs",
                next_id, "test")

    if outline_missing_count > 0:
        print(f"\nWarning: {outline_missing_count} cases missing outline "
              f"annotations (image pixels not masked)")

    dataset_json = {
        "channel_names": {"0": "ultrasound"},
        "labels": {"background": 0, "liver": 1, "mass": 2},
        "numTraining": len(train_cases),
        "file_ending": ".png",
    }
    with open(args.output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    with open(args.output_dir / "case_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nSplit: {len(train_cases)} training, {len(test_cases)} test")
    print(f"Outline masks saved to {outline_subdir}")
    print(f"Case mapping saved to {args.output_dir / 'case_mapping.json'}")
    print(f"Dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
