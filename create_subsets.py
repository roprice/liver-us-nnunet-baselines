"""
Create stratified subsampled training sets for data efficiency study.

Takes the full Dataset001_LiverUS and creates smaller datasets at
specified training set sizes. Test set (imagesTs/labelsTs) is identical
across all sizes. Stratification preserves the ratio of mass-present
vs mass-absent cases.

Note: subsets are independent stratified samples, not nested. The 25
images in the n=25 set are not necessarily a subset of the 50 images
in the n=50 set. Each size is drawn independently using sequential
calls to the same RNG (seed 42).

Usage:
    python create_subsets.py \\
        --source-dir $nnUNet_raw/Dataset001_LiverUS \\
        --output-base $nnUNet_raw
"""

import argparse
import json
import os
import shutil
import numpy as np
from PIL import Image


SUBSET_SIZES = [25, 50, 100, 200, 300, 400, 500]
DATASET_IDS = {25: 2, 50: 3, 100: 4, 200: 5, 300: 6, 400: 7, 500: 8}
RANDOM_SEED = 42


def has_mass(label_path):
    """Check if a label mask contains mass (class 2) pixels."""
    label = np.array(Image.open(label_path))
    return np.any(label == 2)


def stratified_subsample(cases_with_mass, cases_without_mass, n, rng):
    """Subsample n cases preserving the mass/no-mass ratio."""
    total = len(cases_with_mass) + len(cases_without_mass)
    n_mass = int(round(n * len(cases_with_mass) / total))
    n_no_mass = n - n_mass

    n_mass = min(n_mass, len(cases_with_mass))
    n_no_mass = min(n_no_mass, len(cases_without_mass))

    # Adjust if rounding left us short
    while n_mass + n_no_mass < n:
        if n_mass < len(cases_with_mass):
            n_mass += 1
        else:
            n_no_mass += 1

    selected_mass = rng.choice(cases_with_mass, n_mass, replace=False).tolist()
    selected_no_mass = rng.choice(cases_without_mass, n_no_mass, replace=False).tolist()

    return sorted(selected_mass + selected_no_mass)


def create_subset(source_dir, output_base, size, selected_cases):
    """Create a new nnU-Net dataset with the selected training cases."""
    dataset_id = DATASET_IDS[size]
    dataset_name = f"Dataset{dataset_id:03d}_LiverUS_{size}"
    output_dir = os.path.join(output_base, dataset_name)

    # Create directories
    for subdir in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Copy selected training cases (images, labels, and outline masks)
    for case in selected_cases:
        img_src = os.path.join(source_dir, "imagesTr", f"{case}_0000.png")
        lbl_src = os.path.join(source_dir, "labelsTr", f"{case}.png")
        shutil.copy2(img_src, os.path.join(output_dir, "imagesTr", f"{case}_0000.png"))
        shutil.copy2(lbl_src, os.path.join(output_dir, "labelsTr", f"{case}.png"))

        outline_src = os.path.join(source_dir, "outlinesTr", f"{case}.png")
        if os.path.exists(outline_src):
            outline_dst_dir = os.path.join(output_dir, "outlinesTr")
            os.makedirs(outline_dst_dir, exist_ok=True)
            shutil.copy2(outline_src, os.path.join(outline_dst_dir, f"{case}.png"))

    # Copy full test set and outline masks (identical across all sizes)
    for subdir in ["imagesTs", "labelsTs", "outlinesTs"]:
        src_dir = os.path.join(source_dir, subdir)
        if not os.path.exists(src_dir):
            continue
        dst_dir = os.path.join(output_dir, subdir)
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "ultrasound"},
        "labels": {"background": 0, "liver": 1, "mass": 2},
        "numTraining": len(selected_cases),
        "file_ending": ".png",
    }
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"  {dataset_name}: {len(selected_cases)} training cases")
    return dataset_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True,
                        help="Path to Dataset001_LiverUS")
    parser.add_argument("--output-base", required=True,
                        help="Path to nnUNet_raw")
    args = parser.parse_args()

    # Classify training cases by mass presence
    label_dir = os.path.join(args.source_dir, "labelsTr")
    all_cases = sorted(
        f.replace(".png", "")
        for f in os.listdir(label_dir)
        if f.endswith(".png")
    )

    cases_with_mass = []
    cases_without_mass = []
    for case in all_cases:
        label_path = os.path.join(label_dir, f"{case}.png")
        if has_mass(label_path):
            cases_with_mass.append(case)
        else:
            cases_without_mass.append(case)

    print(f"Full training set: {len(all_cases)} cases "
          f"({len(cases_with_mass)} with mass, {len(cases_without_mass)} without)")

    rng = np.random.RandomState(RANDOM_SEED)

    print("\nCreating subsets:")
    for size in SUBSET_SIZES:
        selected = stratified_subsample(
            np.array(cases_with_mass),
            np.array(cases_without_mass),
            size,
            rng
        )
        create_subset(args.source_dir, args.output_base, size, selected)

    print("\nDone. Run run_efficiency_study.sh to train and predict.")


if __name__ == "__main__":
    main()
