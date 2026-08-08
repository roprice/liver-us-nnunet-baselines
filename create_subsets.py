"""
Create nested stratified subsampled training sets for data efficiency study.

Takes the full Dataset001_LiverUS and creates smaller datasets at
specified training set sizes. Test set (imagesTs/labelsTs) is identical
across all sizes. Stratification preserves the ratio of mass-present
vs mass-absent cases.

Subsets are NESTED: the 25-image subset is fully contained within the
50-image subset, which is fully contained within the 100-image subset,
and so on. This means the only variable changing between sizes is the
additional images, making the efficiency curve interpretable as a
monotonic data accumulation story.

The full 625-image pool is shuffled once (seed 42), stratified by mass
presence, then subsets are taken as prefixes of increasing length.

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


def build_nested_subsets(cases_with_mass, cases_without_mass, sizes, rng):
    """Build nested subsets preserving the mass/no-mass ratio at each size.

    Shuffles mass and no-mass cases independently, then for each size
    takes the first N_mass and N_no_mass cases from the shuffled lists.
    Because each size takes a prefix, smaller subsets are always contained
    within larger ones.
    """
    # Shuffle each pool once
    mass_shuffled = rng.permutation(cases_with_mass).tolist()
    no_mass_shuffled = rng.permutation(cases_without_mass).tolist()

    total = len(cases_with_mass) + len(cases_without_mass)
    mass_ratio = len(cases_with_mass) / total

    subsets = {}
    for size in sizes:
        n_mass = int(round(size * mass_ratio))
        n_no_mass = size - n_mass

        n_mass = min(n_mass, len(mass_shuffled))
        n_no_mass = min(n_no_mass, len(no_mass_shuffled))

        # Adjust if rounding left us short
        while n_mass + n_no_mass < size:
            if n_mass < len(mass_shuffled):
                n_mass += 1
            else:
                n_no_mass += 1

        selected = sorted(mass_shuffled[:n_mass] + no_mass_shuffled[:n_no_mass])
        subsets[size] = selected

    return subsets


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
    subsets = build_nested_subsets(cases_with_mass, cases_without_mass,
                                  SUBSET_SIZES, rng)

    # Verify nesting
    for i in range(len(SUBSET_SIZES) - 1):
        smaller = set(subsets[SUBSET_SIZES[i]])
        larger = set(subsets[SUBSET_SIZES[i + 1]])
        assert smaller.issubset(larger), \
            f"Nesting violated: {SUBSET_SIZES[i]} not subset of {SUBSET_SIZES[i+1]}"

    print("\nCreating nested subsets:")
    for size in SUBSET_SIZES:
        selected = subsets[size]
        n_mass = sum(1 for c in selected
                     if has_mass(os.path.join(label_dir, f"{c}.png")))
        name = create_subset(args.source_dir, args.output_base, size, selected)
        print(f"  {name}: {len(selected)} cases "
              f"({n_mass} with mass, {len(selected) - n_mass} without)")

    print("\nNesting verified: each subset is contained within the next.")
    print("Done. Run run_efficiency_study.sh to train and predict.")


if __name__ == "__main__":
    main()
