#!/bin/bash
set -e

# Train the baseline model on the full Dataset001_LiverUS (625 images).
# For the full data efficiency study (including baseline), use
# run_efficiency_study.sh instead.

DATASET_ID=001
CONFIG=2d
FOLD=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Preprocessing and planning ==="
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

echo "=== Training (${CONFIG}, fold ${FOLD}, 150 epochs) ==="
export nnUNet_extTrainer="${SCRIPT_DIR}/custom_trainers"
nnUNetv2_train $DATASET_ID $CONFIG $FOLD \
  --npz \
  -tr nnUNetTrainer150
