#!/bin/bash
set -e

# Data efficiency study: train at multiple dataset sizes and evaluate
# each against the same held-out test set.
#
# Prerequisites:
#   - Dataset001_LiverUS present in nnUNet_raw
#   - create_subsets.py already run
#
# Usage:
#   bash run_efficiency_study.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export nnUNet_extTrainer="${SCRIPT_DIR}/custom_trainers"

# Require nnUNet env vars
: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

TRAINER="nnUNetTrainer150"

# Dataset ID mapping: size -> nnU-Net dataset ID
declare -A DATASETS
DATASETS[25]=2
DATASETS[50]=3
DATASETS[100]=4
DATASETS[200]=5
DATASETS[300]=6
DATASETS[400]=7
DATASETS[500]=8
DATASETS[625]=1

SIZES=(25 50 100 200 300 400 500 625)

echo "=== Data efficiency study ==="
echo ""
echo "=== Environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null || true
python -c "import nnunetv2; print(f'nnU-Net {nnunetv2.__version__}')" 2>/dev/null || true
echo ""
echo "Training sizes: ${SIZES[*]}"
echo "Trainer: ${TRAINER} (150 epochs)"
echo ""

for SIZE in "${SIZES[@]}"; do
    DATASET_ID=${DATASETS[$SIZE]}
    DATASET_NAME="Dataset$(printf '%03d' $DATASET_ID)_LiverUS"
    if [ "$SIZE" -ne 625 ]; then
        DATASET_NAME="${DATASET_NAME}_${SIZE}"
    fi

    echo "============================================"
    echo "=== Training with ${SIZE} images (Dataset ${DATASET_ID}) ==="
    echo "============================================"

    # Preprocess
    echo "--- Preprocessing ---"
    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

    # Train
    echo "--- Training ---"
    nnUNetv2_train $DATASET_ID 2d 0 --npz -tr $TRAINER

    # Predict on test set
    echo "--- Predicting ---"
    PRED_DIR="${nnUNet_results}/predictions_${SIZE}"
    nnUNetv2_predict \
        -i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" \
        -o "$PRED_DIR" \
        -d $DATASET_ID -c 2d -f 0 \
        -tr $TRAINER \
        -chk checkpoint_best.pth

    echo "=== Done with ${SIZE} images ==="
    echo ""
done

echo "============================================"
echo "=== All training runs complete ==="
echo "============================================"
echo ""
echo "Run evaluate_efficiency.py to compute Dice scores."
