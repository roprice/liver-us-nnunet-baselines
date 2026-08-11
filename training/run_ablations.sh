#!/bin/bash
set -e

# Ablation experiments on 200-image subset (Dataset005_LiverUS_200)
#
# Test 1: Does the 150-epoch cap matter?
#   PlainConvUNet, 300 epochs vs existing 150-epoch baseline
#
# Test 2: Does architecture matter?
#   ResEncUNet (M preset), 300 epochs
#
# Prerequisites:
#   - Dataset005_LiverUS_200 present in nnUNet_raw
#   - For Test 1: nnUNetTrainer300 in custom_trainers/
#   - For Test 2: ResEnc plans generated (see below)
#
# Usage:
#   bash training/run_ablations.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export nnUNet_extTrainer="${SCRIPT_DIR}/custom_trainers"

: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

DATASET_ID=5  # Dataset005_LiverUS_200
DATASET_NAME="Dataset005_LiverUS_200"

echo "=== Ablation experiments on 200-image subset ==="
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# ============================================================
# Test 1: PlainConvUNet, 300 epochs
# ============================================================
echo "============================================"
echo "=== Test 1: PlainConvUNet, 300 epochs ==="
echo "============================================"

# Preprocess (uses standard plans, may already exist from efficiency study)
echo "--- Preprocessing (standard plans) ---"
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

echo "--- Training ---"
nnUNetv2_train $DATASET_ID 2d 0 --npz -tr nnUNetTrainer300

echo "--- Predicting ---"
PRED_DIR="${nnUNet_results}/ablation_plainconv300_200"
nnUNetv2_predict \
    -i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" \
    -o "$PRED_DIR" \
    -d $DATASET_ID -c 2d -f 0 \
    -tr nnUNetTrainer300 \
    -chk checkpoint_best.pth

echo "=== Test 1 complete ==="
echo ""

# ============================================================
# Test 2: ResEncUNet (M preset), 300 epochs
# ============================================================
echo "============================================"
echo "=== Test 2: ResEncUNet M, 300 epochs ==="
echo "============================================"

# Plan with ResEnc planner (generates separate plans, won't overwrite)
echo "--- Preprocessing (ResEnc M plans) ---"
nnUNetv2_plan_and_preprocess -d $DATASET_ID -pl nnUNetPlannerResEncM --verify_dataset_integrity

echo "--- Training ---"
nnUNetv2_train $DATASET_ID 2d 0 --npz -tr nnUNetTrainer300 -p nnUNetResEncUNetMPlans

echo "--- Predicting ---"
PRED_DIR="${nnUNet_results}/ablation_resenc300_200"
nnUNetv2_predict \
    -i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" \
    -o "$PRED_DIR" \
    -d $DATASET_ID -c 2d -f 0 \
    -tr nnUNetTrainer300 \
    -p nnUNetResEncUNetMPlans \
    -chk checkpoint_best.pth

echo "=== Test 2 complete ==="
echo ""

echo "============================================"
echo "=== All ablation experiments complete ==="
echo "============================================"
echo ""
echo "Compare results:"
echo "  Baseline (150 epoch):  predictions_200/"
echo "  PlainConv 300 epoch:   ablation_plainconv300_200/"
echo "  ResEnc M 300 epoch:    ablation_resenc300_200/"
