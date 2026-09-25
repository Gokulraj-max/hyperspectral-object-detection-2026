#!/usr/bin/env bash
# ==============================================================================
# Hyperspectral Object Detection Challenge 2026 - Official Reproduction Script
# Model: HS-SAFD (Hyperspectral Spectral-Attention Fusion Detector) - Single Model
# ==============================================================================
set -e

echo "=== STEP 1: Setting up environment ==="
conda env create -f environment.yml || true
conda activate hsi-detect || true

echo "=== STEP 2: Running Single-Model Inference on Test + Ranking Sets ==="
python inference/predict.py \
    --config configs/final.yaml \
    --checkpoint checkpoints/final.pt \
    --input data/raw/test \
    --output outputs/predictions/test \
    --tta

if [ -d "data/raw/ranking" ]; then
    python inference/predict.py \
        --config configs/final.yaml \
        --checkpoint checkpoints/final.pt \
        --input data/raw/ranking \
        --output outputs/predictions/ranking \
        --tta
fi

echo "=== STEP 3: Generating Submission CSV ==="
python submission/generate_submission.py \
    --predictions outputs/predictions/predictions.json \
    --output outputs/submissions/submission.csv

echo "=== STEP 4: Validating Submission ==="
python submission/validate_submission.py \
    --input outputs/submissions/submission.csv \
    --phase phase2

echo "=== STEP 5: Verifying SHA256 Hash ==="
sha256sum outputs/submissions/submission.csv > release/submission.sha256
cat release/submission.sha256

echo "=== REPRODUCTION COMPLETED SUCCESSFULLY ==="
