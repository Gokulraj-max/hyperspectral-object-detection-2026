@echo off
REM ==============================================================================
REM Hyperspectral Object Detection Challenge 2026 - Official Reproduction Script (Windows)
REM Model: HS-SAFD (Hyperspectral Spectral-Attention Fusion Detector) - Single Model
REM ==============================================================================

echo === STEP 1: Running Single-Model Inference on Test Set ===
py -3.10 inference\predict.py --config configs\final.yaml --checkpoint checkpoints\final.pt --input data\raw\test --output outputs\predictions\test --tta

if exist "data\raw\ranking" (
    echo === Running Inference on Ranking Set ===
    py -3.10 inference\predict.py --config configs\final.yaml --checkpoint checkpoints\final.pt --input data\raw\ranking --output outputs\predictions\ranking --tta
)

echo === STEP 2: Generating Submission CSV ===
py -3.10 submission\generate_submission.py --predictions outputs\predictions\predictions.json --output outputs\submissions\submission.csv

echo === STEP 3: Validating Submission ===
py -3.10 submission\validate_submission.py --input outputs\submissions\submission.csv --phase phase2

echo === STEP 4: Computing SHA256 Hash ===
certutil -hashfile outputs\submissions\submission.csv SHA256 > release\submission.sha256
type release\submission.sha256

echo === REPRODUCTION COMPLETED SUCCESSFULLY ===
pause
