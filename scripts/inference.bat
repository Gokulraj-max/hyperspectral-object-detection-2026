@echo off
py -3.10 inference\predict.py --config configs\final.yaml --checkpoint checkpoints\exp04_final\best.pt --input data\raw\test --output outputs\predictions --tta %*
