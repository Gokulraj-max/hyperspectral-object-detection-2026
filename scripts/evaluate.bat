@echo off
py -3.10 evaluation\evaluate.py --config configs\final.yaml --checkpoint checkpoints\exp04_final\best.pt %*
