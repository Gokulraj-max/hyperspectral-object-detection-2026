#!/usr/bin/env bash
python evaluation/evaluate.py --config configs/final.yaml --checkpoint checkpoints/exp04_final/best.pt "$@"
