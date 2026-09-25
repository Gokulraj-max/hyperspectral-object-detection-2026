#!/usr/bin/env bash
# Train HS-SAFD single model
python training/train.py --config configs/final.yaml "$@"
