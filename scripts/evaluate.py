"""
Evaluation Script CLI:
python scripts/evaluate.py --config configs/final.yaml --checkpoint checkpoints/best.pt
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.evaluate import parse_args, run_evaluation

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        synthetic=args.synthetic_eval
    )
