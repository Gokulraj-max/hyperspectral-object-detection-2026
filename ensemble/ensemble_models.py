"""
Ensemble Models CLI Entry Point (Section 21):
python ensemble/ensemble_models.py --config configs/ensemble.yaml --input data/raw/test
Combines predictions from multiple distinct checkpoints (RGB, 16-band, HS-SAFD, High-res) using WBF.
Generates fused predictions and submission.
"""

import os
import sys
import glob
import argparse
import yaml
import json
import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predict import load_cube
from ensemble.model_fusion import ModelFusionEnsemble
from ensemble.weighted_boxes import EnsembleBoxFuser
from submission.generate_submission import create_submission_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Hyperspectral Detectors")
    parser.add_argument("--config", type=str, default="configs/ensemble.yaml", help="Path to ensemble YAML")
    parser.add_argument("--input", type=str, default="data/raw/test", help="Test images directory")
    parser.add_argument("--output", type=str, default="outputs/predictions/ensemble", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ens_cfg = cfg.get("ensemble", {})
    models_cfg = cfg.get("models", [])

    print(f"Initializing ensemble with {len(models_cfg)} model branches...")
    engine = ModelFusionEnsemble(models_cfg, device=args.device)

    fuser = EnsembleBoxFuser(
        weights=engine.weights,
        iou_thr=float(ens_cfg.get("iou_threshold", 0.55)),
        skip_box_thr=float(ens_cfg.get("skip_box_threshold", 0.05)),
        conf_type=ens_cfg.get("conf_type", "avg")
    )

    # Discover test files
    input_files = []
    if os.path.isdir(args.input):
        for ext in (".npy", ".npz", ".tif", ".tiff"):
            input_files.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
    
    if len(input_files) == 0:
        print(f"No test files found in {args.input}. Generating 2 synthetic test samples for verification...")
        for s_i in range(2):
            syn_cube = np.random.uniform(0.1, 0.9, size=(512, 512, 16)).astype(np.float32)
            syn_path = os.path.join(args.output, f"sample_test_cube_{s_i}.npy")
            np.save(syn_path, syn_cube)
            input_files.append(syn_path)

    fused_results = {}
    print(f"Fusing predictions for {len(input_files)} test scenes...")

    for fp in input_files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        cube = load_cube(fp)
        
        # Get individual model predictions
        model_preds = engine.predict_image(
            cube,
            conf_thr=float(ens_cfg.get("skip_box_threshold", 0.05)),
            iou_thr=float(ens_cfg.get("iou_threshold", 0.55))
        )

        # WBF fusion
        fused = fuser.fuse(model_preds)
        fused_results[stem] = {
            "boxes": fused["boxes"].tolist(),
            "scores": fused["scores"].tolist(),
            "labels": fused["labels"].tolist()
        }

    # Save fused JSON
    json_path = os.path.join(args.output, "fused_predictions.json")
    with open(json_path, "w") as f:
        json.dump(fused_results, f, indent=2)

    # Generate ensemble submission CSV
    sub_path = cfg.get("output", {}).get("submission_file", "outputs/submissions/submission_ensemble.csv")
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    df = create_submission_dataframe(fused_results)
    df.to_csv(sub_path, index=False)

    print(f"Ensemble fusion completed successfully!")
    print(f"Predictions saved to: {json_path}")
    print(f"Ensemble submission saved to: {sub_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
