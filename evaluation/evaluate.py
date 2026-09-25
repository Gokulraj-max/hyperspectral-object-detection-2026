"""
Standalone Model Evaluation Script:
Runs complete evaluation suite on validation set:
- mAP@50, mAP@75, mAP@90, mAP@50-95
- Per-class AP table
- Real vs Counterfeit pair discrimination analysis
- Diagnostic error analysis
- Confusion matrix generation
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List
import yaml
import json
import torch
from torch.utils.data import DataLoader

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import HyperspectralDataset, HyperspectralAugmentation, detection_collate_fn, CLASS_NAMES
from models import build_detector
from evaluation.metrics import evaluate_detections
from evaluation.per_class import format_per_class_table, analyze_real_vs_counterfeit
from evaluation.confusion_matrix import DetectionConfusionMatrix
from evaluation.error_analysis import analyze_errors


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hyperspectral Object Detector")
    parser.add_argument("--config", type=str, default="configs/hyperspectral.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics", help="Directory to save evaluation reports")
    parser.add_argument("--synthetic-eval", action="store_true", help="Run quick evaluation on synthetic samples")
    return parser.parse_args()


def run_evaluation(config_path: str, checkpoint_path: Optional[str] = None, device: str = "cpu", output_dir: str = "outputs/metrics", synthetic: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    device_obj = torch.device(device)

    # 1. Build Model
    model = build_detector(cfg)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device_obj)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
    else:
        print("No valid checkpoint specified or file missing. Evaluating initialized weights.")
    model.to(device_obj)
    model.eval()

    # 2. Build Dataset & DataLoader
    img_size = tuple(cfg.get("data", {}).get("image_size", [512, 512]))
    val_transforms = HyperspectralAugmentation.build_val_transforms(image_size=img_size)

    val_dataset = HyperspectralDataset(
        data_dir=cfg.get("data", {}).get("raw_dir", "data/raw"),
        split_file=cfg.get("data", {}).get("val_split", "data/splits/val.txt"),
        mode=cfg.get("model", {}).get("mode", "16band"),
        selected_bands=cfg.get("model", {}).get("selected_bands", [2, 7, 15]),
        transforms=val_transforms,
        image_size=img_size,
        is_train=False,
        synthetic_count=10 if (synthetic or len(os.listdir("data/raw/train") if os.path.exists("data/raw/train") else []) == 0) else 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn
    )

    print(f"Evaluating on {len(val_dataset)} samples...")

    # 3. Inference Loop
    all_predictions = []
    all_targets = []
    eval_cfg = cfg.get("evaluation", {})
    conf_thresh = eval_cfg.get("conf_threshold", 0.05)
    iou_thresh = eval_cfg.get("nms_iou_threshold", 0.50)

    cm = DetectionConfusionMatrix(num_classes=cfg.get("model", {}).get("num_classes", 18))

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device_obj)
            preds = model.predict(images, conf_threshold=conf_thresh, nms_iou_threshold=iou_thresh)
            all_predictions.extend(preds)
            all_targets.extend(targets)
            cm.update(preds, targets)

    # 4. Compute Metrics
    metrics = evaluate_detections(all_predictions, all_targets, num_classes=cfg.get("model", {}).get("num_classes", 18))
    table_str = format_per_class_table(metrics, CLASS_NAMES)
    pair_analysis = analyze_real_vs_counterfeit(metrics, CLASS_NAMES)
    error_summary = analyze_errors(all_predictions, all_targets)

    # 5. Print Results
    print("\n" + "=" * 65)
    print("HYPERSPECTRAL OBJECT DETECTION EVALUATION REPORT")
    print("=" * 65)
    print(f"mAP@0.50:       {metrics['mAP_50']:.4f}")
    print(f"mAP@0.75:       {metrics['mAP_75']:.4f}")
    print(f"mAP@0.90:       {metrics['mAP_90']:.4f}")
    print(f"mAP@[0.50:0.95]:{metrics['mAP_50_95']:.4f}")
    print("-" * 65)
    print("PER-CLASS AVERAGE PRECISION:")
    print(table_str)
    print("-" * 65)
    print(f"Real vs Counterfeit Pair Mean AP50: {pair_analysis['mean_pair_ap50']:.4f}")
    print("-" * 65)
    print("ERROR BREAKDOWN:")
    for k, v in error_summary["summary"].items():
        print(f"  {k:22s}: {v}")
    print("=" * 65)

    # 6. Save Reports
    report_file = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_file, "w") as f:
        f.write(f"mAP@0.50: {metrics['mAP_50']:.4f}\n")
        f.write(f"mAP@0.75: {metrics['mAP_75']:.4f}\n")
        f.write(f"mAP@0.90: {metrics['mAP_90']:.4f}\n")
        f.write(f"mAP@[0.50:0.95]: {metrics['mAP_50_95']:.4f}\n\n")
        f.write(table_str + "\n\n")
        f.write(f"Real vs Counterfeit Mean AP50: {pair_analysis['mean_pair_ap50']:.4f}\n")
        f.write(json.dumps(error_summary["summary"], indent=2))

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    cm.plot(save_path=cm_path)
    print(f"Saved reports and confusion matrix to {output_dir}")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        synthetic=args.synthetic_eval
    )
