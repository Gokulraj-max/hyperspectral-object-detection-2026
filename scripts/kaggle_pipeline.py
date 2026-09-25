"""
All-in-One Kaggle Runner for Hyperspectral Object Detection Challenge 2026
HS-SAFD Single Detection Model Pipeline (Strictly Rules-Compliant: NO ensembles)

Usage in Kaggle Notebook:
    !python scripts/kaggle_pipeline.py --epochs 30 --batch-size 8 --tta
"""

import os
import sys
import glob
import json
import shutil
import hashlib
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import build_detector
from datasets import CLASS_NAMES
from datasets.voc_parser import parse_voc_directory
from scripts.calculate_stats import calculate_dataset_stats
from scripts.create_split import create_split
from submission.generate_submission import create_submission_dataframe, parse_image_id
from submission.validate_submission import validate_submission_file


def banner(text: str):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def auto_detect_kaggle_paths() -> Dict[str, Optional[str]]:
    """
    Intelligently discovers dataset directories under /kaggle/input
    or falls back to local data/raw directories.
    """
    paths = {
        "train_dir": None,
        "annotations_dir": None,
        "test_dir": None,
        "ranking_dir": None,
        "sample_submission": None,
        "root_input": None
    }

    base_inputs = ["/kaggle/input", "data/raw", "../data/raw"]
    search_root = None
    for cand in base_inputs:
        if os.path.isdir(cand):
            search_root = cand
            break

    if search_root is None:
        print("[WARNING] Could not locate /kaggle/input or data/raw.")
        return paths

    paths["root_input"] = search_root
    print(f"[INFO] Scanning dataset tree under: {search_root}")

    # Search for sample_submission.csv
    for root, _, files in os.walk(search_root):
        for f in files:
            if "sample_submission" in f.lower() and f.endswith(".csv"):
                paths["sample_submission"] = os.path.join(root, f)
                break
        if paths["sample_submission"]:
            break

    # Search for XML annotations
    xml_dirs = set()
    for root, _, files in os.walk(search_root):
        if any(f.endswith(".xml") for f in files):
            xml_dirs.add(root)

    if xml_dirs:
        # Prefer directory named 'annotations' or 'Annotations'
        preferred = [d for d in xml_dirs if "annotation" in d.lower()]
        paths["annotations_dir"] = preferred[0] if preferred else list(xml_dirs)[0]

    # Search for image cubes (.npy, .npz, .tif, .tiff, .mat)
    cube_exts = (".npy", ".npz", ".tif", ".tiff", ".mat")
    cube_dirs = set()
    for root, _, files in os.walk(search_root):
        if any(f.endswith(cube_exts) for f in files):
            cube_dirs.add(root)

    for d in sorted(cube_dirs):
        d_lower = d.lower()
        if "train" in d_lower and paths["train_dir"] is None:
            paths["train_dir"] = d
        elif "test" in d_lower and paths["test_dir"] is None:
            paths["test_dir"] = d
        elif "ranking" in d_lower and paths["ranking_dir"] is None:
            paths["ranking_dir"] = d

    # Fallbacks if names are simple
    if paths["train_dir"] is None and len(cube_dirs) > 0:
        paths["train_dir"] = list(cube_dirs)[0]

    return paths


def print_detection_summary(paths: Dict[str, Optional[str]]):
    banner("DATASET DISCOVERY SUMMARY")
    cube_exts = (".npy", ".npz", ".tif", ".tiff", ".mat")

    for key, path in paths.items():
        if path and os.path.exists(path):
            if os.path.isdir(path):
                count = sum(1 for root, _, files in os.walk(path) for f in files if f.endswith(cube_exts) or f.endswith(".xml"))
                print(f"  {key:20s}: {path} ({count} relevant files)")
            else:
                size_kb = os.path.getsize(path) / 1024
                print(f"  {key:20s}: {path} ({size_kb:.1f} KB)")
        else:
            print(f"  {key:20s}: [Not Found / None]")


def run_pipeline(args):
    banner("HYPERSPECTRAL OBJECT DETECTION 2026 - KAGGLE EXECUTION")
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 1. Discover Paths
    paths = auto_detect_kaggle_paths()
    if args.data_dir:
        paths["train_dir"] = os.path.join(args.data_dir, "train") if os.path.isdir(os.path.join(args.data_dir, "train")) else args.data_dir
        paths["test_dir"] = os.path.join(args.data_dir, "test") if os.path.isdir(os.path.join(args.data_dir, "test")) else paths["test_dir"]
        paths["annotations_dir"] = os.path.join(args.data_dir, "annotations") if os.path.isdir(os.path.join(args.data_dir, "annotations")) else paths["annotations_dir"]

    print_detection_summary(paths)

    train_dir = paths["train_dir"]
    test_dir = paths["test_dir"]
    ann_dir = paths["annotations_dir"] or train_dir

    if train_dir is None or not os.path.exists(train_dir):
        print("[ERROR] Train directory not found. Please specify --data-dir pointing to the dataset.")
        return

    # 2. Compute Spectral Stats
    banner("STEP 1: Computing Per-Band Spectral Normalization Statistics")
    os.makedirs("data/processed", exist_ok=True)
    stats_file = "data/processed/spectral_stats.npz"
    if not os.path.exists(stats_file):
        stats = calculate_dataset_stats(train_dir, num_samples=100)
        np.savez(stats_file, mean=stats["mean"], std=stats["std"])
        print(f"Saved stats to: {stats_file}")
    else:
        print(f"Reusing existing stats at: {stats_file}")

    # 3. Create Train / Validation Splits
    banner("STEP 2: Creating 80/20 Train/Validation Split")
    os.makedirs("data/splits", exist_ok=True)
    train_split = "data/splits/train.txt"
    val_split = "data/splits/val.txt"
    if not os.path.exists(train_split) or not os.path.exists(val_split):
        create_split(data_dir=train_dir, val_ratio=args.val_ratio, output_dir="data/splits")
    else:
        print(f"Reusing splits: {train_split}, {val_split}")

    # 4. Train Model
    banner(f"STEP 3: Training HS-SAFD Single Detection Model ({args.epochs} Epochs)")
    from training.train import main as train_main
    save_dir = "checkpoints/kaggle_run"
    os.makedirs(save_dir, exist_ok=True)

    # Invoke training directly with CLI arguments
    train_args = [
        "scripts/train.py",
        "--config", args.config,
        "--data-dir", train_dir,
        "--annotations-dir", ann_dir,
        "--save-dir", save_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", device.type
    ]
    if args.lr:
        train_args.extend(["--lr", str(args.lr)])

    old_argv = sys.argv
    sys.argv = train_args
    try:
        train_main()
    finally:
        sys.argv = old_argv

    best_checkpoint = os.path.join(save_dir, "best.pt")
    if not os.path.exists(best_checkpoint):
        best_checkpoint = os.path.join(save_dir, "last.pt")

    print(f"\n[INFO] Training finished. Selected checkpoint: {best_checkpoint}")

    # 5. Inference on Test Images
    banner("STEP 4: Running Inference on Competition Test Set")
    if test_dir is None or not os.path.exists(test_dir):
        print(f"[WARNING] Test directory {test_dir} not found. Skipping inference.")
        return

    from inference.predict import main as predict_main
    pred_output = "outputs/predictions"
    os.makedirs(pred_output, exist_ok=True)

    predict_args = [
        "scripts/predict.py",
        "--config", args.config,
        "--checkpoint", best_checkpoint,
        "--input", test_dir,
        "--output", pred_output,
        "--conf-threshold", str(args.conf_threshold),
        "--device", device.type
    ]
    if args.tta:
        predict_args.append("--tta")

    sys.argv = predict_args
    try:
        predict_main()
    finally:
        sys.argv = old_argv

    # 6. Generate Submission CSV
    banner("STEP 5: Generating Kaggle submission.csv")
    from submission.generate_submission import main as gen_sub_main
    sub_output = "outputs/submissions/submission.csv"
    kaggle_working_sub = "/kaggle/working/submission.csv"

    gen_args = [
        "scripts/generate_submission.py",
        "--predictions", os.path.join(pred_output, "predictions.json"),
        "--output", sub_output,
        "--conf-threshold", str(args.conf_threshold)
    ]
    if paths.get("sample_submission"):
        gen_args.extend(["--sample-submission", paths["sample_submission"]])
    if os.path.isdir("/kaggle/working"):
        gen_args.extend(["--copy-to", kaggle_working_sub])

    sys.argv = gen_args
    try:
        gen_sub_main()
    finally:
        sys.argv = old_argv

    # 7. Validate Submission Against Competition Rules
    banner("STEP 6: Running 12-Point Submission Validation Suite")
    is_valid = validate_submission_file(sub_output, phase=args.phase)
    if is_valid:
        print("\n[SUCCESS] submission.csv passed ALL competition verification tests!")
        if os.path.exists(kaggle_working_sub):
            print(f"[KAGGLE READY] submission.csv is placed at: {kaggle_working_sub}")
    else:
        print("\n[WARNING] Validation flagged potential issues. Check output above.")

    # 8. SHA-256 Checksum
    with open(sub_output, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"\nFinal submission.csv SHA-256: {sha256}")
    banner("PIPELINE COMPLETE - READY FOR KAGGLE SUBMISSION")


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle Runner for Hyperspectral Object Detection")
    parser.add_argument("--config", type=str, default="configs/final.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default=None, help="Root competition data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (8 fits 16GB GPUs)")
    parser.add_argument("--lr", type=float, default=0.0004, help="Learning rate")
    parser.add_argument("--conf-threshold", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--tta", action="store_true", default=True, help="Enable single-model TTA")
    parser.add_argument("--phase", type=str, default="phase1", choices=["phase1", "phase2"], help="Competition phase")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda/cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
