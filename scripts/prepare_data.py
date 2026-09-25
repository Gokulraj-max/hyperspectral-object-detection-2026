"""
Data Preparation Script:
- Parses VOC XML annotations from data/raw/annotations
- Computes and saves per-band normalization statistics
- Creates default train/val split if missing
- Prepares directories for processed tensors
"""

import os
import sys
import glob
import json
import argparse
import numpy as np

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.voc_parser import parse_voc_directory
from scripts.calculate_stats import calculate_dataset_stats
from scripts.create_split import create_split


def prepare_dataset(raw_dir: str = "data/raw", processed_dir: str = "data/processed", val_ratio: float = 0.2):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)

    print("=" * 60)
    print("STEP 1: Checking Raw Datasets and VOC Annotations...")
    print("=" * 60)

    train_dir = os.path.join(raw_dir, "train")
    test_dir = os.path.join(raw_dir, "test")
    ranking_dir = os.path.join(raw_dir, "ranking")
    ann_dir = os.path.join(raw_dir, "annotations")
    if not os.path.exists(ann_dir):
        ann_dir = train_dir

    # Check VOC XMLs
    xmls = glob.glob(os.path.join(ann_dir, "*.xml"))
    if not xmls:
        xmls = glob.glob(os.path.join(ann_dir, "**", "*.xml"), recursive=True)

    print(f"Found {len(xmls)} VOC XML files.")
    if xmls:
        annotations_map = parse_voc_directory(ann_dir)
        index_file = os.path.join(processed_dir, "annotations_index.json")
        with open(index_file, "w") as f:
            json.dump(annotations_map, f, indent=2)
        print(f"Saved parsed VOC annotations index to: {index_file}")

    print("\n" + "=" * 60)
    print("STEP 2: Computing Spectral Normalization Parameters...")
    print("=" * 60)
    stats = calculate_dataset_stats(train_dir, num_samples=100)
    stats_file = os.path.join(processed_dir, "spectral_stats.npz")
    np.savez(stats_file, mean=stats["mean"], std=stats["std"])
    print(f"Saved spectral statistics to: {stats_file}")

    print("\n" + "=" * 60)
    print("STEP 3: Creating Train / Validation Split...")
    print("=" * 60)
    train_split = "data/splits/train.txt"
    val_split = "data/splits/val.txt"
    if not os.path.exists(train_split) or not os.path.exists(val_split):
        create_split(raw_dir, val_ratio=val_ratio, output_dir="data/splits")
    else:
        print(f"Existing train/val splits found at {train_split} and {val_split}.")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Hyperspectral Dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw dataset directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    args = parser.parse_args()

    prepare_dataset(args.raw_dir, args.processed_dir, args.val_ratio)
