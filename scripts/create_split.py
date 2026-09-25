"""
Create Stratified Train / Validation Split.
Outputs:
- data/splits/train.txt
- data/splits/val.txt
"""

import os
import sys
import glob
import random
import argparse
from typing import List, Dict

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.voc_parser import parse_voc_annotation


def create_split(data_dir: str = "data/raw", val_ratio: float = 0.2, seed: int = 42, output_dir: str = "data/splits"):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(data_dir, "train")
    image_files = []
    if os.path.exists(train_dir):
        for ext in (".npy", ".npz", ".tif", ".tiff", ".mat"):
            image_files.extend(glob.glob(os.path.join(train_dir, f"*{ext}")))
    
    stems = sorted([os.path.splitext(os.path.basename(fp))[0] for fp in image_files])

    if len(stems) == 0:
        print("[INFO] No raw training images found in data/raw/train. Generating dummy 100 split IDs for testing.")
        stems = [f"image_{i:04d}" for i in range(100)]

    random.shuffle(stems)
    n_val = max(1, int(len(stems) * val_ratio))
    val_ids = stems[:n_val]
    train_ids = stems[n_val:]

    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_ids:
            f.write(f"{item}\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_ids:
            f.write(f"{item}\n")

    print(f"Created dataset split:")
    print(f"  Train: {len(train_ids)} -> {train_file}")
    print(f"  Val:   {len(val_ids)}   -> {val_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Train / Validation Split")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory")
    args = parser.parse_args()

    create_split(args.data_dir, args.val_ratio, args.seed, args.output_dir)
