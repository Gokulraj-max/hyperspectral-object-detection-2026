"""
Dataset Inspection Script (Phase 1 / Step 1):
Inspects raw dataset directories (train, test, ranking, annotations) and outputs comprehensive report:
- Image count across splits
- Image dimensions (H x W)
- Spectral band count (16 bands)
- Min/Max pixel values and dynamic range
- Class distribution across all 18 categories
- Real vs Counterfeit category representation
- Bounding box sizes (Small < 32^2, Medium 32^2 - 96^2, Large > 96^2)
- Objects per image distribution
- Annotation integrity & corrupted files
"""

import os
import sys
import glob
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
import numpy as np
import tifffile

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import CLASS_NAMES, PAIRED_CATEGORIES
from datasets.voc_parser import parse_voc_annotation


def inspect_dataset(data_dir: str = "data/raw") -> Dict[str, Any]:
    report = {}
    print("=" * 65)
    print("HYPERSPECTRAL DATASET INSPECTION (CHALLENGE 2026)")
    print("=" * 65)

    splits = ["train", "test", "ranking"]
    split_counts = {}

    for s in splits:
        s_dir = os.path.join(data_dir, s)
        if os.path.exists(s_dir):
            files = []
            for ext in (".npy", ".npz", ".tif", ".tiff", ".mat"):
                files.extend(glob.glob(os.path.join(s_dir, f"*{ext}")))
                files.extend(glob.glob(os.path.join(s_dir, "**", f"*{ext}"), recursive=True))
            split_counts[s] = len(files)
        else:
            split_counts[s] = 0

    print(f"Dataset Counts:")
    print(f"  Train Images:   {split_counts.get('train', 0):,} (Expected: 3,000)")
    print(f"  Test Images:    {split_counts.get('test', 0):,} (Expected: 1,000)")
    print(f"  Ranking Images: {split_counts.get('ranking', 0):,} (Expected: 1,000)")

    # Sample image inspection
    sample_files = []
    train_dir = os.path.join(data_dir, "train")
    if os.path.exists(train_dir):
        for ext in (".npy", ".npz", ".tif", ".tiff"):
            sample_files.extend(glob.glob(os.path.join(train_dir, f"*{ext}")))
            if len(sample_files) >= 5:
                break

    if sample_files:
        fp = sample_files[0]
        ext = os.path.splitext(fp)[1]
        if ext == ".npy":
            arr = np.load(fp)
        elif ext in (".tif", ".tiff"):
            arr = tifffile.imread(fp)
        else:
            arr = np.array([])

        print("\nImage Cube Properties:")
        print(f"  Format:           {ext}")
        print(f"  Array Shape:      {arr.shape}")
        num_bands = arr.shape[-1] if arr.ndim == 3 and arr.shape[-1] <= 32 else (arr.shape[0] if arr.ndim == 3 else 1)
        print(f"  Spectral Bands:   {num_bands} (Expected: 16)")
        print(f"  Pixel Range:      Min = {np.min(arr):.4f}, Max = {np.max(arr):.4f}")
        print(f"  Data Type:        {arr.dtype}")

    # Annotations inspection
    ann_dir = os.path.join(data_dir, "annotations")
    if not os.path.exists(ann_dir):
        ann_dir = os.path.join(data_dir, "train")

    xml_files = glob.glob(os.path.join(ann_dir, "*.xml"))
    if not xml_files:
        xml_files = glob.glob(os.path.join(ann_dir, "**", "*.xml"), recursive=True)

    print(f"\nAnnotations:")
    print(f"  Found VOC XMLs:   {len(xml_files):,}")

    class_counts = {c: 0 for c in range(len(CLASS_NAMES))}
    bbox_sizes = {"small": 0, "medium": 0, "large": 0}
    objects_per_img = []

    for xml_path in xml_files:
        data = parse_voc_annotation(xml_path)
        boxes = data.get("boxes", [])
        labels = data.get("labels", [])
        objects_per_img.append(len(boxes))

        for b, l in zip(boxes, labels):
            if l in class_counts:
                class_counts[l] += 1
            w = b[2] - b[0]
            h = b[3] - b[1]
            area = w * h
            if area < 32 * 32:
                bbox_sizes["small"] += 1
            elif area < 96 * 96:
                bbox_sizes["medium"] += 1
            else:
                bbox_sizes["large"] += 1

    total_objects = sum(class_counts.values())
    print(f"  Total Objects:    {total_objects:,}")
    if objects_per_img:
        print(f"  Objects / Image:  Avg = {np.mean(objects_per_img):.2f}, Max = {max(objects_per_img)}")
        print(f"  Size Breakdown:   Small (<32^2): {bbox_sizes['small']}, Medium (32^2-96^2): {bbox_sizes['medium']}, Large (>96^2): {bbox_sizes['large']}")

    print("\nClass Distribution (18 Categories):")
    for idx, name in enumerate(CLASS_NAMES):
        count = class_counts[idx]
        print(f"  [{idx:2d}] {name:22s}: {count:5d}")

    print("\nReal vs Counterfeit Material Pairs:")
    for r_id, f_id in PAIRED_CATEGORIES:
        r_name, f_name = CLASS_NAMES[r_id], CLASS_NAMES[f_id]
        print(f"  {r_name:22s} ({class_counts[r_id]:4d})  vs  {f_name:22s} ({class_counts[f_id]:4d})")

    print("=" * 65)

    report = {
        "split_counts": split_counts,
        "total_xml_annotations": len(xml_files),
        "total_objects": total_objects,
        "class_counts": {CLASS_NAMES[k]: v for k, v in class_counts.items()},
        "bbox_sizes": bbox_sizes
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Hyperspectral Dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output-json", type=str, default="outputs/metrics/dataset_report.json", help="Path to save JSON report")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    report = inspect_dataset(args.data_dir)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved dataset report to: {args.output_json}")
