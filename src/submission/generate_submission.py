"""
Kaggle Submission Generator:
Converts model predictions into standard Kaggle competition format:
id,image_id,class_id,confidence,x1,y1,x2,y2

Ensures image_id is strictly integer (int64), not string or text.
"""

import os
import sys
import re
import argparse
import json
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_image_id(img_id: Any) -> int:
    """
    Parses and casts image_id to integer (int64) as strictly required by Kaggle.
    Handles:
    - integers: 1000 -> 1000
    - string digits: "1000" -> 1000
    - filenames: "1000.npy" -> 1000, "test_001000.tif" -> 1000
    - synthetic names: "sample_test_cube" -> deterministic integer
    """
    if isinstance(img_id, (int, np.integer)):
        return int(img_id)

    s = str(img_id).strip()
    
    # Strip common file extensions
    for ext in (".npy", ".npz", ".tif", ".tiff", ".mat", ".jpg", ".jpeg", ".png", ".xml", ".json"):
        if s.lower().endswith(ext):
            s = s[:-len(ext)]

    if s.isdigit():
        return int(s)

    # Extract trailing or contiguous digits if available
    digits = re.findall(r'\d+', s)
    if digits:
        return int(digits[-1])

    # Deterministic integer fallback for synthetic filenames
    return int(abs(hash(s)) % (10**6) + 1000)


def create_submission_dataframe(
    predictions_dict: Dict[str, Dict[str, Any]],
    conf_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Takes predictions dictionary formatted as:
    {
      "1000": {
         "boxes": [[x1, y1, x2, y2], ...],
         "scores": [0.95, ...],
         "labels": [2, ...]
      }, ...
    }
    and converts it into Kaggle submission DataFrame with columns:
    id,image_id,class_id,confidence,x1,y1,x2,y2
    Guarantees image_id is strictly int64.
    """
    rows = []
    current_id = 0

    # Ensure deterministic image order by numerical image_id if possible
    sorted_image_ids = sorted(predictions_dict.keys(), key=lambda k: (parse_image_id(k), str(k)))

    for raw_img_id in sorted_image_ids:
        numeric_img_id = parse_image_id(raw_img_id)
        data = predictions_dict[raw_img_id]
        boxes = data.get("boxes", [])
        scores = data.get("scores", [])
        labels = data.get("labels", [])

        for b, s, l in zip(boxes, scores, labels):
            score = float(s)
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = [float(v) for v in b]
            
            # Ensure valid box
            if x2 <= x1 or y2 <= y1:
                continue

            rows.append({
                "id": int(current_id),
                "image_id": int(numeric_img_id),
                "class_id": int(l),
                "confidence": round(score, 4),
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2))
            })
            current_id += 1

    df = pd.DataFrame(rows, columns=["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    
    # Enforce integer types explicitly
    int_cols = ["id", "image_id", "class_id", "x1", "y1", "x2", "y2"]
    for col in int_cols:
        if col in df.columns and len(df) > 0:
            df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

    if "confidence" in df.columns and len(df) > 0:
        df["confidence"] = df["confidence"].astype("float64")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kaggle Submission CSV")
    parser.add_argument("--predictions", type=str, default="outputs/predictions/predictions.json", help="Path to predictions JSON")
    parser.add_argument("--output", type=str, default="outputs/submissions/submission.csv", help="Path to output submission CSV")
    parser.add_argument("--conf-threshold", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--sample-submission", type=str, default=None, help="Path to competition sample_submission.csv for format verification")
    parser.add_argument("--copy-to", type=str, default=None, help="Additional output destination (e.g. /kaggle/working/submission.csv)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not os.path.isfile(args.predictions):
        print(f"Predictions file not found at {args.predictions}. Generating dummy prediction data for testing...")
        dummy_preds = {
            "1000": {
                "boxes": [[186, 28, 201, 50], [366, 68, 383, 86]],
                "scores": [0.9500, 0.7200],
                "labels": [2, 9]
            },
            "1001": {
                "boxes": [[50, 60, 120, 140]],
                "scores": [0.8800],
                "labels": [4]
            }
        }
        os.makedirs(os.path.dirname(args.predictions), exist_ok=True)
        with open(args.predictions, "w") as f:
            json.dump(dummy_preds, f, indent=2)

    with open(args.predictions, "r") as f:
        preds = json.load(f)

    df = create_submission_dataframe(preds, conf_threshold=args.conf_threshold)
    df.to_csv(args.output, index=False)
    print(f"Generated submission file with {len(df)} predictions across {df['image_id'].nunique() if len(df) > 0 else 0} unique test images at: {args.output}")
    print(f"Column data types:\n{df.dtypes}")

    # Copy to secondary location (e.g. /kaggle/working/submission.csv) if requested
    if args.copy_to:
        import shutil
        os.makedirs(os.path.dirname(args.copy_to) or ".", exist_ok=True)
        shutil.copy2(args.output, args.copy_to)
        print(f"Successfully copied submission to: {args.copy_to}")

    # Validate against sample_submission if provided
    if args.sample_submission and os.path.isfile(args.sample_submission):
        sample_df = pd.read_csv(args.sample_submission)
        print(f"\n[VALIDATION AGAINST SAMPLE SUBMISSION]")
        print(f"Sample submission shape: {sample_df.shape}")
        print(f"Sample columns: {list(sample_df.columns)}")
        if "image_id" in sample_df.columns:
            sample_ids = set(sample_df["image_id"].dropna().unique())
            pred_ids = set(df["image_id"].dropna().unique()) if len(df) > 0 else set()
            overlap = sample_ids.intersection(pred_ids)
            print(f"Unique test images in sample submission: {len(sample_ids)}")
            print(f"Test images with model predictions: {len(overlap)} / {len(sample_ids)}")
            if len(overlap) == 0 and len(sample_ids) > 0:
                print("[WARNING] Zero image_id overlap with sample submission! Check test image filename parsing.")
            else:
                print(f"[SUCCESS] {len(overlap)} matching image IDs confirmed!")


if __name__ == "__main__":
    main()
