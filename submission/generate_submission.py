"""
Kaggle Submission Generator (Section 23):
Converts model predictions or prediction dictionaries into standard Kaggle competition format:
id,image_id,class_id,confidence,x1,y1,x2,y2
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_submission_dataframe(predictions_dict: Dict[str, Dict[str, Any]], conf_threshold: float = 0.05) -> pd.DataFrame:
    """
    Takes predictions dictionary formatted as:
    {
      "image_id_1": {
         "boxes": [[x1, y1, x2, y2], ...],
         "scores": [0.95, ...],
         "labels": [2, ...]
      }, ...
    }
    and converts it into Kaggle submission DataFrame with columns:
    id,image_id,class_id,confidence,x1,y1,x2,y2
    """
    rows = []
    current_id = 0

    # Ensure deterministic image order
    sorted_image_ids = sorted(predictions_dict.keys())

    for img_id in sorted_image_ids:
        data = predictions_dict[img_id]
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
                "id": current_id,
                "image_id": str(img_id),
                "class_id": int(l),
                "confidence": round(score, 4),
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2))
            })
            current_id += 1

    df = pd.DataFrame(rows, columns=["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kaggle Submission CSV")
    parser.add_argument("--predictions", type=str, default="outputs/predictions/predictions.json", help="Path to predictions JSON")
    parser.add_argument("--output", type=str, default="outputs/submissions/submission.csv", help="Path to output submission CSV")
    parser.add_argument("--conf-threshold", type=float, default=0.05, help="Confidence threshold")
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
    print(f"Generated submission file with {len(df)} predictions at: {args.output}")


if __name__ == "__main__":
    main()
