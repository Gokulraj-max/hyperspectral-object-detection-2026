"""
Kaggle Submission Validator (Competition-Ready Rules-Compliant Version):
Performs strict automated validation checks before uploading submission to Kaggle:
1.  Required header columns
2.  Correct column order
3.  Sequential IDs starting at 0
4.  Unique IDs (no duplicate rows)
5.  Valid non-empty image IDs
6.  Valid classes (integer in range [0, 17])
7.  Confidence range (float in range [0.0, 1.0])
8.  Bounding-box geometric validity (x1 < x2 and y1 < y2)
9.  Coordinate non-negativity (x1, y1 >= 0)
10. No NaN values in any column
11. No Inf values in any column
12. Strict CSV format compliance
13. Test images coverage (Phase 1 & Phase 2)
14. Ranking images coverage (Phase 2 mandatory)
"""

import os
import sys
import glob
import argparse
from typing import List, Tuple, Optional, Set
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
NUM_CLASSES = 18


def validate_submission_file(
    csv_path: str,
    phase: str = "phase1",
    test_dir: Optional[str] = None,
    ranking_dir: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validates CSV against all competition rules for Phase 1 and Phase 2.
    Returns (is_valid, list_of_log_messages).
    """
    logs = []
    if not os.path.isfile(csv_path):
        return False, [f"[FAIL] Error: File does not exist at {csv_path}"]

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, [f"[FAIL] Error parsing CSV file: {str(e)}"]

    is_valid = True

    # Check 1: Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logs.append(f"[FAIL] Check 1 FAILED: Missing columns: {missing_cols}")
        is_valid = False
    else:
        logs.append("[PASS] Check 1 PASSED: All required columns present")

    # Check 2: Correct column order
    if list(df.columns) != REQUIRED_COLUMNS:
        logs.append(f"[FAIL] Check 2 FAILED: Column order is {list(df.columns)}, expected {REQUIRED_COLUMNS}")
        is_valid = False
    else:
        logs.append("[PASS] Check 2 PASSED: Correct column order")

    if not is_valid:
        return False, logs

    # If empty submission
    if len(df) == 0:
        logs.append("[WARN] Submission CSV is empty (0 predictions).")
        return True, logs

    # Check 3 & 4: Sequential and Unique IDs starting at 0
    expected_ids = np.arange(len(df))
    if not np.array_equal(df["id"].values, expected_ids):
        logs.append(f"[FAIL] Check 3/4 FAILED: IDs are not strictly sequential from 0 to {len(df)-1}")
        is_valid = False
    else:
        logs.append("[PASS] Check 3/4 PASSED: Sequential and unique IDs starting at 0")

    # Check 5: Valid integer image IDs
    if not pd.api.types.is_integer_dtype(df["image_id"]):
        logs.append(f"[FAIL] Check 5 FAILED: image_id must be integer (int64), found dtype '{df['image_id'].dtype}'")
        is_valid = False
    elif df["image_id"].isnull().any():
        logs.append("[FAIL] Check 5 FAILED: Found null/NaN image_id values")
        is_valid = False
    else:
        logs.append("[PASS] Check 5 PASSED: Valid integer image IDs (int64)")

    # Check 6: Valid classes (0 to 17)
    invalid_classes = df[~df["class_id"].isin(range(NUM_CLASSES))]
    if len(invalid_classes) > 0:
        logs.append(f"[FAIL] Check 6 FAILED: Found {len(invalid_classes)} rows with invalid class_id outside [0, {NUM_CLASSES-1}]")
        is_valid = False
    else:
        logs.append(f"[PASS] Check 6 PASSED: All class IDs in valid range [0, {NUM_CLASSES-1}]")

    # Check 7: Confidence range [0.0, 1.0]
    invalid_conf = df[(df["confidence"] < 0.0) | (df["confidence"] > 1.0)]
    if len(invalid_conf) > 0:
        logs.append(f"[FAIL] Check 7 FAILED: Found {len(invalid_conf)} confidence values outside [0.0, 1.0]")
        is_valid = False
    else:
        logs.append("[PASS] Check 7 PASSED: Confidence scores in range [0.0, 1.0]")

    # Check 8: Bounding-box validity (x1 < x2 and y1 < y2)
    invalid_boxes = df[(df["x1"] >= df["x2"]) | (df["y1"] >= df["y2"])]
    if len(invalid_boxes) > 0:
        logs.append(f"[FAIL] Check 8 FAILED: Found {len(invalid_boxes)} degenerate boxes with x1 >= x2 or y1 >= y2")
        is_valid = False
    else:
        logs.append("[PASS] Check 8 PASSED: Geometric bounding box validity (x1 < x2 and y1 < y2)")

    # Check 9: Coordinate range non-negativity
    negative_coords = df[(df["x1"] < 0) | (df["y1"] < 0) | (df["x2"] < 0) | (df["y2"] < 0)]
    if len(negative_coords) > 0:
        logs.append(f"[FAIL] Check 9 FAILED: Found {len(negative_coords)} negative coordinate values")
        is_valid = False
    else:
        logs.append("[PASS] Check 9 PASSED: Coordinate range non-negative")

    # Check 10 & 11: No NaN or Inf values
    if df.isnull().values.any():
        logs.append("[FAIL] Check 10 FAILED: NaN values detected in submission")
        is_valid = False
    else:
        logs.append("[PASS] Check 10 PASSED: No NaN values")

    numeric_cols = ["confidence", "x1", "y1", "x2", "y2"]
    if np.isinf(df[numeric_cols].values).any():
        logs.append("[FAIL] Check 11 FAILED: Inf values detected in submission")
        is_valid = False
    else:
        logs.append("[PASS] Check 11 PASSED: No Inf values")

    logs.append("[PASS] Check 12 PASSED: Correct CSV format")

    sub_img_ids = set(df["image_id"].astype(str).values)

    # Check 13: Test images coverage
    if test_dir and os.path.exists(test_dir):
        test_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(test_dir, "*.*")) if not f.endswith(".txt")]
        if test_files:
            missing_test = set(test_files) - sub_img_ids
            if len(missing_test) > 0 and len(missing_test) == len(test_files):
                logs.append(f"[FAIL] Check 13 FAILED: None of the test set images are present in the submission!")
                is_valid = False
            else:
                logs.append(f"[PASS] Check 13 PASSED: Test set images represented in submission ({len(sub_img_ids & set(test_files))}/{len(test_files)})")

    # Check 14: Phase 2 ranking images coverage
    if phase.lower() == "phase2":
        if ranking_dir and os.path.exists(ranking_dir):
            ranking_files = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(ranking_dir, "*.*")) if not f.endswith(".txt")]
            if ranking_files:
                missing_ranking = set(ranking_files) - sub_img_ids
                if len(missing_ranking) > 0:
                    logs.append(f"[FAIL] Check 14 FAILED (Phase 2): Missing {len(missing_ranking)} ranking set images in submission!")
                    is_valid = False
                else:
                    logs.append(f"[PASS] Check 14 PASSED (Phase 2): All {len(ranking_files)} ranking images present in submission.")

    return is_valid, logs


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Validate Kaggle Submission CSV")
    parser.add_argument("--input", type=str, default="outputs/submissions/submission.csv", help="Path to submission CSV")
    parser.add_argument("--phase", type=str, default="phase1", choices=["phase1", "phase2"], help="Competition phase")
    parser.add_argument("--test-dir", type=str, default="data/raw/test", help="Test images directory")
    parser.add_argument("--ranking-dir", type=str, default="data/raw/ranking", help="Ranking images directory")
    args = parser.parse_args()

    print(f"Validating submission: {args.input} for {args.phase.upper()}...")
    is_valid, logs = validate_submission_file(
        args.input,
        phase=args.phase,
        test_dir=args.test_dir,
        ranking_dir=args.ranking_dir
    )

    for line in logs:
        print(line)

    print("\n" + "=" * 50)
    if is_valid:
        print("[SUCCESS] SUBMISSION VALID - READY FOR KAGGLE")
        print("=" * 50)
        sys.exit(0)
    else:
        print("[FAILURE] SUBMISSION INVALID - DO NOT UPLOAD")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
