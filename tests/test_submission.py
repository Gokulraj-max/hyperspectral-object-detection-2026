"""
Unit tests for Submission Generation and 12-point Kaggle Submission Validator.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
import pytest

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from submission import create_submission_dataframe, validate_submission_file


def test_submission_generator():
    mock_preds = {
        "1000": {
            "boxes": [[10.0, 20.0, 100.0, 150.0], [5.0, 5.0, 30.0, 40.0]],
            "scores": [0.95432, 0.45],
            "labels": [2, 7]
        },
        "1001": {
            "boxes": [[20.0, 20.0, 80.0, 80.0]],
            "scores": [0.88],
            "labels": [14]
        }
    }

    df = create_submission_dataframe(mock_preds, conf_threshold=0.1)
    assert len(df) == 3
    assert list(df.columns) == ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    assert list(df["id"].values) == [0, 1, 2]
    assert df["confidence"].iloc[0] == 0.9543
    assert df["class_id"].iloc[0] == 2


def test_submission_validator_pass_and_fail():
    # 1. Valid test
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("id,image_id,class_id,confidence,x1,y1,x2,y2\n")
        f.write("0,1000,2,0.9500,10,20,100,150\n")
        f.write("1,1001,17,0.8500,30,40,90,110\n")

    try:
        is_valid, logs = validate_submission_file(csv_path, phase="phase1")
        assert is_valid is True
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    # 2. Invalid class ID (> 17)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("id,image_id,class_id,confidence,x1,y1,x2,y2\n")
        f.write("0,1000,18,0.9500,10,20,100,150\n") # class 18 is invalid

    try:
        is_valid, logs = validate_submission_file(csv_path, phase="phase1")
        assert is_valid is False
        assert any("Check 6 FAILED" in l for l in logs)
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    # 3. Degenerate box (x1 >= x2)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        csv_path = f.name
        f.write("id,image_id,class_id,confidence,x1,y1,x2,y2\n")
        f.write("0,1000,2,0.9500,100,20,10,150\n") # x1 > x2

    try:
        is_valid, logs = validate_submission_file(csv_path, phase="phase1")
        assert is_valid is False
        assert any("Check 8 FAILED" in l for l in logs)
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
