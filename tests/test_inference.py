"""
Unit tests for Inference, TTA, NMS, and Postprocessing.
"""

import os
import sys
import numpy as np
import torch
import pytest

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference import non_max_suppression, soft_nms, TestTimeAugmentation, unpad_and_scale_boxes
from models import HS_SAFD


def test_nms_and_soft_nms():
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 52.0, 52.0], # high overlap with first
        [100.0, 100.0, 150.0, 150.0] # disjoint
    ])
    scores = torch.tensor([0.9, 0.7, 0.85])

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 2
    assert 0 in keep
    assert 2 in keep

    s_boxes, s_scores = soft_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.2)
    assert len(s_boxes) >= 2


def test_unpad_and_scale():
    boxes = np.array([[20.0, 30.0, 120.0, 150.0]])
    pad_info = (10, 20, 2.0) # pad_left=10, pad_top=20, scale=2.0
    orig_shape = (200, 200)

    unpadded = unpad_and_scale_boxes(boxes, pad_info, orig_shape)
    # x1 = (20 - 10) / 2.0 = 5.0
    # y1 = (30 - 20) / 2.0 = 5.0
    # x2 = (120 - 10) / 2.0 = 55.0
    # y2 = (150 - 20) / 2.0 = 65.0
    assert abs(unpadded[0, 0] - 5.0) < 1e-4
    assert abs(unpadded[0, 1] - 5.0) < 1e-4
    assert abs(unpadded[0, 2] - 55.0) < 1e-4
    assert abs(unpadded[0, 3] - 65.0) < 1e-4


def test_single_model_tta():
    model = HS_SAFD(num_classes=18, head_channels=64)
    model.eval()
    tta = TestTimeAugmentation(model, scales=[0.8, 1.0, 1.2], use_flips=True, conf_thr=0.01)

    x = torch.randn(1, 16, 128, 128)
    preds = tta(x)
    assert "boxes" in preds
    assert "scores" in preds
    assert "labels" in preds
    assert isinstance(preds["boxes"], np.ndarray)
