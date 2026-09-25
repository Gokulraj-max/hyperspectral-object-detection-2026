"""
Automatic Error Analysis System (Section 17):
Classifies detections into:
1. Correct detection (IoU >= 0.5, correct class)
2. False positive (high confidence background prediction)
3. False negative (unmatched ground truth)
4. Poor localization (IoU in [0.1, 0.5), correct class)
5. Wrong class (IoU >= 0.5, wrong class prediction)
6. Duplicate detection (multiple predictions for same GT)
Generates error summary reports and side-by-side diagnostic visualizations.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from datasets import CLASS_NAMES
from datasets.visualization import render_pseudo_rgb, draw_bounding_boxes
from losses.iou_loss import box_iou


def analyze_errors(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, Any]],
    iou_thresh: float = 0.5,
    poor_loc_thresh: float = 0.1
) -> Dict[str, Any]:
    """
    Categorizes all predictions and ground truths into 6 distinct diagnostic error buckets.
    """
    error_counts = {
        "correct_detection": 0,
        "false_positive": 0,
        "false_negative": 0,
        "poor_localization": 0,
        "wrong_class": 0,
        "duplicate_detection": 0
    }

    per_image_diagnostics = []

    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        p_boxes = pred["boxes"]
        p_labels = pred["labels"]
        p_scores = pred["scores"]

        g_boxes = target["boxes"]
        g_labels = target["labels"]

        if isinstance(p_boxes, np.ndarray):
            p_boxes = torch.from_numpy(p_boxes)
        if isinstance(p_labels, np.ndarray):
            p_labels = torch.from_numpy(p_labels)
        if isinstance(g_boxes, np.ndarray):
            g_boxes = torch.from_numpy(g_boxes)
        if isinstance(g_labels, np.ndarray):
            g_labels = torch.from_numpy(g_labels)

        num_p = p_boxes.shape[0]
        num_g = g_boxes.shape[0]

        img_diag = {
            "image_id": target.get("image_id", str(img_idx)),
            "prediction_errors": [],
            "missed_gts": []
        }

        if num_g == 0:
            for i in range(num_p):
                error_counts["false_positive"] += 1
                img_diag["prediction_errors"].append("false_positive")
            per_image_diagnostics.append(img_diag)
            continue

        if num_p == 0:
            for j in range(num_g):
                error_counts["false_negative"] += 1
                img_diag["missed_gts"].append(j)
            per_image_diagnostics.append(img_diag)
            continue

        ious = box_iou(p_boxes, g_boxes) # [num_p, num_g]
        matched_gt_set = set()

        for p_idx in range(num_p):
            p_cls = int(p_labels[p_idx])
            best_iou, best_g_idx = torch.max(ious[p_idx], dim=0)
            best_iou = float(best_iou)
            best_g_idx = int(best_g_idx)
            g_cls = int(g_labels[best_g_idx])

            if best_iou >= iou_thresh:
                if p_cls == g_cls:
                    if best_g_idx not in matched_gt_set:
                        error_counts["correct_detection"] += 1
                        matched_gt_set.add(best_g_idx)
                        img_diag["prediction_errors"].append("correct")
                    else:
                        error_counts["duplicate_detection"] += 1
                        img_diag["prediction_errors"].append("duplicate")
                else:
                    error_counts["wrong_class"] += 1
                    img_diag["prediction_errors"].append("wrong_class")
            elif best_iou >= poor_loc_thresh:
                if p_cls == g_cls:
                    error_counts["poor_localization"] += 1
                    img_diag["prediction_errors"].append("poor_localization")
                else:
                    error_counts["false_positive"] += 1
                    img_diag["prediction_errors"].append("false_positive")
            else:
                error_counts["false_positive"] += 1
                img_diag["prediction_errors"].append("false_positive")

        # Missed GTs (False Negatives)
        for g_idx in range(num_g):
            if g_idx not in matched_gt_set:
                error_counts["false_negative"] += 1
                img_diag["missed_gts"].append(g_idx)

        per_image_diagnostics.append(img_diag)

    total_pred = sum(p["boxes"].shape[0] for p in predictions)
    total_gt = sum(t["boxes"].shape[0] for t in targets)

    return {
        "summary": error_counts,
        "total_predictions": total_pred,
        "total_ground_truth": total_gt,
        "diagnostics": per_image_diagnostics
    }


def visualize_error_diagnostics(
    cube: np.ndarray,
    pred_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, Any],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Renders side-by-side comparison:
    Left: Ground Truth | Right: Predictions
    """
    base_rgb = render_pseudo_rgb(cube)

    gt_overlay = draw_bounding_boxes(
        base_rgb.copy(),
        target_dict["boxes"],
        target_dict["labels"],
        scores=None,
        class_names=CLASS_NAMES
    )

    pred_overlay = draw_bounding_boxes(
        base_rgb.copy(),
        pred_dict["boxes"],
        pred_dict["labels"],
        scores=pred_dict.get("scores", None),
        class_names=CLASS_NAMES
    )

    h, w, _ = base_rgb.shape
    combined = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
    combined[:, :w, :] = gt_overlay
    combined[:, w + 10:, :] = pred_overlay

    # Title header banner
    cv2.putText(combined, "GROUND TRUTH", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "MODEL PREDICTIONS", (w + 25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    return combined
