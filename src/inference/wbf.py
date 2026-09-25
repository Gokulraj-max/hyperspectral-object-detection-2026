"""
Weighted Box Fusion (WBF) Implementation (Sections 20 & 21).
Ensembles bounding box coordinates and confidences across models and TTA variants.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch


def bb_intersection_over_union(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-7)
    return iou


def weighted_box_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.05,
    conf_type: str = "avg"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Box Fusion algorithm:
    boxes_list: list of [N_i, 4] for each model
    scores_list: list of [N_i] for each model
    labels_list: list of [N_i] for each model
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    weights = np.array(weights) / sum(weights)

    # Filter out boxes below skip_box_thr and group by label
    overall_boxes = []
    overall_scores = []
    overall_labels = []

    for m_idx in range(len(boxes_list)):
        m_boxes = boxes_list[m_idx]
        m_scores = scores_list[m_idx]
        m_labels = labels_list[m_idx]
        m_weight = weights[m_idx]

        for b, s, l in zip(m_boxes, m_scores, m_labels):
            if s >= skip_box_thr:
                overall_boxes.append(b)
                overall_scores.append(s * m_weight)
                overall_labels.append(int(l))

    if len(overall_boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    unique_labels = np.unique(overall_labels)
    final_boxes = []
    final_scores = []
    final_labels = []

    for label in unique_labels:
        # Filter by class
        lbl_indices = [i for i, l in enumerate(overall_labels) if l == label]
        lbl_boxes = [overall_boxes[i] for i in lbl_indices]
        lbl_scores = [overall_scores[i] for i in lbl_indices]

        # Sort descending by score
        order = np.argsort(lbl_scores)[::-1]
        lbl_boxes = [lbl_boxes[i] for i in order]
        lbl_scores = [lbl_scores[i] for i in order]

        # Clusters of boxes
        clusters = [] # list of dicts: {'boxes': [...], 'scores': [...], 'fused_box': ...}

        for b, s in zip(lbl_boxes, lbl_scores):
            matched = False
            for cluster in clusters:
                # Check IoU with fused box of cluster
                if bb_intersection_over_union(b, cluster["fused_box"]) >= iou_thr:
                    cluster["boxes"].append(b)
                    cluster["scores"].append(s)
                    # Recompute weighted coordinate
                    w_sum = sum(cluster["scores"])
                    weighted_coords = np.sum([box * sc for box, sc in zip(cluster["boxes"], cluster["scores"])], axis=0) / w_sum
                    cluster["fused_box"] = weighted_coords
                    matched = True
                    break

            if not matched:
                clusters.append({
                    "boxes": [b],
                    "scores": [s],
                    "fused_box": b
                })

        for cluster in clusters:
            fused_b = cluster["fused_box"]
            if conf_type == "avg":
                fused_s = np.mean(cluster["scores"])
            elif conf_type == "max":
                fused_s = np.max(cluster["scores"])
            else:
                fused_s = np.sum(cluster["scores"]) / len(boxes_list)

            final_boxes.append(fused_b)
            final_scores.append(fused_s)
            final_labels.append(label)

    return (
        np.array(final_boxes, dtype=np.float32),
        np.array(final_scores, dtype=np.float32),
        np.array(final_labels, dtype=np.int64)
    )
