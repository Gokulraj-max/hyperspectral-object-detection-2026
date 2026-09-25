"""
Comprehensive COCO-style Detection Metrics (Section 16):
Computes:
- mAP@0.50, mAP@0.55, ..., mAP@0.95
- mAP@[0.50:0.95]
- AP75, AP90
- Precision, Recall, F1 score
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch


def compute_ap_curve(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes Average Precision (AP) using standard 101-point COCO interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101-point interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_thresholds, side='left')
    ap = np.mean([mpre[min(i, len(mpre) - 1)] for i in inds])
    return float(ap)


def evaluate_detections(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, Any]],
    num_classes: int = 18,
    iou_thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Evaluates detector predictions against ground truth targets across IoU thresholds.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05).tolist()]

    # Collect per-class predictions and ground truths
    class_preds = {c: [] for c in range(num_classes)}
    class_gts = {c: {img_idx: [] for img_idx in range(len(targets))} for c in range(num_classes)}

    for img_idx, target in enumerate(targets):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()

        for b, l in zip(gt_boxes, gt_labels):
            l = int(l)
            if l in class_gts:
                class_gts[l][img_idx].append(b)

    for img_idx, pred in enumerate(predictions):
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        for b, s, l in zip(boxes, scores, labels):
            l = int(l)
            if l in class_preds:
                class_preds[l].append({"image_idx": img_idx, "box": b, "score": float(s)})

    # Sort predictions by score descending per class
    for c in range(num_classes):
        class_preds[c].sort(key=lambda x: x["score"], reverse=True)

    per_class_aps = {c: {} for c in range(num_classes)}
    total_gts_per_class = {c: sum(len(boxes) for boxes in class_gts[c].values()) for c in range(num_classes)}

    for iou_thr in iou_thresholds:
        for c in range(num_classes):
            num_gt = total_gts_per_class[c]
            preds = class_preds[c]

            if num_gt == 0:
                per_class_aps[c][iou_thr] = float("nan")
                continue
            if len(preds) == 0:
                per_class_aps[c][iou_thr] = 0.0
                continue

            # Track detected GT boxes
            detected = {img_idx: [False] * len(boxes) for img_idx, boxes in class_gts[c].items()}
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for p_idx, p in enumerate(preds):
                img_idx = p["image_idx"]
                p_box = p["box"]
                gts = class_gts[c][img_idx]

                if len(gts) == 0:
                    fp[p_idx] = 1
                    continue

                # Compute IoUs
                gts_arr = np.array(gts)
                ix1 = np.maximum(p_box[0], gts_arr[:, 0])
                iy1 = np.maximum(p_box[1], gts_arr[:, 1])
                ix2 = np.minimum(p_box[2], gts_arr[:, 2])
                iy2 = np.minimum(p_box[3], gts_arr[:, 3])

                iw = np.maximum(0.0, ix2 - ix1)
                ih = np.maximum(0.0, iy2 - iy1)
                inter = iw * ih

                p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                gt_areas = (gts_arr[:, 2] - gts_arr[:, 0]) * (gts_arr[:, 3] - gts_arr[:, 1])
                union = p_area + gt_areas - inter
                ious = inter / np.maximum(union, 1e-7)

                best_gt_idx = np.argmax(ious)
                best_iou = ious[best_gt_idx]

                if best_iou >= iou_thr:
                    if not detected[img_idx][best_gt_idx]:
                        tp[p_idx] = 1
                        detected[img_idx][best_gt_idx] = True
                    else:
                        fp[p_idx] = 1 # Duplicate detection
                else:
                    fp[p_idx] = 1

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / num_gt
            precisions = cum_tp / (cum_tp + cum_fp + 1e-7)

            ap = compute_ap_curve(recalls, precisions)
            per_class_aps[c][iou_thr] = ap

    # Summary metrics
    map_50 = np.nanmean([per_class_aps[c][0.50] for c in range(num_classes)])
    map_75 = np.nanmean([per_class_aps[c].get(0.75, 0.0) for c in range(num_classes)])
    map_90 = np.nanmean([per_class_aps[c].get(0.90, 0.0) for c in range(num_classes)])
    
    all_maps = []
    for iou_thr in iou_thresholds:
        all_maps.append(np.nanmean([per_class_aps[c][iou_thr] for c in range(num_classes)]))
    map_50_95 = float(np.mean(all_maps))

    return {
        "mAP_50": float(map_50),
        "mAP_75": float(map_75),
        "mAP_90": float(map_90),
        "mAP_50_95": float(map_50_95),
        "per_class_aps": per_class_aps,
        "iou_thresholds": iou_thresholds,
        "total_gt_per_class": total_gts_per_class
    }
