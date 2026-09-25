"""
Post-processing operations for single-model inference:
- Letterbox coordinate unpadding and scaling
- Coordinate boundary clipping
- Confidence score thresholding
- Single-model class-aware NMS
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import torch
from torchvision.ops import nms


def unpad_and_scale_boxes(
    boxes: np.ndarray,
    pad_info: Tuple[int, int, float],
    orig_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Inverts letterbox transformation:
    pad_info: (pad_left, pad_top, scale)
    orig_shape: (orig_h, orig_w)
    """
    if len(boxes) == 0:
        return boxes

    pad_left, pad_top, scale = pad_info
    orig_h, orig_w = orig_shape

    unpadded = boxes.copy()
    unpadded[:, [0, 2]] = (unpadded[:, [0, 2]] - pad_left) / scale
    unpadded[:, [1, 3]] = (unpadded[:, [1, 3]] - pad_top) / scale

    # Clip to image boundaries
    unpadded[:, [0, 2]] = np.clip(unpadded[:, [0, 2]], 0, orig_w)
    unpadded[:, [1, 3]] = np.clip(unpadded[:, [1, 3]], 0, orig_h)
    return unpadded


def postprocess_detections(
    predictions: Dict[str, torch.Tensor],
    pad_info: Tuple[int, int, float],
    orig_shape: Tuple[int, int],
    conf_threshold: float = 0.05,
    nms_iou_threshold: float = 0.50,
    max_detections: int = 100
) -> Dict[str, np.ndarray]:
    """
    Decodes and postprocesses raw model output dictionary into final predictions for a single image.
    """
    decoded_boxes = predictions["decoded_boxes"][0] # [N, 4]
    cls_logits = predictions["cls_logits"][0]       # [N, num_classes]
    obj_logits = predictions["obj_logits"][0]       # [N, 1]

    cls_scores = torch.sigmoid(cls_logits)
    obj_scores = torch.sigmoid(obj_logits)
    scores = obj_scores * cls_scores # [N, num_classes]

    max_scores, class_ids = torch.max(scores, dim=-1) # [N], [N]
    keep = max_scores >= conf_threshold

    filt_boxes = decoded_boxes[keep]
    filt_scores = max_scores[keep]
    filt_classes = class_ids[keep]

    if filt_boxes.shape[0] == 0:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64)
        }

    # Class-aware NMS
    max_coord = filt_boxes.max() + 1.0
    offsets = filt_classes.float() * max_coord
    boxes_for_nms = filt_boxes + offsets.unsqueeze(1)

    keep_idx = nms(boxes_for_nms, filt_scores, nms_iou_threshold)[:max_detections]

    final_boxes = filt_boxes[keep_idx].cpu().numpy()
    final_scores = filt_scores[keep_idx].cpu().numpy()
    final_labels = filt_classes[keep_idx].cpu().numpy()

    # Invert letterbox
    final_boxes = unpad_and_scale_boxes(final_boxes, pad_info, orig_shape)

    return {
        "boxes": final_boxes,
        "scores": final_scores,
        "labels": final_labels
    }
