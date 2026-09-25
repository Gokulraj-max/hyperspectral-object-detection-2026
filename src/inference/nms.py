"""
Non-Maximum Suppression (Standard & Soft-NMS).
"""

from typing import Tuple
import torch
from torchvision.ops import nms as tv_nms


def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.50
) -> torch.Tensor:
    """Standard Greedy NMS."""
    return tv_nms(boxes, scores, iou_threshold)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.50,
    sigma: float = 0.5,
    score_threshold: float = 0.05,
    method: str = "gaussian"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-NMS: Decays confidence score of overlapping boxes rather than hard removal.
    """
    if boxes.shape[0] == 0:
        return boxes, scores

    b = boxes.clone()
    s = scores.clone()
    n = b.shape[0]

    for i in range(n):
        max_idx = torch.argmax(s[i:]) + i
        # Swap
        if max_idx != i:
            b_temp = b[i].clone()
            b[i] = b[max_idx]
            b[max_idx] = b_temp

            s_temp = s[i].clone()
            s[i] = s[max_idx]
            s[max_idx] = s_temp

        # Compute IoU between box i and rest
        ix1 = torch.clamp(b[i + 1:, 0], min=b[i, 0])
        iy1 = torch.clamp(b[i + 1:, 1], min=b[i, 1])
        ix2 = torch.clamp(b[i + 1:, 2], max=b[i, 2])
        iy2 = torch.clamp(b[i + 1:, 3], max=b[i, 3])

        iw = torch.clamp(ix2 - ix1, min=0.0)
        ih = torch.clamp(iy2 - iy1, min=0.0)
        inter = iw * ih

        area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        area_rest = (b[i + 1:, 2] - b[i + 1:, 0]) * (b[i + 1:, 3] - b[i + 1:, 1])
        union = area_i + area_rest - inter
        ious = inter / torch.clamp(union, min=1e-7)

        # Decay scores
        if method == "gaussian":
            decay = torch.exp(-(ious ** 2) / sigma)
        else: # linear
            decay = torch.where(ious >= iou_threshold, 1.0 - ious, torch.ones_like(ious))

        s[i + 1:] = s[i + 1:] * decay

    keep = s >= score_threshold
    return b[keep], s[keep]
