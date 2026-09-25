"""
IoU, GIoU, DIoU, and CIoU (Complete IoU) Loss Functions.
CIoU considers overlap area, distance between central points, and aspect ratio consistency.
Directly optimizes for AP50, AP75, and AP90.
"""

import math
import torch
import torch.nn as nn


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise IoU between two sets of boxes [N, 4] and [M, 4] in [x1, y1, x2, y2].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N, M, 2]

    wh = (rb - lt).clamp(min=0) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)
    return iou


def calculate_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes element-wise Complete IoU (CIoU) between box1 and box2 of shape [N, 4] (x1, y1, x2, y2).
    Returns CIoU values in [-1, 1].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas
    w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)
    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area

    # Center points distance
    cx1 = (b1_x1 + b1_x2) / 2.0
    cy1 = (b1_y1 + b1_y2) / 2.0
    cx2 = (b2_x1 + b2_x2) / 2.0
    cy2 = (b2_y1 + b2_y2) / 2.0
    rho2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    # Enclosing smallest box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c_diag2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Aspect ratio penalty
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / ((1.0 - iou) + v + eps)

    ciou = iou - (rho2 / c_diag2) - alpha * v
    return ciou


class CIoULoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Calculates CIoU Loss = 1.0 - CIoU.
        """
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        ciou = calculate_ciou(pred_boxes, target_boxes, self.eps)
        loss = 1.0 - ciou
        return loss.mean()
