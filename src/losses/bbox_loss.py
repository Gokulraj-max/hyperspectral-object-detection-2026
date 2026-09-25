"""
Bounding Box Regression Losses:
Smooth L1, GIoU, DIoU, CIoU wrappers.
"""

from typing import Optional
import torch
import torch.nn as nn
from .iou_loss import CIoULoss


class BBoxLoss(nn.Module):
    def __init__(self, loss_type: str = "ciou"):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == "ciou":
            self.loss_fn = CIoULoss()
        elif self.loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(beta=1.0 / 9.0)
        elif self.loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported bbox loss type: {loss_type}")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_boxes, target_boxes)
