"""
Losses module initialization.
"""

from .classification_loss import FocalLoss, LabelSmoothingCrossEntropy
from .iou_loss import box_iou, calculate_ciou, CIoULoss
from .bbox_loss import BBoxLoss
from .hyperspectral_loss import HyperspectralLoss
from .detection_loss import DetectionLoss

__all__ = [
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "box_iou",
    "calculate_ciou",
    "CIoULoss",
    "BBoxLoss",
    "HyperspectralLoss",
    "DetectionLoss"
]
