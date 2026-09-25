"""
Detection Loss Module:
Combines CIoU Box Loss, Focal Classification Loss, and Objectness Loss.
Directly optimizes for mAP@[0.50:0.95].
"""

from .hyperspectral_loss import HyperspectralLoss


class DetectionLoss(HyperspectralLoss):
    """Unified Detection Loss alias."""
    pass


__all__ = ["DetectionLoss"]
