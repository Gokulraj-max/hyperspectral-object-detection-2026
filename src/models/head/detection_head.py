"""
Detection Head for 18-class Hyperspectral Object Detection.
Exports DecoupledHead and MultiScaleDetectionHead.
"""

from ..detector.detection_head import DecoupledHead, MultiScaleDetectionHead

__all__ = ["DecoupledHead", "MultiScaleDetectionHead"]
