"""
Detector module exports.
"""

from .detection_head import DecoupledHead, MultiScaleDetectionHead
from .hyperspectral_detector import HyperspectralDetector, build_detector

__all__ = [
    "DecoupledHead",
    "MultiScaleDetectionHead",
    "HyperspectralDetector",
    "build_detector"
]
