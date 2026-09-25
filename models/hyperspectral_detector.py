"""
Hyperspectral Detector Module:
HS-SAFD (Hyperspectral Spectral-Attention Fusion Detector) & Baseline Detectors.
Compliant with Competition Rules: Strictly a single standalone detector model.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from .detector.hyperspectral_detector import HyperspectralDetector, build_detector


class HS_SAFD(HyperspectralDetector):
    """
    Hyperspectral Spectral-Attention Fusion Detector (HS-SAFD).
    Pre-configured with:
    - 16-band input
    - Spectral projection (1x1 conv 16->32, 3x3 conv 32->64)
    - Spectral attention
    - Spatial attention
    - Multi-scale spatial backbone
    - PAN neck
    - Decoupled 18-class detection heads
    """
    def __init__(self, num_classes: int = 18, head_channels: int = 128):
        super().__init__(
            mode="hs_safd",
            input_channels=16,
            num_classes=num_classes,
            backbone_type="hybrid",
            neck_type="pan",
            neck_channels=head_channels,
            head_channels=head_channels,
            use_spectral_proj=True,
            use_spectral_att=True,
            use_spatial_att=True
        )


__all__ = ["HyperspectralDetector", "HS_SAFD", "build_detector"]
