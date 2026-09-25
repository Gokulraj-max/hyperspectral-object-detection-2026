"""
Models package initialization.
"""

from .hyperspectral_detector import HyperspectralDetector, HS_SAFD, build_detector
from .modules.spectral_projection import SpectralProjection
from .backbone import SpatialBackbone, SpectralBackbone, HybridBackbone
from .attention import SpectralAttention, SpatialAttention, ChannelAttention, CBAM
from .neck.feature_pyramid import FeaturePyramidNetwork, FPN, PAN
from .head.detection_head import DecoupledHead, MultiScaleDetectionHead

__all__ = [
    "HyperspectralDetector",
    "HS_SAFD",
    "build_detector",
    "SpectralProjection",
    "SpatialBackbone",
    "SpectralBackbone",
    "HybridBackbone",
    "SpectralAttention",
    "SpatialAttention",
    "ChannelAttention",
    "CBAM",
    "FeaturePyramidNetwork",
    "FPN",
    "PAN",
    "DecoupledHead",
    "MultiScaleDetectionHead"
]
