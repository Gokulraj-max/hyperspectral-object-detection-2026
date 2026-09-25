"""
Spectral 1D/3D Backbone:
Extracts fine-grained spectral signatures along the 16 bands prior to spatial pooling.
"""

from typing import Tuple
import torch
import torch.nn as nn
from ..modules.spectral_attention import SpectralAttention


class SpectralBackbone(nn.Module):
    """
    Dedicated spectral feature extractor treating the 16 bands with 1D/3D spectral kernels.
    Extracts continuous absorption and emission profiles characteristic of distinct materials.
    """
    def __init__(self, in_channels: int = 16, out_channels: int = 64):
        super().__init__()
        # Input shape: [B, 1, 16, H, W] for 3D conv or [B, 16, H, W] with 1x1 depthwise
        self.spectral_conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        self.act1 = nn.SiLU(inplace=True)

        self.spectral_attention = SpectralAttention(in_channels=in_channels * 2, reduction=4)

        self.spectral_conv2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 16, H, W]
        feat = self.act1(self.bn1(self.spectral_conv1(x)))
        feat = self.spectral_attention(feat)
        out = self.act2(self.bn2(self.spectral_conv2(feat)))
        return out
