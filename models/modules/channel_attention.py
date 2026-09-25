"""
Channel Attention and CBAM-style Modules.
"""

import torch
import torch.nn as nn
from .spectral_attention import SpectralAttention
from .spatial_attention import SpatialAttention


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module combining Spectral/Channel and Spatial attention.
    """
    def __init__(self, in_channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.channel_att = SpectralAttention(in_channels=in_channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
