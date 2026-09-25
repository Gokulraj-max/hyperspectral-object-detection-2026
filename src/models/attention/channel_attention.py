"""
Channel Attention & CBAM.
"""

import torch
import torch.nn as nn
from .spectral_attention import SpectralAttention
from .spatial_attention import SpatialAttention


class ChannelAttention(SpectralAttention):
    """Alias for channel-wise squeeze and excitation attention."""
    pass


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels=in_channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))
