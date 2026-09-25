"""
Spectral Attention Module (Section 10):
Learns adaptive importance weights across spectral wavelengths/channels via Global Pooling + MLP + Sigmoid.
"""

import torch
import torch.nn as nn


class SpectralAttention(nn.Module):
    """
    [B, C, H, W]
         |
    Global Average Pooling -> [B, C]
         |
    MLP (C -> C // reduction -> C)
         |
    Sigmoid -> [B, C, 1, 1] weights
         |
    Output = X * weights
    """
    def __init__(self, in_channels: int = 16, reduction: int = 4):
        super().__init__()
        mid_channels = max(4, in_channels // reduction)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # [B, C, 1, 1] -> [B, C]
        y = self.gap(x).view(b, c)
        weights = self.mlp(y).view(b, c, 1, 1)
        return x * weights
