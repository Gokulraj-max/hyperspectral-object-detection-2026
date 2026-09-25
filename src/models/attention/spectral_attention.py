"""
Spectral Attention Module (Section 7):
Wavelength-aware attention mechanism re-weighting 16 spectral channels
using Global Pooling + MLP + Sigmoid.
"""

import torch
import torch.nn as nn


class SpectralAttention(nn.Module):
    """
    16-band input [B, 16, H, W]
           │
           ▼
     Global pooling -> [B, 16]
           │
           ▼
    Spectral descriptor -> MLP (16 -> 4 -> 16)
           │
           ▼
        Sigmoid -> [B, 16, 1, 1]
           │
           ▼
    Weighted spectral features = X * weights
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
        y = self.gap(x).view(b, c)
        weights = self.mlp(y).view(b, c, 1, 1)
        return x * weights
