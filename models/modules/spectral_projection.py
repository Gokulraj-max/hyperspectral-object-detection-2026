"""
Spectral Projection Module (Section 9):
Converts 16-band hyperspectral cubes [B, 16, H, W] into richer learned representations [B, 64, H, W]
using 1x1 spectral convolutions followed by 3x3 spatial convolutions.
"""

import torch
import torch.nn as nn


class SpectralProjection(nn.Module):
    """
    Input: [B, in_channels, H, W] (e.g. 16 bands)
    1x1 Conv -> BatchNorm -> Activation -> [B, mid_channels, H, W] (e.g. 32)
    3x3 Conv -> BatchNorm -> Activation -> [B, out_channels, H, W] (e.g. 64)
    """
    def __init__(
        self,
        in_channels: int = 16,
        mid_channels: int = 32,
        out_channels: int = 64,
        act_type: str = "silu"
    ):
        super().__init__()
        act_layer = nn.SiLU if act_type == "silu" else nn.ReLU

        self.spectral_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            act_layer(inplace=True)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(inplace=True)
        )

        # Residual shortcut if channel dimensions match or projection
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.spectral_conv(x)
        out = self.spatial_conv(out)
        return out + residual
