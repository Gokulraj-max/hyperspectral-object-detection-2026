"""
Spatial Backbone:
Extracts multi-scale spatial representations (P3: stride 8, P4: stride 16, P5: stride 32).
Supports 3-channel (RGB baseline) and 16-channel (Hyperspectral baseline) inputs.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, k=3, s=1, p=1)
        self.conv2 = ConvBNAct(channels, channels, k=3, s=1, p=1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class SpatialBackbone(nn.Module):
    """
    Modular CNN Backbone producing C3 (stride 8), C4 (stride 16), C5 (stride 32).
    Channels: C3 -> 128, C4 -> 256, C5 -> 512.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Stem: stride 2
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base_channels // 2, k=3, s=2, p=1),  # /2
            ConvBNAct(base_channels // 2, base_channels, k=3, s=1, p=1)
        )

        # Stage 1: stride 4
        self.stage1 = nn.Sequential(
            ConvBNAct(base_channels, base_channels, k=3, s=2, p=1),     # /4
            ResidualBlock(base_channels)
        )

        # Stage 2 (C3): stride 8
        c3_channels = base_channels * 2  # 128
        self.stage2 = nn.Sequential(
            ConvBNAct(base_channels, c3_channels, k=3, s=2, p=1),      # /8
            ResidualBlock(c3_channels),
            ResidualBlock(c3_channels)
        )

        # Stage 3 (C4): stride 16
        c4_channels = base_channels * 4  # 256
        self.stage3 = nn.Sequential(
            ConvBNAct(c3_channels, c4_channels, k=3, s=2, p=1),        # /16
            ResidualBlock(c4_channels),
            ResidualBlock(c4_channels),
            ResidualBlock(c4_channels)
        )

        # Stage 4 (C5): stride 32
        c5_channels = base_channels * 8  # 512
        self.stage4 = nn.Sequential(
            ConvBNAct(c4_channels, c5_channels, k=3, s=2, p=1),        # /32
            ResidualBlock(c5_channels),
            ResidualBlock(c5_channels)
        )

        self.out_channels = [c3_channels, c4_channels, c5_channels]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5
