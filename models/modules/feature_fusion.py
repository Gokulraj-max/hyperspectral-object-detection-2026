"""
Feature Fusion Module for combining low-level spectral projections with deep spatial features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFeatureFusion(nn.Module):
    """
    Fuses two feature maps of potentially different channels and resolutions:
    x_spectral (e.g. [B, C1, H1, W1]) and x_spatial (e.g. [B, C2, H2, W2]).
    Learns dynamic attention gating weights to balance spectral vs spatial cues.
    """
    def __init__(self, in_channels_1: int, in_channels_2: int, out_channels: int):
        super().__init__()
        self.proj1 = nn.Conv2d(in_channels_1, out_channels, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(in_channels_2, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions to x1
        h1, w1 = x1.shape[2], x1.shape[3]
        if x2.shape[2:] != (h1, w1):
            x2 = F.interpolate(x2, size=(h1, w1), mode="bilinear", align_corners=False)

        f1 = self.bn1(self.proj1(x1))
        f2 = self.bn2(self.proj2(x2))

        # Compute gating weights
        concat_feats = torch.cat([f1, f2], dim=1)
        gates = self.gate_conv(concat_feats) # [B, 2, H, W]
        w1_gate = gates[:, 0:1, :, :]
        w2_gate = gates[:, 1:2, :, :]

        fused = w1_gate * f1 + w2_gate * f2
        return self.out_conv(fused)
