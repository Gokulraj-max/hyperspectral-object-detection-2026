"""
Feature Pyramid Network (FPN) Neck:
Fuses multi-scale features via top-down pathways.
Produces P3 (stride 8), P4 (stride 16), P5 (stride 32).
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 128):
        super().__init__()
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)

        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.out_channels = out_channels

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features

        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)

        return p3, p4, p5
