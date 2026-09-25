"""
Path Aggregation Network (PAN) Neck:
Enhances feature hierarchy with top-down and bottom-up pathways.
Crucial for high IoU localization (AP75, AP90).
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PAN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 128):
        super().__init__()
        c3_c, c4_c, c5_c = in_channels_list

        # Top-down pathway (FPN)
        self.lateral_c5 = nn.Conv2d(c5_c, out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(c4_c, out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(c3_c, out_channels, kernel_size=1)

        self.smooth_c4 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)
        self.smooth_c3 = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)

        # Bottom-up pathway (PAN)
        self.down_p3 = ConvBNAct(out_channels, out_channels, k=3, s=2, p=1)
        self.pan_n4 = ConvBNAct(out_channels * 2, out_channels, k=3, s=1, p=1)

        self.down_n4 = ConvBNAct(out_channels, out_channels, k=3, s=2, p=1)
        self.pan_n5 = ConvBNAct(out_channels * 2, out_channels, k=3, s=1, p=1)

        self.out_channels = out_channels

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features

        # Top-down
        p5 = self.lateral_c5(c5)
        p4 = self.smooth_c4(self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest"))
        p3 = self.smooth_c3(self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest"))

        # Bottom-up
        n3 = p3
        n4 = self.pan_n4(torch.cat([self.down_p3(n3), p4], dim=1))
        n5 = self.pan_n5(torch.cat([self.down_n4(n4), p5], dim=1))

        # Returns P3 (small objects), P4 (medium objects), P5 (large objects)
        return n3, n4, n5
