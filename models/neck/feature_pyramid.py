"""
Multi-Scale Feature Pyramid Network (FPN / PAN) for Hyperspectral Detection.
Provides enhanced semantic and localization features for P3, P4, P5 levels.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fpn import FPN
from .pan import PAN


class FeaturePyramidNetwork(nn.Module):
    """
    Unified Feature Pyramid Interface supporting both FPN and PAN modes.
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 128, neck_type: str = "pan"):
        super().__init__()
        self.neck_type = neck_type.lower()
        if self.neck_type == "pan":
            self.neck = PAN(in_channels_list=in_channels_list, out_channels=out_channels)
        else:
            self.neck = FPN(in_channels_list=in_channels_list, out_channels=out_channels)
        self.out_channels = out_channels

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.neck(features)
