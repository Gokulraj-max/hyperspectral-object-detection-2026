"""
Spatial Attention Module (Section 11):
Suppresses background pixels and emphasizes object regions using inter-spatial feature relationships.
"""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Computes spatial attention mask using channel-wise AvgPool and MaxPool:
    [B, C, H, W] -> AvgPool & MaxPool -> [B, 2, H, W] -> 7x7 Conv -> Sigmoid -> [B, 1, H, W]
    Output = X * spatial_mask
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        mask = self.sigmoid(self.conv(combined))
        return x * mask
