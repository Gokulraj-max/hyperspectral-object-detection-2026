"""
Hybrid Backbone for HS-SAFD (Hyperspectral Spectral-Attention Fusion Detector):
Combines Spectral Projection (1x1 + 3x3) -> Spectral Attention -> Spatial Attention -> Multi-scale CNN stages.
Outputs C3 (stride 8), C4 (stride 16), C5 (stride 32).
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
from ..modules.spectral_projection import SpectralProjection
from ..modules.spectral_attention import SpectralAttention
from ..modules.spatial_attention import SpatialAttention
from .spatial_backbone import ConvBNAct, ResidualBlock


class HybridBackbone(nn.Module):
    """
    16-band HSI
        │
        ▼
    Spectral Projection (1x1 Conv 16->32, 3x3 Conv 32->64)
        │
        ▼
    Spectral Attention (band importance weighting)
        │
        ▼
    Spatial Attention (foreground region focusing)
        │
        ├────────────────┐
        ▼                │
    Stage 2 (C3: /8)     │
        │                │
        ▼                │
    Stage 3 (C4: /16)    │
        │                │
        ▼                │
    Stage 4 (C5: /32)    │
    """
    def __init__(
        self,
        in_channels: int = 16,
        base_channels: int = 64,
        use_spectral_proj: bool = True,
        use_spectral_att: bool = True,
        use_spatial_att: bool = True
    ):
        super().__init__()
        self.use_spectral_proj = use_spectral_proj
        self.use_spectral_att = use_spectral_att
        self.use_spatial_att = use_spatial_att

        # Spectral Projection
        if use_spectral_proj:
            self.spectral_proj = SpectralProjection(
                in_channels=in_channels,
                mid_channels=base_channels // 2,
                out_channels=base_channels
            )
            proj_channels = base_channels
        else:
            self.spectral_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
            proj_channels = base_channels

        # Spectral Attention
        self.spectral_attention = SpectralAttention(in_channels=proj_channels, reduction=4) if use_spectral_att else nn.Identity()

        # Spatial Attention
        self.spatial_attention = SpatialAttention(kernel_size=7) if use_spatial_att else nn.Identity()

        # Stem downsampler: stride 4
        self.down_stem = nn.Sequential(
            ConvBNAct(proj_channels, base_channels, k=3, s=2, p=1),  # /2
            ConvBNAct(base_channels, base_channels, k=3, s=2, p=1),  # /4
            ResidualBlock(base_channels)
        )

        # Stage 2 (C3): stride 8
        c3_channels = base_channels * 2  # 128
        self.stage2 = nn.Sequential(
            ConvBNAct(base_channels, c3_channels, k=3, s=2, p=1),   # /8
            ResidualBlock(c3_channels),
            ResidualBlock(c3_channels)
        )

        # Stage 3 (C4): stride 16
        c4_channels = base_channels * 4  # 256
        self.stage3 = nn.Sequential(
            ConvBNAct(c3_channels, c4_channels, k=3, s=2, p=1),     # /16
            ResidualBlock(c4_channels),
            ResidualBlock(c4_channels),
            ResidualBlock(c4_channels)
        )

        # Stage 4 (C5): stride 32
        c5_channels = base_channels * 8  # 512
        self.stage4 = nn.Sequential(
            ConvBNAct(c4_channels, c5_channels, k=3, s=2, p=1),     # /32
            ResidualBlock(c5_channels),
            ResidualBlock(c5_channels)
        )

        self.out_channels = [c3_channels, c4_channels, c5_channels]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Step 1: Spectral Projection
        x_proj = self.spectral_proj(x)

        # Step 2: Spectral Attention
        x_spec = self.spectral_attention(x_proj)

        # Step 3: Spatial Attention
        x_spat = self.spatial_attention(x_spec)

        # Step 4: Multi-scale spatial hierarchy
        feat = self.down_stem(x_spat)
        c3 = self.stage2(feat)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return c3, c4, c5
