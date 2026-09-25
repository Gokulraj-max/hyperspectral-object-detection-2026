"""
Unit tests for HS-SAFD Model Architecture & Attention Modules.
"""

import os
import sys
import torch
import pytest

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (
    HS_SAFD,
    SpectralProjection,
    SpectralAttention,
    SpatialAttention,
    HybridBackbone,
    SpatialBackbone,
    FeaturePyramidNetwork,
    DecoupledHead,
    MultiScaleDetectionHead
)
from losses import DetectionLoss, calculate_ciou


def test_spectral_projection():
    x = torch.randn(2, 16, 64, 64)
    proj = SpectralProjection(in_channels=16, mid_channels=32, out_channels=64)
    out = proj(x)
    assert out.shape == (2, 64, 64, 64)


def test_spectral_attention():
    x = torch.randn(2, 16, 64, 64)
    sa = SpectralAttention(in_channels=16, reduction=4)
    out = sa(x)
    assert out.shape == (2, 16, 64, 64)


def test_spatial_attention():
    x = torch.randn(2, 64, 64, 64)
    spa = SpatialAttention(kernel_size=7)
    out = spa(x)
    assert out.shape == (2, 64, 64, 64)


def test_hybrid_backbone():
    x = torch.randn(2, 16, 128, 128)
    backbone = HybridBackbone(in_channels=16, base_channels=64)
    c3, c4, c5 = backbone(x)
    # Stride 8, 16, 32
    assert c3.shape == (2, 128, 16, 16)
    assert c4.shape == (2, 256, 8, 8)
    assert c5.shape == (2, 512, 4, 4)


def test_feature_pyramid():
    c3 = torch.randn(2, 128, 16, 16)
    c4 = torch.randn(2, 256, 8, 8)
    c5 = torch.randn(2, 512, 4, 4)

    fpn = FeaturePyramidNetwork(in_channels_list=[128, 256, 512], out_channels=128, neck_type="pan")
    p3, p4, p5 = fpn((c3, c4, c5))
    assert p3.shape == (2, 128, 16, 16)
    assert p4.shape == (2, 128, 8, 8)
    assert p5.shape == (2, 128, 4, 4)


def test_hs_safd_forward_and_loss():
    model = HS_SAFD(num_classes=18, head_channels=64)
    x = torch.randn(2, 16, 128, 128)
    outputs = model(x)

    assert "decoded_boxes" in outputs
    assert "cls_logits" in outputs
    assert "obj_logits" in outputs
    assert outputs["decoded_boxes"].shape[0] == 2
    assert outputs["cls_logits"].shape[-1] == 18

    # Test Loss Computation
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([2])},
        {"boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]]), "labels": torch.tensor([5])}
    ]
    loss_fn = DetectionLoss(num_classes=18)
    total_loss, loss_dict = loss_fn(outputs, targets)

    assert total_loss.item() > 0.0
    assert "loss_cls" in loss_dict
    assert "loss_box" in loss_dict
    assert "loss_obj" in loss_dict


def test_ciou_calculation():
    b1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    b2 = torch.tensor([[10.0, 10.0, 50.0, 50.0]]) # Identical
    ciou_identical = calculate_ciou(b1, b2)
    assert abs(ciou_identical.item() - 1.0) < 1e-4

    b3 = torch.tensor([[100.0, 100.0, 150.0, 150.0]]) # Far away
    ciou_far = calculate_ciou(b1, b3)
    assert ciou_far.item() < 0.0
