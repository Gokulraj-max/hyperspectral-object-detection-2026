"""
Unit tests for Hyperspectral Dataset, VOC XML Parser, Transforms, and Collation.
"""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import torch
import pytest

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import (
    HyperspectralDataset,
    HyperspectralAugmentation,
    detection_collate_fn,
    render_pseudo_rgb,
    CLASS_NAMES
)
from datasets.voc_parser import parse_voc_annotation
from datasets.transforms import BandAwareNormalize, LetterboxResize


def test_voc_parser_and_synthetic_dataset():
    # 1. Test VOC XML Parser with mock xml
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        xml_content = """<annotation>
            <filename>mock_001.tif</filename>
            <size><width>512</width><height>512</height><depth>16</depth></size>
            <object>
                <name>real_leather</name>
                <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>100</xmax><ymax>120</ymax></bndbox>
            </object>
            <object>
                <name>fake_leather</name>
                <bndbox><xmin>150</xmin><ymin>160</ymin><xmax>250</xmax><ymax>280</ymax></bndbox>
            </object>
        </annotation>"""
        f.write(xml_content)
        xml_path = f.name

    try:
        parsed = parse_voc_annotation(xml_path)
        assert parsed["image_id"] == "mock_001"
        assert len(parsed["boxes"]) == 2
        assert parsed["labels"] == [0, 1] # real_leather is 0, fake_leather is 1
        assert parsed["boxes"][0] == [10.0, 20.0, 100.0, 120.0]
    finally:
        if os.path.exists(xml_path):
            os.remove(xml_path)


def test_hyperspectral_dataset_synthetic():
    # Test synthetic generation mode (16-band)
    transforms = HyperspectralAugmentation.build_train_transforms(image_size=(256, 256))
    ds = HyperspectralDataset(
        data_dir="data/raw",
        mode="16band",
        transforms=transforms,
        image_size=(256, 256),
        synthetic_count=4
    )
    assert len(ds) == 4
    img, target = ds[0]
    # Check tensor shape [C, H, W]
    assert img.shape == (16, 256, 256)
    assert isinstance(img, torch.Tensor)
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].shape[1] == 4


def test_band_aware_normalization():
    # Test band-aware normalization
    dummy_cube = np.random.uniform(10.0, 50.0, size=(64, 64, 16)).astype(np.float32)
    normalizer = BandAwareNormalize()
    norm_cube, _ = normalizer(dummy_cube, {})
    # Check mean across each band is near 0 and std is near 1
    for b in range(16):
        b_slice = norm_cube[:, :, b]
        assert abs(b_slice.mean()) < 0.2
        assert abs(b_slice.std() - 1.0) < 0.2


def test_detection_collate():
    # Test collate function
    img1 = torch.randn(16, 256, 256)
    t1 = {"boxes": torch.randn(2, 4), "labels": torch.tensor([0, 1])}
    img2 = torch.randn(16, 256, 256)
    t2 = {"boxes": torch.randn(5, 4), "labels": torch.tensor([2, 3, 4, 5, 6])}

    batch = [(img1, t1), (img2, t2)]
    imgs, targets = detection_collate_fn(batch)
    assert imgs.shape == (2, 16, 256, 256)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (2, 4)
    assert targets[1]["boxes"].shape == (5, 4)


def test_pseudo_rgb_render():
    cube = np.random.uniform(0.0, 1.0, size=(64, 64, 16)).astype(np.float32)
    rgb = render_pseudo_rgb(cube)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
