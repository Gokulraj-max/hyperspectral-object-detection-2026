"""
Hyperspectral Object Detection Challenge 2026 - Datasets Module
"""

CLASS_NAMES = [
    "real_leather", "fake_leather",
    "real_silk", "synthetic_silk",
    "real_gemstone", "glass_gemstone",
    "real_banknote", "counterfeit_banknote",
    "real_wood", "laminated_wood",
    "real_wool", "synthetic_wool",
    "real_jade", "counterfeit_jade",
    "authentic_chip", "counterfeit_chip",
    "natural_honey", "adulterated_honey"
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Paired real vs counterfeit categories for domain analysis
PAIRED_CATEGORIES = [
    (0, 1),    # real_leather vs fake_leather
    (2, 3),    # real_silk vs synthetic_silk
    (4, 5),    # real_gemstone vs glass_gemstone
    (6, 7),    # real_banknote vs counterfeit_banknote
    (8, 9),    # real_wood vs laminated_wood
    (10, 11),  # real_wool vs synthetic_wool
    (12, 13),  # real_jade vs counterfeit_jade
    (14, 15),  # authentic_chip vs counterfeit_chip
    (16, 17)   # natural_honey vs adulterated_honey
]

# 16 spectral bands between 460 nm and 600 nm
WAVELENGTHS = [
    460.0, 469.3, 478.7, 488.0, 497.3, 506.7, 516.0, 525.3,
    534.7, 544.0, 553.3, 562.7, 572.0, 581.3, 590.7, 600.0
]

from .annotation_parser import parse_annotations, convert_bbox
from .voc_parser import parse_voc_annotation, parse_voc_directory
from .transforms import Compose, HyperspectralAugmentation, BandAwareNormalize
from .collate import detection_collate_fn
from .hyperspectral_dataset import HyperspectralDataset
from .visualization import (
    render_pseudo_rgb,
    render_false_color,
    plot_spectral_signature,
    draw_bounding_boxes
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_ID",
    "ID_TO_CLASS",
    "PAIRED_CATEGORIES",
    "WAVELENGTHS",
    "parse_annotations",
    "convert_bbox",
    "Compose",
    "HyperspectralAugmentation",
    "BandAwareNormalize",
    "detection_collate_fn",
    "HyperspectralDataset",
    "render_pseudo_rgb",
    "render_false_color",
    "plot_spectral_signature",
    "draw_bounding_boxes"
]
