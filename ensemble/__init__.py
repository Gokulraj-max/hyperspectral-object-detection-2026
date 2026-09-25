"""
Ensemble module initialization.
"""

from .weighted_boxes import EnsembleBoxFuser
from .model_fusion import ModelFusionEnsemble

__all__ = [
    "EnsembleBoxFuser",
    "ModelFusionEnsemble"
]
