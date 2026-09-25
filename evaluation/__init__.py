"""
Evaluation module initialization.
"""

from .metrics import compute_ap_curve, evaluate_detections
from .per_class import format_per_class_table, analyze_real_vs_counterfeit
from .confusion_matrix import DetectionConfusionMatrix
from .error_analysis import analyze_errors, visualize_error_diagnostics

__all__ = [
    "compute_ap_curve",
    "evaluate_detections",
    "format_per_class_table",
    "analyze_real_vs_counterfeit",
    "DetectionConfusionMatrix",
    "analyze_errors",
    "visualize_error_diagnostics"
]
