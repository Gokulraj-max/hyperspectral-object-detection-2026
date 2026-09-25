"""
Multi-Model Weighted Box Fusion Wrapper (Section 21).
Combines predictions across diverse detector architectures:
- Model A: RGB baseline
- Model B: 16-band baseline
- Model C: PCA detector
- Model D: HS-SAFD (Spectral attention)
- Model E: High-resolution detector
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from inference.wbf import weighted_box_fusion


class EnsembleBoxFuser:
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        iou_thr: float = 0.55,
        skip_box_thr: float = 0.05,
        conf_type: str = "avg"
    ):
        self.weights = weights
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.conf_type = conf_type

    def fuse(
        self,
        model_predictions: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        model_predictions: List of dicts, one per model:
          [{'boxes': np.ndarray, 'scores': np.ndarray, 'labels': np.ndarray}, ...]
        """
        boxes_list = [np.asarray(m["boxes"], dtype=np.float32) for m in model_predictions]
        scores_list = [np.asarray(m["scores"], dtype=np.float32) for m in model_predictions]
        labels_list = [np.asarray(m["labels"], dtype=np.int64) for m in model_predictions]

        f_boxes, f_scores, f_labels = weighted_box_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=self.weights,
            iou_thr=self.iou_thr,
            skip_box_thr=self.skip_box_thr,
            conf_type=self.conf_type
        )

        return {
            "boxes": f_boxes,
            "scores": f_scores,
            "labels": f_labels
        }
