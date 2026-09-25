"""
Confusion Matrix Generator for Object Detection (18 Classes + Background).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import CLASS_NAMES
from losses.iou_loss import box_iou


class DetectionConfusionMatrix:
    """
    Constructs an (N+1) x (N+1) confusion matrix where index N represents Background.
    Row: Ground Truth, Column: Prediction.
    """
    def __init__(self, num_classes: int = 18, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        # Matrix shape: [num_classes + 1, num_classes + 1]
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, Any]]):
        for pred, target in zip(predictions, targets):
            p_boxes = pred["boxes"]
            p_labels = pred["labels"]
            p_scores = pred["scores"]
            
            g_boxes = target["boxes"]
            g_labels = target["labels"]

            if isinstance(p_boxes, np.ndarray):
                p_boxes = torch.from_numpy(p_boxes)
            if isinstance(p_labels, np.ndarray):
                p_labels = torch.from_numpy(p_labels)
            if isinstance(g_boxes, np.ndarray):
                g_boxes = torch.from_numpy(g_boxes)
            if isinstance(g_labels, np.ndarray):
                g_labels = torch.from_numpy(g_labels)

            num_p = p_boxes.shape[0]
            num_g = g_boxes.shape[0]

            if num_g == 0:
                for p_idx in range(num_p):
                    p_cls = int(p_labels[p_idx])
                    self.matrix[self.num_classes, p_cls] += 1 # False positive from background
                continue

            if num_p == 0:
                for g_idx in range(num_g):
                    g_cls = int(g_labels[g_idx])
                    self.matrix[g_cls, self.num_classes] += 1 # False negative missed to background
                continue

            ious = box_iou(p_boxes, g_boxes) # [num_p, num_g]

            matched_gts = set()
            for p_idx in range(num_p):
                p_cls = int(p_labels[p_idx])
                best_iou, best_g_idx = torch.max(ious[p_idx], dim=0)
                best_iou = float(best_iou)
                best_g_idx = int(best_g_idx)

                if best_iou >= self.iou_threshold:
                    g_cls = int(g_labels[best_g_idx])
                    self.matrix[g_cls, p_cls] += 1
                    matched_gts.add(best_g_idx)
                else:
                    self.matrix[self.num_classes, p_cls] += 1 # FP

            # Missed GTs
            for g_idx in range(num_g):
                if g_idx not in matched_gts:
                    g_cls = int(g_labels[g_idx])
                    self.matrix[g_cls, self.num_classes] += 1 # FN

    def plot(self, save_path: Optional[str] = None, class_names: Optional[List[str]] = None) -> plt.Figure:
        classes = (class_names or CLASS_NAMES) + ["background"]
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalized matrix
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.divide(self.matrix.astype(float), row_sums, out=np.zeros_like(self.matrix, dtype=float), where=row_sums != 0)

        im = ax.imshow(norm_matrix, cmap="Blues", interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)

        ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight="bold")
        ax.set_title("Hyperspectral Object Detection Confusion Matrix", fontsize=14, fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig
