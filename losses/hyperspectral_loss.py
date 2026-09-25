"""
Combined Hyperspectral Multi-Task Loss Function (Section 13):
Total Loss = lambda1 * cls_loss + lambda2 * box_loss (CIoU) + lambda3 * objectness_loss
Optimizes for high-precision bounding boxes (AP50, AP75, AP90).
"""

from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classification_loss import FocalLoss
from .iou_loss import calculate_ciou, CIoULoss


class HyperspectralLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 18,
        cls_weight: float = 1.0,
        box_weight: float = 7.5,
        objectness_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        center_radius: float = 1.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.objectness_weight = objectness_weight
        self.center_radius = center_radius

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="sum")
        self.ciou_loss = CIoULoss()
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        predictions: dict with decoded_boxes, cls_logits, obj_logits, grid_points, strides
        targets: list of target dicts for each image in batch (each has 'boxes' and 'labels')
        """
        cls_logits = predictions["cls_logits"]       # [B, N, num_classes]
        decoded_boxes = predictions["decoded_boxes"] # [B, N, 4]
        obj_logits = predictions["obj_logits"]       # [B, N, 1]
        grid_points = predictions["grid_points"]     # [N, 2]
        strides = predictions["strides"]             # [N, 1]

        batch_size = cls_logits.shape[0]
        device = cls_logits.device

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)

        total_positives = 0

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"].to(device)   # [M, 4]
            gt_labels = targets[b]["labels"].to(device) # [M]
            num_gt = gt_boxes.shape[0]

            b_cls = cls_logits[b]       # [N, num_classes]
            b_boxes = decoded_boxes[b]  # [N, 4]
            b_obj = obj_logits[b]       # [N, 1]

            target_obj = torch.zeros_like(b_obj)
            target_cls = torch.zeros_like(b_cls)

            if num_gt == 0:
                # No objects in this image
                obj_loss = self.bce_obj(b_obj, target_obj)
                total_obj_loss += obj_loss
                continue

            # Assign GT boxes to grid points
            # Points inside GT box + center sampling
            pts_x = grid_points[:, 0]
            pts_y = grid_points[:, 1]

            # Pairwise distance from points to GT centers
            gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
            gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

            dx = pts_x[:, None] - gt_cx[None, :] # [N, M]
            dy = pts_y[:, None] - gt_cy[None, :] # [N, M]
            dist = torch.sqrt(dx ** 2 + dy ** 2) # [N, M]

            # Candidate match: point inside GT box
            in_x = (pts_x[:, None] >= gt_boxes[None, :, 0]) & (pts_x[:, None] <= gt_boxes[None, :, 2])
            in_y = (pts_y[:, None] >= gt_boxes[None, :, 1]) & (pts_y[:, None] <= gt_boxes[None, :, 3])
            inside_box = in_x & in_y # [N, M]

            # Center sampling within radius * stride
            max_dist = strides * self.center_radius
            near_center = dist <= max_dist # [N, M]

            valid_candidates = inside_box | near_center # [N, M]

            # For each candidate point, find best matching GT (closest center)
            dist_masked = torch.where(valid_candidates, dist, torch.tensor(1e6, device=device))
            min_dist, matched_gt_idx = torch.min(dist_masked, dim=1) # [N]

            pos_mask = min_dist < 1e5 # [N]
            num_pos = pos_mask.sum().item()

            if num_pos > 0:
                matched_gt = matched_gt_idx[pos_mask]
                pos_pred_boxes = b_boxes[pos_mask]
                pos_gt_boxes = gt_boxes[matched_gt]
                pos_labels = gt_labels[matched_gt]

                # Box regression CIoU loss
                box_loss = self.ciou_loss(pos_pred_boxes, pos_gt_boxes)
                total_box_loss += box_loss * num_pos

                # Classification focal loss
                target_cls[pos_mask, pos_labels] = 1.0
                cls_loss = self.focal_loss(b_cls[pos_mask], target_cls[pos_mask])
                total_cls_loss += cls_loss

                # Objectness target
                target_obj[pos_mask] = 1.0
                total_positives += num_pos

            obj_loss = self.bce_obj(b_obj, target_obj)
            total_obj_loss += obj_loss

        normalizer = max(1, total_positives)
        mean_cls = total_cls_loss / normalizer
        mean_box = total_box_loss / normalizer
        mean_obj = total_obj_loss / batch_size

        total_loss = (
            self.cls_weight * mean_cls +
            self.box_weight * mean_box +
            self.objectness_weight * mean_obj
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_cls": mean_cls.item(),
            "loss_box": mean_box.item(),
            "loss_obj": mean_obj.item(),
            "num_positives": total_positives
        }

        return total_loss, loss_dict
