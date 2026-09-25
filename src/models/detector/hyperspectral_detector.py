"""
Hyperspectral Object Detector (HS-SAFD & Baselines):
Supports:
- Model A: RGB 3-band Baseline Detector
- Model B: 16-band Baseline Detector
- Model C: PCA 3-band Baseline Detector
- Model D: HS-SAFD (Hyperspectral Spectral-Attention Fusion Detector)
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.ops import nms

from ..backbone.spatial_backbone import SpatialBackbone
from ..backbone.hybrid_backbone import HybridBackbone
from ..neck.fpn import FPN
from ..neck.pan import PAN
from .detection_head import MultiScaleDetectionHead


class HyperspectralDetector(nn.Module):
    def __init__(
        self,
        mode: str = "hs_safd",
        input_channels: int = 16,
        num_classes: int = 18,
        backbone_type: str = "hybrid",
        neck_type: str = "pan",
        neck_channels: int = 128,
        head_channels: int = 128,
        use_spectral_proj: bool = True,
        use_spectral_att: bool = True,
        use_spatial_att: bool = True
    ):
        super().__init__()
        self.mode = mode
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Select backbone
        if backbone_type == "hybrid" or mode == "hs_safd":
            self.backbone = HybridBackbone(
                in_channels=input_channels,
                base_channels=64,
                use_spectral_proj=use_spectral_proj,
                use_spectral_att=use_spectral_att,
                use_spatial_att=use_spatial_att
            )
        else:
            self.backbone = SpatialBackbone(
                in_channels=input_channels,
                base_channels=64
            )

        # Select neck
        backbone_out_channels = self.backbone.out_channels
        if neck_type == "pan":
            self.neck = PAN(in_channels_list=backbone_out_channels, out_channels=neck_channels)
        else:
            self.neck = FPN(in_channels_list=backbone_out_channels, out_channels=neck_channels)

        # Multi-scale detection head
        self.head = MultiScaleDetectionHead(
            in_channels=neck_channels,
            num_classes=num_classes,
            head_channels=head_channels,
            strides=[8, 16, 32]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass.
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            Dict containing cls_logits, reg_preds, obj_logits, decoded_boxes, grid_points, strides
        """
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck((c3, c4, c5))
        outputs = self.head((p3, p4, p5))
        return outputs

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_iou_threshold: float = 0.50,
        max_detections: int = 100
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Inference prediction with confidence filtering and Non-Maximum Suppression (NMS).
        Returns a list of dicts for each image in batch:
        {
            'boxes': [K, 4] (x1, y1, x2, y2),
            'scores': [K],
            'labels': [K]
        }
        """
        self.eval()
        outputs = self.forward(x)
        decoded_boxes = outputs["decoded_boxes"] # [B, N, 4]
        cls_logits = outputs["cls_logits"]       # [B, N, num_classes]
        obj_logits = outputs["obj_logits"]       # [B, N, 1]

        cls_scores = torch.sigmoid(cls_logits)
        obj_scores = torch.sigmoid(obj_logits)
        
        # Joint confidence = sqrt(obj_conf * max_cls_conf) or obj_conf * cls_conf
        scores = obj_scores * cls_scores # [B, N, num_classes]

        batch_size = x.shape[0]
        results = []

        for b in range(batch_size):
            b_boxes = decoded_boxes[b]     # [N, 4]
            b_scores = scores[b]           # [N, num_classes]

            # Find best class per candidate anchor
            max_scores, class_ids = torch.max(b_scores, dim=-1) # [N], [N]

            # Filter by confidence threshold
            keep_mask = max_scores >= conf_threshold
            filt_boxes = b_boxes[keep_mask]
            filt_scores = max_scores[keep_mask]
            filt_classes = class_ids[keep_mask]

            if filt_boxes.shape[0] == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=x.device),
                    "scores": torch.zeros((0,), device=x.device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=x.device)
                })
                continue

            # Class-aware NMS via offset trick
            max_coord = filt_boxes.max() + 1.0
            offsets = filt_classes.float() * max_coord
            boxes_for_nms = filt_boxes + offsets.unsqueeze(1)

            keep_idx = nms(boxes_for_nms, filt_scores, nms_iou_threshold)
            keep_idx = keep_idx[:max_detections]

            results.append({
                "boxes": filt_boxes[keep_idx],
                "scores": filt_scores[keep_idx],
                "labels": filt_classes[keep_idx]
            })

        return results


def build_detector(config: Dict[str, Any]) -> HyperspectralDetector:
    """Factory builder initializing detector from configuration dict."""
    m_cfg = config.get("model", {})
    mode = m_cfg.get("mode", "hs_safd")
    input_channels = m_cfg.get("input_channels", 16)
    num_classes = m_cfg.get("num_classes", 18)
    backbone_type = m_cfg.get("backbone", "hybrid")
    neck_type = m_cfg.get("neck", "pan")
    neck_channels = m_cfg.get("neck_channels", 128)
    head_channels = m_cfg.get("head_channels", 128)

    sp_cfg = m_cfg.get("spectral_projection", {})
    sa_cfg = m_cfg.get("spectral_attention", {})
    spa_cfg = m_cfg.get("spatial_attention", {})

    detector = HyperspectralDetector(
        mode=mode,
        input_channels=input_channels,
        num_classes=num_classes,
        backbone_type=backbone_type,
        neck_type=neck_type,
        neck_channels=neck_channels,
        head_channels=head_channels,
        use_spectral_proj=sp_cfg.get("enabled", True),
        use_spectral_att=sa_cfg.get("enabled", True),
        use_spatial_att=spa_cfg.get("enabled", True)
    )
    return detector
