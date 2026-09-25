"""
Multi-Scale Decoupled Detection Head (Section 12):
Operates on P3 (stride 8: small objects), P4 (stride 16: medium objects), and P5 (stride 32: large objects).
Decoupled branches for classification (18 classes), bounding-box regression, and objectness confidence.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn


class DecoupledHead(nn.Module):
    """
    Decoupled detection head for a single scale level.
    """
    def __init__(self, in_channels: int, num_classes: int = 18, head_channels: int = 128, num_anchors: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification branch
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, num_anchors * num_classes, kernel_size=1)
        )

        # Bounding box regression branch (4 coords: dx, dy, dw, dh or l, t, r, b)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, num_anchors * 4, kernel_size=1)
        )

        # Objectness branch
        self.obj_branch = nn.Sequential(
            nn.Conv2d(in_channels, head_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels // 2, num_anchors * 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_branch(x)
        reg_preds = self.reg_branch(x)
        obj_logits = self.obj_branch(x)
        return cls_logits, reg_preds, obj_logits


class MultiScaleDetectionHead(nn.Module):
    """
    Combines heads across P3, P4, P5 strides: [8, 16, 32].
    Produces decoded bounding boxes [B, total_points, 4], class scores [B, total_points, 18],
    and objectness scores [B, total_points, 1].
    """
    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 18,
        head_channels: int = 128,
        strides: List[int] = [8, 16, 32]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.heads = nn.ModuleList([
            DecoupledHead(in_channels=in_channels, num_classes=num_classes, head_channels=head_channels)
            for _ in strides
        ])

    def forward(
        self,
        pyramid_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, Any]:
        cls_outputs = []
        reg_outputs = []
        obj_outputs = []
        grid_points = []
        stride_tensors = []

        batch_size = pyramid_features[0].shape[0]
        device = pyramid_features[0].device

        for i, (feat, stride) in enumerate(zip(pyramid_features, self.strides)):
            cls_out, reg_out, obj_out = self.heads[i](feat)
            b, _, h, w = feat.shape

            # Reshape from [B, C, H, W] to [B, H*W, C]
            cls_out = cls_out.permute(0, 2, 3, 1).contiguous().view(b, h * w, self.num_classes)
            reg_out = reg_out.permute(0, 2, 3, 1).contiguous().view(b, h * w, 4)
            obj_out = obj_out.permute(0, 2, 3, 1).contiguous().view(b, h * w, 1)

            # Generate grid centers (x, y)
            shift_y, shift_x = torch.meshgrid(
                torch.arange(0, h, device=device, dtype=torch.float32),
                torch.arange(0, w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            shift_x = (shift_x.reshape(-1) + 0.5) * stride
            shift_y = (shift_y.reshape(-1) + 0.5) * stride
            points = torch.stack([shift_x, shift_y], dim=-1) # [H*W, 2]

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            obj_outputs.append(obj_out)
            grid_points.append(points)
            stride_tensors.append(torch.full((h * w, 1), stride, device=device, dtype=torch.float32))

        all_cls = torch.cat(cls_outputs, dim=1)        # [B, N, num_classes]
        all_reg = torch.cat(reg_outputs, dim=1)        # [B, N, 4]
        all_obj = torch.cat(obj_outputs, dim=1)        # [B, N, 1]
        all_points = torch.cat(grid_points, dim=0)     # [N, 2]
        all_strides = torch.cat(stride_tensors, dim=0) # [N, 1]

        # Decode bounding boxes (dx, dy, dw, dh relative to grid center and stride)
        # decoded x1, y1, x2, y2
        cx = all_points[:, 0:1] + all_reg[:, :, 0:1] * all_strides
        cy = all_points[:, 1:2] + all_reg[:, :, 1:2] * all_strides
        w = torch.exp(torch.clamp(all_reg[:, :, 2:3], min=-4.0, max=4.0)) * all_strides * 2.0
        h = torch.exp(torch.clamp(all_reg[:, :, 3:4], min=-4.0, max=4.0)) * all_strides * 2.0

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        decoded_boxes = torch.cat([x1, y1, x2, y2], dim=-1)

        return {
            "cls_logits": all_cls,
            "reg_preds": all_reg,
            "obj_logits": all_obj,
            "decoded_boxes": decoded_boxes,
            "grid_points": all_points,
            "strides": all_strides
        }
