"""
Test-Time Augmentation (TTA) Pipeline (Section 20):
Evaluates model on augmented variations:
1. Original
2. Horizontal Flip
3. Vertical Flip
4. Scale x 0.8
5. Scale x 1.0
6. Scale x 1.2
Inverts coordinates back to original image space and merges detections via WBF.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from .wbf import weighted_box_fusion


class TestTimeAugmentation:
    __test__ = False

    def __init__(
        self,
        model: torch.nn.Module,
        scales: List[float] = [0.8, 1.0, 1.2],
        use_flips: bool = True,
        conf_thr: float = 0.05,
        iou_thr: float = 0.50
    ):
        self.model = model
        self.scales = scales
        self.use_flips = use_flips
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

    @torch.no_grad()
    def __call__(self, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        image_tensor: [1, C, H, W]
        """
        self.model.eval()
        _, c, orig_h, orig_w = image_tensor.shape
        device = image_tensor.device

        all_boxes = []
        all_scores = []
        all_labels = []

        # 1. Multi-scale variations
        for scale in self.scales:
            if scale == 1.0:
                scaled_tensor = image_tensor
            else:
                sh, sw = int(orig_h * scale), int(orig_w * scale)
                # Ensure dimensions divisible by 32
                sh, sw = (sh // 32) * 32, (sw // 32) * 32
                scaled_tensor = F.interpolate(image_tensor, size=(sh, sw), mode="bilinear", align_corners=False)

            preds = self.model.predict(scaled_tensor, conf_threshold=self.conf_thr, nms_iou_threshold=self.iou_thr)[0]
            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            if boxes.shape[0] > 0:
                # Scale boxes back to original dimensions
                scale_x = orig_w / scaled_tensor.shape[3]
                scale_y = orig_h / scaled_tensor.shape[2]
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        # 2. Horizontal Flip
        if self.use_flips:
            hflip_tensor = torch.flip(image_tensor, dims=[3])
            preds = self.model.predict(hflip_tensor, conf_threshold=self.conf_thr, nms_iou_threshold=self.iou_thr)[0]
            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            if boxes.shape[0] > 0:
                # Invert x coords: new_x1 = W - old_x2, new_x2 = W - old_x1
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = orig_w - x2
                boxes[:, 2] = orig_w - x1
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        # 3. Vertical Flip
        if self.use_flips:
            vflip_tensor = torch.flip(image_tensor, dims=[2])
            preds = self.model.predict(vflip_tensor, conf_threshold=self.conf_thr, nms_iou_threshold=self.iou_thr)[0]
            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            if boxes.shape[0] > 0:
                # Invert y coords: new_y1 = H - old_y2, new_y2 = H - old_y1
                y1 = boxes[:, 1].copy()
                y2 = boxes[:, 3].copy()
                boxes[:, 1] = orig_h - y2
                boxes[:, 3] = orig_h - y1
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        if len(all_boxes) == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64)
            }

        # Ensemble all TTA predictions with WBF
        f_boxes, f_scores, f_labels = weighted_box_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=self.iou_thr,
            skip_box_thr=self.conf_thr
        )

        return {
            "boxes": f_boxes,
            "scores": f_scores,
            "labels": f_labels
        }
