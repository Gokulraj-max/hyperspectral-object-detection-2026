"""
Batch Predictor Helper:
Optimized batched inference on collections of hyperspectral scenes.
"""

from typing import List, Dict, Any
import numpy as np
import torch
from .predict import load_cube
from datasets.transforms import BandAwareNormalize, LetterboxResize


class BatchPredictor:
    def __init__(self, model: torch.nn.Module, image_size=(512, 512), device="cpu"):
        self.model = model
        self.image_size = image_size
        self.device = torch.device(device)
        self.normalizer = BandAwareNormalize()
        self.resizer = LetterboxResize(target_size=image_size)

    @torch.no_grad()
    def predict_batch(self, file_paths: List[str], conf_thr: float = 0.1, iou_thr: float = 0.5) -> List[Dict[str, Any]]:
        batch_tensors = []
        meta_list = []

        for fp in file_paths:
            cube = load_cube(fp)
            orig_h, orig_w = cube.shape[0], cube.shape[1]
            norm_cube, _ = self.normalizer(cube.copy(), {})
            resized_cube, pad_meta = self.resizer(norm_cube, {})
            pad_left, pad_top, scale = pad_meta["pad_info"]

            tensor = torch.from_numpy(np.transpose(resized_cube, (2, 0, 1))).float()
            batch_tensors.append(tensor)
            meta_list.append((orig_h, orig_w, pad_left, pad_top, scale))

        batch_x = torch.stack(batch_tensors, dim=0).to(self.device)
        raw_preds = self.model.predict(batch_x, conf_threshold=conf_thr, nms_iou_threshold=iou_thr)

        results = []
        for i, res in enumerate(raw_preds):
            orig_h, orig_w, pad_left, pad_top, scale = meta_list[i]
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()

            if len(boxes) > 0:
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

            results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })

        return results
