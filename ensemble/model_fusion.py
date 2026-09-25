"""
Model Fusion abstraction for loading and ensembling multiple models.
"""

from typing import List, Dict, Any, Tuple
import os
import yaml
import torch
from models import build_detector
from datasets.transforms import BandAwareNormalize, LetterboxResize
from inference.predict import load_cube


class ModelFusionEnsemble:
    def __init__(self, model_configs: List[Dict[str, Any]], device: str = "cpu"):
        self.device = torch.device(device)
        self.models = []
        self.configs = []
        self.weights = []

        for m_item in model_configs:
            cfg_path = m_item["config"]
            ckpt_path = m_item.get("checkpoint", None)
            weight = float(m_item.get("weight", 1.0))

            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)

            model = build_detector(cfg)
            if ckpt_path and os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            model.to(self.device)
            model.eval()

            self.models.append(model)
            self.configs.append(cfg)
            self.weights.append(weight)

    @torch.no_grad()
    def predict_image(self, cube: np.ndarray, conf_thr: float = 0.05, iou_thr: float = 0.5) -> List[Dict[str, Any]]:
        orig_h, orig_w = cube.shape[0], cube.shape[1]
        model_preds = []

        for model, cfg in zip(self.models, self.configs):
            img_size = tuple(cfg.get("data", {}).get("image_size", [512, 512]))
            normalizer = BandAwareNormalize()
            resizer = LetterboxResize(target_size=img_size)

            norm_cube, _ = normalizer(cube.copy(), {})
            resized_cube, pad_meta = resizer(norm_cube, {})
            pad_left, pad_top, scale = pad_meta["pad_info"]

            in_tensor = torch.from_numpy(np.transpose(resized_cube, (2, 0, 1))).unsqueeze(0).float().to(self.device)
            res = model.predict(in_tensor, conf_threshold=conf_thr, nms_iou_threshold=iou_thr)[0]

            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()

            if len(boxes) > 0:
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

            model_preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })

        return model_preds
