"""
Inference CLI Entry Point:
python inference/predict.py --checkpoint checkpoints/best.pt --input data/raw/test --output outputs/predictions
"""

import os
import sys
import glob
import argparse
import yaml
import json
import cv2
import numpy as np
import torch
import tifffile

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import build_detector
from datasets import CLASS_NAMES, render_pseudo_rgb, draw_bounding_boxes
from datasets.transforms import BandAwareNormalize, LetterboxResize
from inference.tta import TestTimeAugmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with Hyperspectral Object Detector")
    parser.add_argument("--config", type=str, default="configs/spectral_attention.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--input", type=str, default="data/raw/test", help="Path to input HSI cube or directory")
    parser.add_argument("--output", type=str, default="outputs/predictions", help="Directory to save predictions")
    parser.add_argument("--conf-threshold", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="NMS/WBF IoU threshold")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def load_cube(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".npy":
        cube = np.load(file_path)
    elif ext == ".npz":
        archive = np.load(file_path)
        cube = archive[list(archive.keys())[0]]
    elif ext in (".tif", ".tiff"):
        cube = tifffile.imread(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim == 3 and cube.shape[0] == 16 and cube.shape[2] != 16:
        cube = np.transpose(cube, (1, 2, 0))
    elif cube.ndim == 2:
        cube = np.expand_dims(cube, -1)
    return np.nan_to_num(cube, nan=0.0, posinf=1.0, neginf=0.0)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Build model
    model = build_detector(cfg)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    else:
        print("[WARNING] No checkpoint provided or file missing. Using initialized model.")
    model.to(device)
    model.eval()

    tta_pipeline = TestTimeAugmentation(model, conf_thr=args.conf_threshold, iou_thr=args.iou_threshold) if args.tta else None

    # Discover test files
    input_files = []
    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        for ext in (".npy", ".npz", ".tif", ".tiff"):
            input_files.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
    
    if len(input_files) == 0:
        print(f"No test files found in {args.input}. Generating 1 synthetic test sample for demonstration...")
        syn_cube = np.random.uniform(0.1, 0.9, size=(512, 512, 16)).astype(np.float32)
        syn_path = os.path.join(args.output, "sample_test_cube.npy")
        np.save(syn_path, syn_cube)
        input_files = [syn_path]

    print(f"Running inference on {len(input_files)} images...")
    normalizer = BandAwareNormalize()
    img_size = tuple(cfg.get("data", {}).get("image_size", [512, 512]))
    resizer = LetterboxResize(target_size=img_size)

    all_results = {}

    for fp in input_files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        cube = load_cube(fp)
        orig_h, orig_w = cube.shape[0], cube.shape[1]

        # Preprocess
        norm_cube, _ = normalizer(cube.copy(), {})
        resized_cube, pad_meta = resizer(norm_cube, {})
        pad_left, pad_top, scale = pad_meta["pad_info"]

        # To tensor
        in_tensor = torch.from_numpy(np.transpose(resized_cube, (2, 0, 1))).unsqueeze(0).float().to(device)

        # Predict
        if args.tta:
            preds = tta_pipeline(in_tensor)
            boxes = preds["boxes"]
            scores = preds["scores"]
            labels = preds["labels"]
        else:
            with torch.no_grad():
                res = model.predict(in_tensor, conf_threshold=args.conf_threshold, nms_iou_threshold=args.iou_threshold)[0]
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                labels = res["labels"].cpu().numpy()

        # Invert letterbox padding
        if len(boxes) > 0:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
            # Clip
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        all_results[stem] = {
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist()
        }

        # Visualize overlay
        rgb_img = render_pseudo_rgb(cube)
        vis_img = draw_bounding_boxes(
            rgb_img,
            boxes,
            labels,
            scores=scores,
            class_names=CLASS_NAMES
        )
        save_vis_path = os.path.join(args.output, f"{stem}_pred.jpg")
        cv2.imwrite(save_vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    # Save summary JSON
    json_path = os.path.join(args.output, "predictions.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Inference completed. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
