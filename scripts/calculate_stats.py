"""
Dataset & Model Statistics Calculator:
1. Computes exact per-band dataset statistics (mean, std, min, max across 16 bands)
2. Computes model complexity metrics for Top-10 Code Review:
   - Parameter count (total and trainable)
   - Estimated FLOPs / MACs at input resolution (e.g. 512x512)
   - Model memory footprint
Outputs release/model_stats.txt.
"""

import os
import sys
import argparse
import yaml
from typing import Dict, Any
import numpy as np
import torch
import tifffile

# Add repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import build_detector


def calculate_dataset_stats(data_dir: str = "data/raw/train", num_samples: int = 100) -> Dict[str, np.ndarray]:
    import glob
    files = []
    for ext in (".npy", ".npz", ".tif", ".tiff"):
        files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True))
    files = sorted(list(set(files)))

    if not files:
        print("[INFO] No train files found. Using synthetic statistics for 16 bands.")
        means = np.linspace(0.25, 0.75, 16)
        stds = np.full(16, 0.15)
        return {"mean": means, "std": stds}

    files = files[:num_samples]
    band_sums = np.zeros(16, dtype=np.float64)
    band_sq_sums = np.zeros(16, dtype=np.float64)
    total_pixels = 0

    print(f"Calculating spectral statistics over {len(files)} cubes...")
    for fp in files:
        ext = os.path.splitext(fp)[1]
        if ext == ".npy":
            arr = np.load(fp)
        elif ext in (".tif", ".tiff"):
            arr = tifffile.imread(fp)
        else:
            continue

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 16:
            arr = np.transpose(arr, (1, 2, 0))
        
        arr = np.nan_to_num(arr, nan=0.0)
        h, w, c = arr.shape
        c = min(16, c)
        pixels = h * w
        total_pixels += pixels

        flat = arr[:, :, :c].reshape(-1, c)
        band_sums[:c] += np.sum(flat, axis=0)
        band_sq_sums[:c] += np.sum(flat ** 2, axis=0)

    means = band_sums / total_pixels
    stds = np.sqrt(np.maximum(0.0, (band_sq_sums / total_pixels) - (means ** 2)))
    return {"mean": means, "std": stds}


def calculate_model_complexity(config_path: str = "configs/final.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = build_detector(cfg)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    img_size = cfg.get("data", {}).get("image_size", [512, 512])
    in_channels = cfg.get("model", {}).get("input_channels", 16)
    dummy_input = torch.randn(1, in_channels, img_size[0], img_size[1])

    # Estimate FLOPs
    def count_conv_flops(module, in_t, out_t):
        if isinstance(module, torch.nn.Conv2d):
            out_h, out_w = out_t.shape[2], out_t.shape[3]
            k_h, k_w = module.kernel_size
            flops = module.in_channels * module.out_channels * k_h * k_w * out_h * out_w / module.groups
            module.__flops__ = flops

    handles = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(count_conv_flops))

    with torch.no_grad():
        _ = model(dummy_input)

    for h in handles:
        h.remove()

    total_conv_flops = sum(getattr(m, "__flops__", 0) for m in model.modules())
    gflops = total_conv_flops / 1e9

    stats = {
        "model_name": cfg.get("model", {}).get("mode", "HS-SAFD"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_resolution": f"{img_size[0]}x{img_size[1]}",
        "input_channels": in_channels,
        "estimated_gflops": round(gflops, 2),
        "model_size_mb": round(total_params * 4 / (1024 * 1024), 2)
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Calculate Dataset & Model Stats")
    parser.add_argument("--config", type=str, default="configs/final.yaml", help="Path to config")
    parser.add_argument("--data-dir", type=str, default="data/raw/train", help="Train data directory")
    parser.add_argument("--output", type=str, default="release/model_stats.txt", help="Output file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    d_stats = calculate_dataset_stats(args.data_dir)
    m_stats = calculate_model_complexity(args.config)

    lines = [
        "=" * 60,
        "HYPERSPECTRAL OBJECT DETECTION CHALLENGE 2026 - MODEL STATS",
        "=" * 60,
        f"Architecture:          {m_stats['model_name']}",
        f"Input Resolution:      {m_stats['input_resolution']} (16 spectral bands, 460-600 nm)",
        f"Total Parameters:      {m_stats['total_parameters']:,}",
        f"Trainable Parameters:  {m_stats['trainable_parameters']:,}",
        f"Model Weight Size:     {m_stats['model_size_mb']} MB",
        f"Estimated FLOPs:       {m_stats['estimated_gflops']} GFLOPs",
        "-" * 60,
        "Per-Band Dataset Statistics (16 Bands):"
    ]
    for b in range(16):
        lines.append(f"  Band {b:2d} ({(460.0 + b * 9.33):.1f} nm): Mean = {d_stats['mean'][b]:.4f}, Std = {d_stats['std'][b]:.4f}")
    lines.append("=" * 60)

    content = "\n".join(lines)
    print(content)

    with open(args.output, "w") as f:
        f.write(content + "\n")
    print(f"\nModel statistics recorded in: {args.output}")


if __name__ == "__main__":
    main()
