"""
Hyperspectral visualization tools:
1. Pseudo-RGB visualization (natural visible bands)
2. False-color composite visualization
3. Spectral signature plotting (real vs counterfeit material comparison)
4. Bounding box overlay visualization
"""

from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


# 16 spectral bands between 460 nm and 600 nm
DEFAULT_WAVELENGTHS = [
    460.0, 469.3, 478.7, 488.0, 497.3, 506.7, 516.0, 525.3,
    534.7, 544.0, 553.3, 562.7, 572.0, 581.3, 590.7, 600.0
]


def _ensure_hwc(cube: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Ensures input cube is a numpy array of shape (H, W, C)."""
    if isinstance(cube, torch.Tensor):
        cube = cube.detach().cpu().numpy()
    if cube.ndim == 3:
        if cube.shape[0] <= 32 and cube.shape[2] > 32: # (C, H, W)
            cube = np.transpose(cube, (1, 2, 0))
    elif cube.ndim == 2:
        cube = np.expand_dims(cube, axis=-1)
    return cube.astype(np.float32)


def render_pseudo_rgb(
    cube: Union[np.ndarray, torch.Tensor],
    red_band: int = 15,    # ~600 nm
    green_band: int = 7,   # ~525 nm
    blue_band: int = 1     # ~469 nm
) -> np.ndarray:
    """
    Renders a natural-looking pseudo-RGB image from 16-band HSI cube.
    Returns uint8 RGB array (H, W, 3) in [0, 255].
    """
    cube_hwc = _ensure_hwc(cube)
    num_bands = cube_hwc.shape[2]
    
    r_idx = min(red_band, num_bands - 1)
    g_idx = min(green_band, num_bands - 1)
    b_idx = min(blue_band, num_bands - 1)

    r = cube_hwc[:, :, r_idx]
    g = cube_hwc[:, :, g_idx]
    b = cube_hwc[:, :, b_idx]

    rgb = np.stack([r, g, b], axis=-1)

    # Robust contrast normalization per channel
    rgb_norm = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        ch = rgb[:, :, c]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            scaled = (ch - p2) / (p98 - p2)
            rgb_norm[:, :, c] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_norm[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    return rgb_norm


def render_false_color(
    cube: Union[np.ndarray, torch.Tensor],
    r_band: int = 15,
    g_band: int = 8,
    b_band: int = 2
) -> np.ndarray:
    """
    Renders false-color composite to emphasize subtle material and chemical differences.
    """
    return render_pseudo_rgb(cube, red_band=r_band, green_band=g_band, blue_band=b_band)


def plot_spectral_signature(
    cube: Union[np.ndarray, torch.Tensor],
    boxes: List[List[float]],
    labels: List[int],
    class_names: List[str],
    wavelengths: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Spectral Signatures of Detected Objects"
) -> plt.Figure:
    """
    Extracts and plots mean spectral reflectance curves (+- std deviation)
    over each object's bounding box region across all 16 wavelengths (460 - 600 nm).
    Particularly useful for analyzing Real vs Counterfeit pairs.
    """
    cube_hwc = _ensure_hwc(cube)
    h, w, num_bands = cube_hwc.shape
    wl = wavelengths or DEFAULT_WAVELENGTHS[:num_bands]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        region = cube_hwc[y1:y2, x1:x2, :] # (H_crop, W_crop, C)
        mean_curve = np.mean(region, axis=(0, 1))
        std_curve = np.std(region, axis=(0, 1))

        label_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        color = colors[idx % len(colors)]

        # Highlight real vs fake naming
        is_fake = "fake" in label_name.lower() or "counterfeit" in label_name.lower() or "adulterated" in label_name.lower()
        linestyle = "--" if is_fake else "-"

        ax.plot(wl, mean_curve, label=f"Obj {idx}: {label_name}", color=color, linestyle=linestyle, linewidth=2)
        ax.fill_between(wl, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Spectral Intensity / Reflectance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Union[List[List[float]], torch.Tensor, np.ndarray],
    labels: Union[List[int], torch.Tensor, np.ndarray],
    scores: Optional[Union[List[float], torch.Tensor, np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
    color_by_class: bool = True
) -> np.ndarray:
    """
    Overlays high-precision bounding boxes, category labels, and confidence scores onto an RGB image.
    Uses distinctive colors for Real vs Counterfeit pairs.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    elif canvas.shape[2] > 3: # HSI cube
        canvas = render_pseudo_rgb(canvas)

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    for i in range(len(boxes)):
        box = boxes[i]
        label = int(labels[i])
        score = float(scores[i]) if scores is not None else None

        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        
        name = class_names[label] if (class_names and label < len(class_names)) else f"cls_{label}"
        is_counterfeit = any(kw in name.lower() for kw in ["fake", "counterfeit", "synthetic", "adulterated", "laminated", "glass"])
        
        # Color coding: Greenish for Authentic / Real, Crimson/Red for Counterfeit
        if is_counterfeit:
            box_color = (0, 0, 220) # Red in BGR / RGB
        else:
            box_color = (0, 180, 0) # Green

        # Draw bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        # Label tag
        tag = f"{name}"
        if score is not None:
            tag += f" {score:.2f}"

        # Text banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, thickness)
        
        tag_y1 = max(0, y1 - th - baseline - 4)
        tag_y2 = y1
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 6, tag_y2), box_color, -1)
        cv2.putText(canvas, tag, (x1 + 3, y1 - baseline - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas
