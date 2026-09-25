"""
Hyperspectral and Spatial Augmentation Pipeline.
Preserves continuous spectral correlations while applying spatial and spectral transforms.
"""

import math
import random
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import cv2


class Compose:
    """Sequential composition of transforms."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class BandAwareNormalize:
    """
    Per-band normalization:
    X[b] = (X[b] - mean[b]) / (std[b] + eps)
    or per-image per-band min-max normalization.
    """
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # image shape: (H, W, C) or (C, H, W)
        img = image.astype(np.float32)
        
        # Replace NaNs or Infs if any
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

        if self.mean is not None and self.std is not None:
            # Broadcast over spatial dimensions
            if img.ndim == 3 and img.shape[2] == len(self.mean): # (H, W, C)
                img = (img - self.mean.reshape(1, 1, -1)) / (self.std.reshape(1, 1, -1) + self.eps)
            elif img.ndim == 3 and img.shape[0] == len(self.mean): # (C, H, W)
                img = (img - self.mean.reshape(-1, 1, 1)) / (self.std.reshape(-1, 1, 1) + self.eps)
        else:
            # Per-image per-band robust standardization (clip 1st to 99th percentile)
            if img.ndim == 3:
                channels = img.shape[2] if img.shape[2] <= 32 else img.shape[0]
                is_hwc = (img.shape[2] == channels)
                for c in range(channels):
                    band = img[:, :, c] if is_hwc else img[c, :, :]
                    p1, p99 = np.percentile(band, 1), np.percentile(band, 99)
                    if p99 > p1:
                        band = np.clip(band, p1, p99)
                        b_mean = np.mean(band)
                        b_std = np.std(band) + self.eps
                        band = (band - b_mean) / b_std
                    if is_hwc:
                        img[:, :, c] = band
                    else:
                        img[c, :, :] = band
        return img, target


class SpectralNoise:
    """
    Adds Gaussian noise across bands preserving wavelength smoothness.
    Draws a smooth noise curve across the 16 bands instead of uncorrelated white noise.
    """
    def __init__(self, p: float = 0.5, std: float = 0.02):
        self.p = p
        self.std = std

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if random.random() > self.p:
            return image, target
        
        # image: (H, W, C)
        is_hwc = (image.ndim == 3 and image.shape[2] <= 64)
        c = image.shape[2] if is_hwc else image.shape[0]

        # Generate smooth spectral noise vector
        t = np.linspace(0, 2 * np.pi, c)
        smooth_noise = np.random.normal(0, self.std) * np.sin(t + np.random.uniform(0, 2 * np.pi))
        smooth_noise += np.random.normal(0, self.std * 0.3, size=c) # small high-frequency component

        if is_hwc:
            image = image + smooth_noise.reshape(1, 1, c)
        else:
            image = image + smooth_noise.reshape(c, 1, 1)
        return image, target


class SpectralBandDropout:
    """
    Simulates detector sensor failure or band occlusion by zeroing out 1 or 2 bands with probability p.
    """
    def __init__(self, p: float = 0.3, max_drop: int = 2):
        self.p = p
        self.max_drop = max_drop

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if random.random() > self.p:
            return image, target

        is_hwc = (image.ndim == 3 and image.shape[2] <= 64)
        num_bands = image.shape[2] if is_hwc else image.shape[0]
        
        drop_count = random.randint(1, self.max_drop)
        dropped_indices = random.sample(range(num_bands), drop_count)

        image = image.copy()
        for idx in dropped_indices:
            if is_hwc:
                image[:, :, idx] = 0.0
            else:
                image[idx, :, :] = 0.0
        return image, target


class SpectralIntensityScale:
    """
    Simulates illumination variation by applying a smooth multiplicative spectral curve.
    """
    def __init__(self, p: float = 0.5, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if random.random() > self.p:
            return image, target

        is_hwc = (image.ndim == 3 and image.shape[2] <= 64)
        num_bands = image.shape[2] if is_hwc else image.shape[0]
        
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        # Smooth tilt across spectrum (e.g. warmer or cooler illumination)
        tilt = np.linspace(-0.05, 0.05, num_bands) * np.random.uniform(-1, 1)
        factors = (scale_factor + tilt).astype(np.float32)

        if is_hwc:
            image = image * factors.reshape(1, 1, num_bands)
        else:
            image = image * factors.reshape(num_bands, 1, 1)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if random.random() > self.p:
            return image, target

        is_hwc = (image.ndim == 3 and image.shape[2] <= 64)
        if is_hwc:
            w = image.shape[1]
            image = np.ascontiguousarray(np.fliplr(image))
        else:
            w = image.shape[2]
            image = np.ascontiguousarray(image[:, :, ::-1])

        boxes = target.get("boxes", [])
        if len(boxes) > 0:
            new_boxes = []
            for b in boxes:
                x1, y1, x2, y2 = b
                new_x1 = w - x2
                new_x2 = w - x1
                new_boxes.append([new_x1, y1, new_x2, y2])
            target["boxes"] = new_boxes
        return image, target


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if random.random() > self.p:
            return image, target

        is_hwc = (image.ndim == 3 and image.shape[2] <= 64)
        if is_hwc:
            h = image.shape[0]
            image = np.ascontiguousarray(np.flipud(image))
        else:
            h = image.shape[1]
            image = np.ascontiguousarray(image[:, ::-1, :])

        boxes = target.get("boxes", [])
        if len(boxes) > 0:
            new_boxes = []
            for b in boxes:
                x1, y1, x2, y2 = b
                new_y1 = h - y2
                new_y2 = h - y1
                new_boxes.append([x1, new_y1, x2, new_y2])
            target["boxes"] = new_boxes
        return image, target


class LetterboxResize:
    """
    Resizes image maintaining aspect ratio and pads to target size (height, width).
    """
    def __init__(self, target_size: Tuple[int, int] = (512, 512), pad_value: float = 0.0):
        self.target_h, self.target_w = target_size
        self.pad_value = pad_value

    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        is_hwc = (image.ndim == 3 and image.shape[2] <= 64)
        if not is_hwc:
            # (C, H, W) -> (H, W, C) for cv2 resize
            image = np.transpose(image, (1, 2, 0))

        h, w, c = image.shape
        target_h, target_w = self.target_h, self.target_w

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # Resize each band or vectorized
        resized_channels = []
        for i in range(c):
            band_res = cv2.resize(image[:, :, i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_channels.append(band_res)
        resized_img = np.stack(resized_channels, axis=-1)

        # Pad canvas
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        padded_img = np.pad(
            resized_img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=self.pad_value
        )

        # Update boxes
        boxes = target.get("boxes", [])
        if len(boxes) > 0:
            new_boxes = []
            for b in boxes:
                x1, y1, x2, y2 = b
                x1 = x1 * scale + pad_left
                x2 = x2 * scale + pad_left
                y1 = y1 * scale + pad_top
                y2 = y2 * scale + pad_top
                # Clamp within target dimensions
                x1 = max(0.0, min(float(target_w), x1))
                y1 = max(0.0, min(float(target_h), y1))
                x2 = max(0.0, min(float(target_w), x2))
                y2 = max(0.0, min(float(target_h), y2))
                if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                    new_boxes.append([x1, y1, x2, y2])
            target["boxes"] = new_boxes

        target["pad_info"] = (pad_left, pad_top, scale)
        return padded_img, target


class ToTensor:
    """Converts numpy array (H, W, C) to torch FloatTensor (C, H, W)."""
    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        is_hwc = (image.ndim == 3 and image.shape[2] <= 64)
        if is_hwc:
            img_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        else:
            img_tensor = torch.from_numpy(image).float()

        boxes = target.get("boxes", [])
        labels = target.get("labels", [])

        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target


class HyperspectralAugmentation:
    """Factory helper to build train and val transform pipelines from config."""
    @staticmethod
    def build_train_transforms(image_size: Tuple[int, int] = (512, 512), aug_config: Optional[Dict[str, Any]] = None) -> Compose:
        cfg = aug_config or {}
        transforms = []
        transforms.append(BandAwareNormalize())
        
        if cfg.get("spectral_noise", True):
            transforms.append(SpectralNoise(p=0.5, std=cfg.get("spectral_noise_std", 0.02)))
        if cfg.get("band_dropout", True):
            transforms.append(SpectralBandDropout(p=cfg.get("band_dropout_prob", 0.2)))
        if cfg.get("spectral_intensity_scale", True):
            transforms.append(SpectralIntensityScale(p=0.5))
        if cfg.get("horizontal_flip", 0.5) > 0:
            transforms.append(RandomHorizontalFlip(p=cfg.get("horizontal_flip", 0.5)))
        if cfg.get("vertical_flip", 0.5) > 0:
            transforms.append(RandomVerticalFlip(p=cfg.get("vertical_flip", 0.5)))

        transforms.append(LetterboxResize(target_size=image_size))
        transforms.append(ToTensor())
        return Compose(transforms)

    @staticmethod
    def build_val_transforms(image_size: Tuple[int, int] = (512, 512)) -> Compose:
        return Compose([
            BandAwareNormalize(),
            LetterboxResize(target_size=image_size),
            ToTensor()
        ])
