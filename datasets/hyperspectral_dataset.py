"""
Hyperspectral Object Detection PyTorch Dataset.
Supports 16-band HSI cubes (.npy, .npz, .tif, .mat), on-the-fly or cached PCA / RGB extraction,
and multi-format bounding box annotations.
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import tifffile

from .annotation_parser import parse_annotations, convert_bbox
from .transforms import Compose


class HyperspectralDataset(Dataset):
    """
    Hyperspectral Object Detection Dataset.
    Modes:
    - '16band': returns full 16-channel hyperspectral cube [16, H, W]
    - 'rgb_baseline': extracts 3 visible bands [3, H, W]
    - 'pca': fits/projects 16 bands to 3 principal component channels [3, H, W]
    """
    def __init__(
        self,
        data_dir: str,
        split_file: Optional[str] = None,
        annotations_file: Optional[str] = None,
        mode: str = "16band",
        selected_bands: Optional[List[int]] = None,
        transforms: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512),
        is_train: bool = True,
        synthetic_count: int = 0  # If >0, generates synthetic HSI cubes for testing
    ):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.selected_bands = selected_bands or [2, 7, 15]
        self.transforms = transforms
        self.image_size = image_size
        self.is_train = is_train
        self.synthetic_count = synthetic_count

        self.samples: List[Dict[str, Any]] = []
        self._load_dataset(data_dir, split_file, annotations_file)

    def _load_dataset(self, data_dir: str, split_file: Optional[str], annotations_file: Optional[str]):
        """Discovers images and loads annotations."""
        if self.synthetic_count > 0:
            for i in range(self.synthetic_count):
                self.samples.append({
                    "image_id": f"synthetic_{i:04d}",
                    "file_path": None,
                    "is_synthetic": True,
                    "height": self.image_size[0],
                    "width": self.image_size[1]
                })
            return

        # Find all valid image files
        allowed_extensions = (".npy", ".npz", ".tif", ".tiff", ".mat")
        found_files = []
        if os.path.exists(data_dir):
            for ext in allowed_extensions:
                found_files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
                found_files.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True))

        # Filter by split if provided
        valid_ids = None
        if split_file and os.path.isfile(split_file):
            with open(split_file, 'r', encoding='utf-8') as f:
                valid_ids = set(line.strip() for line in f if line.strip())

        # Load annotations if available
        annotations_map = {}
        if annotations_file and (os.path.isfile(annotations_file) or os.path.isdir(annotations_file)):
            if os.path.isdir(annotations_file) and glob.glob(os.path.join(annotations_file, "*.xml")):
                from .voc_parser import parse_voc_directory
                annotations_map = parse_voc_directory(annotations_file)
            else:
                annotations_map = parse_annotations(annotations_file)
        else:
            # Check default candidate paths inside data_dir or adjacent annotations folder
            candidates = [
                os.path.join(data_dir, "annotations"),
                os.path.join(data_dir, "..", "annotations"),
                os.path.join(data_dir, "annotations.json"),
                os.path.join(data_dir, "annotations.csv"),
                os.path.join(data_dir, "labels"),
                "data/raw/annotations"
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*.xml")):
                        from .voc_parser import parse_voc_directory
                        annotations_map = parse_voc_directory(cand)
                        break
                    elif os.path.isfile(cand):
                        annotations_map = parse_annotations(cand)
                        break

        for fp in sorted(found_files):
            stem = os.path.splitext(os.path.basename(fp))[0]
            if valid_ids is not None and stem not in valid_ids and os.path.basename(fp) not in valid_ids:
                continue

            ann = annotations_map.get(stem, {"boxes": [], "labels": []})
            self.samples.append({
                "image_id": stem,
                "file_path": fp,
                "is_synthetic": False,
                "boxes": ann.get("boxes", []),
                "labels": ann.get("labels", [])
            })

    def __len__(self) -> int:
        return len(self.samples)

    def _read_cube(self, file_path: str) -> np.ndarray:
        """Reads 16-band HSI cube into numpy array of shape (H, W, 16)."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            data = np.load(file_path)
        elif ext == ".npz":
            archive = np.load(file_path)
            key = list(archive.keys())[0]
            data = archive[key]
        elif ext in (".tif", ".tiff"):
            data = tifffile.imread(file_path)
        elif ext == ".mat":
            from scipy.io import loadmat
            mat = loadmat(file_path)
            # Find the largest 3D array in the mat dict
            keys = [k for k in mat.keys() if not k.startswith("__")]
            key = max(keys, key=lambda k: mat[k].size)
            data = mat[key]
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Ensure shape (H, W, 16)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 3:
            if data.shape[0] == 16 and data.shape[2] != 16: # (16, H, W)
                data = np.transpose(data, (1, 2, 0))
        elif data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        
        # Handle invalid values
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        return data

    def _generate_synthetic_sample(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generates realistic synthetic 16-band HSI cube and bounding boxes for testing."""
        np.random.seed(idx)
        h, w = self.image_size
        num_bands = 16

        # Base background with continuous spectral gradient
        base_spectrum = np.linspace(0.2, 0.6, num_bands)
        background = np.ones((h, w, num_bands), dtype=np.float32) * base_spectrum.reshape(1, 1, num_bands)
        background += np.random.normal(0, 0.02, size=(h, w, num_bands)).astype(np.float32)

        # Place 1 to 4 objects with distinct material signatures
        num_objects = np.random.randint(1, 4)
        boxes, labels = [], []

        for _ in range(num_objects):
            bw = np.random.randint(40, min(140, w // 2))
            bh = np.random.randint(40, min(140, h // 2))
            x1 = np.random.randint(10, w - bw - 10)
            y1 = np.random.randint(10, h - bh - 10)
            x2 = x1 + bw
            y2 = y1 + bh

            cls_id = int(np.random.randint(0, 18))
            # Material reflectance signature (e.g. real vs counterfeit pairs have shifted absorption peaks)
            is_counterfeit = (cls_id % 2 == 1)
            peak_idx = 10 if is_counterfeit else 6
            obj_spectrum = 0.5 + 0.3 * np.exp(-0.5 * ((np.arange(num_bands) - peak_idx) / 2.5) ** 2)
            
            background[y1:y2, x1:x2, :] = obj_spectrum.reshape(1, 1, num_bands) + np.random.normal(0, 0.01, size=(bh, bw, num_bands))
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(cls_id)

        target = {
            "image_id": f"synthetic_{idx:04d}",
            "boxes": boxes,
            "labels": labels,
            "orig_size": (h, w)
        }
        return background, target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        sample_info = self.samples[idx]

        if sample_info.get("is_synthetic", False):
            cube, target = self._generate_synthetic_sample(idx)
        else:
            cube = self._read_cube(sample_info["file_path"])
            target = {
                "image_id": sample_info["image_id"],
                "boxes": [list(b) for b in sample_info.get("boxes", [])],
                "labels": [int(l) for l in sample_info.get("labels", [])],
                "orig_size": (cube.shape[0], cube.shape[1])
            }

        # Handle projection mode
        if self.mode == "rgb_baseline":
            # Extract 3 selected bands
            num_bands = cube.shape[2]
            bands = [min(b, num_bands - 1) for b in self.selected_bands]
            cube = cube[:, :, bands]
        elif self.mode == "pca":
            # PCA 16 -> 3 channels
            h, w, c = cube.shape
            flat = cube.reshape(-1, c)
            pca = PCA(n_components=3)
            projected = pca.fit_transform(flat)
            cube = projected.reshape(h, w, 3).astype(np.float32)

        # Apply augmentations / transforms
        if self.transforms is not None:
            image_tensor, target = self.transforms(cube, target)
        else:
            # Default to tensor [C, H, W]
            image_tensor = torch.from_numpy(np.transpose(cube, (2, 0, 1))).float()
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32) if target["boxes"] else torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64) if target["labels"] else torch.zeros((0,), dtype=torch.int64)

        return image_tensor, target
