"""
Collate function for PyTorch DataLoader handling variable-length detection targets.
"""

from typing import List, Tuple, Dict, Any
import torch


def detection_collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Collates a list of (image_tensor, target_dict) pairs into:
    - images: Tensor of shape [B, C, H, W]
    - targets: List of target dicts of length B
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
