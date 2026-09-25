"""
Learning Rate Scheduler Builder:
Cosine Annealing with Linear Warmup.
"""

import math
from typing import Dict, Any
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: Dict[str, Any],
    epochs: int
) -> torch.optim.lr_scheduler._LRScheduler:
    sched_type = sched_cfg.get("type", "cosine").lower()
    warmup_epochs = sched_cfg.get("warmup_epochs", 3)
    min_lr = float(sched_cfg.get("min_lr", 1e-6))

    if sched_type == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif sched_type == "step":
        step_size = sched_cfg.get("step_size", 20)
        gamma = sched_cfg.get("gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
