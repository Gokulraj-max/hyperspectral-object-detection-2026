"""
Optimizer Builder:
Creates AdamW or SGD optimizer with decoupled weight decay for normalization and bias layers.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, opt_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_type = opt_cfg.get("type", "AdamW").lower()
    lr = float(opt_cfg.get("learning_rate", 0.0005))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0001))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D normalizations shouldn't have weight decay
        if param.ndim <= 1 or "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    if opt_type == "adamw":
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
    elif opt_type == "adam":
        return torch.optim.Adam(optim_groups, lr=lr, betas=betas)
    elif opt_type == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(optim_groups, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
