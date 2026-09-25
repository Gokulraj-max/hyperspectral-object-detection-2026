"""
Training CLI Entry Point:
python training/train.py --config configs/baseline.yaml
python training/train.py --config configs/hyperspectral.yaml
python training/train.py --config configs/spectral_attention.yaml
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import HyperspectralDataset, HyperspectralAugmentation, detection_collate_fn
from models import build_detector
from losses import HyperspectralLoss
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.checkpoint import load_checkpoint
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hyperspectral Object Detector")
    parser.add_argument("--config", type=str, default="configs/spectral_attention.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data generation for testing/dry-run")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("optimizer", {})["learning_rate"] = args.lr

    device = torch.device(args.device)
    print(f"Using compute device: {device}")

    # Build Model
    model = build_detector(cfg)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {cfg.get('model', {}).get('mode', 'detector')} with {total_params:,} trainable parameters.")

    # Build Datasets
    data_cfg = cfg.get("data", {})
    img_size = tuple(data_cfg.get("image_size", [512, 512]))
    aug_cfg = cfg.get("augmentation", {})

    train_transforms = HyperspectralAugmentation.build_train_transforms(image_size=img_size, aug_config=aug_cfg)
    val_transforms = HyperspectralAugmentation.build_val_transforms(image_size=img_size)

    # Check if raw data directory has files, else enable synthetic mode
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    has_raw_data = False
    if os.path.exists(raw_dir):
        for root, _, files in os.walk(raw_dir):
            if any(f.endswith((".npy", ".npz", ".tif", ".mat")) for f in files):
                has_raw_data = True
                break

    use_synthetic = args.synthetic or (not has_raw_data)
    if use_synthetic:
        print("[INFO] No raw dataset files detected in data/raw. Running with generated hyperspectral cubes for verification.")

    train_dataset = HyperspectralDataset(
        data_dir=raw_dir,
        split_file=data_cfg.get("train_split", "data/splits/train.txt"),
        mode=cfg.get("model", {}).get("mode", "16band"),
        selected_bands=cfg.get("model", {}).get("selected_bands", [2, 7, 15]),
        transforms=train_transforms,
        image_size=img_size,
        is_train=True,
        synthetic_count=20 if use_synthetic else 0
    )

    val_dataset = HyperspectralDataset(
        data_dir=raw_dir,
        split_file=data_cfg.get("val_split", "data/splits/val.txt"),
        mode=cfg.get("model", {}).get("mode", "16band"),
        selected_bands=cfg.get("model", {}).get("selected_bands", [2, 7, 15]),
        transforms=val_transforms,
        image_size=img_size,
        is_train=False,
        synthetic_count=8 if use_synthetic else 0
    )

    batch_size = cfg.get("training", {}).get("batch_size", 8)
    num_workers = 0 if os.name == "nt" else data_cfg.get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn
    )

    # Build Loss
    loss_cfg = cfg.get("loss", {})
    criterion = HyperspectralLoss(
        num_classes=cfg.get("model", {}).get("num_classes", 18),
        cls_weight=loss_cfg.get("cls_weight", 1.0),
        box_weight=loss_cfg.get("box_weight", 7.5),
        objectness_weight=loss_cfg.get("objectness_weight", 1.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0)
    )

    # Build Optimizer & Scheduler
    epochs = cfg.get("training", {}).get("epochs", 50)
    optimizer = build_optimizer(model, cfg.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, cfg.get("scheduler", {}), epochs=epochs)

    # Resume from checkpoint if requested
    if args.resume and os.path.isfile(args.resume):
        load_checkpoint(args.resume, model, optimizer, scheduler, device=str(device))
        print(f"Resumed training state from: {args.resume}")

    # Launch Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device
    )

    trainer.fit()


if __name__ == "__main__":
    main()
