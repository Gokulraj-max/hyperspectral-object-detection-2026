"""
Trainer Engine for Hyperspectral Object Detection.
Features:
- Mixed precision (AMP)
- Gradient accumulation
- Dynamic learning rate warmup and decay
- Per-epoch validation and COCO-style mAP calculation
- Checkpoint management (best.pt and last.pt)
- CSV metrics history tracking
"""

import os
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import evaluate_detections
from evaluation.per_class import format_per_class_table
from .callbacks import CSVLogger, EarlyStopping
from .checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
        self.use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.save_dir = train_cfg.get("save_dir", "checkpoints/default")
        self.log_interval = train_cfg.get("log_interval", 10)
        self.eval_interval = train_cfg.get("eval_interval", 1)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.logger = CSVLogger(self.save_dir)
        self.early_stopping = EarlyStopping(patience=train_cfg.get("patience", 15), mode="max")

        self.best_map = 0.0
        self.start_epoch = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_obj_loss = 0.0
        steps = len(self.train_loader)

        self.optimizer.zero_grad()

        pbar = tqdm(enumerate(self.train_loader), total=steps, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]")
        for step, (images, targets) in pbar:
            images = images.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, targets)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0 or (step + 1) == steps:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss_dict["loss_total"]
            total_cls_loss += loss_dict["loss_cls"]
            total_box_loss += loss_dict["loss_box"]
            total_obj_loss += loss_dict["loss_obj"]

            if (step + 1) % self.log_interval == 0 or (step + 1) == steps:
                pbar.set_postfix({
                    "loss": f"{loss_dict['loss_total']:.3f}",
                    "cls": f"{loss_dict['loss_cls']:.3f}",
                    "box": f"{loss_dict['loss_box']:.3f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

        return {
            "train_loss": total_loss / max(1, steps),
            "train_loss_cls": total_cls_loss / max(1, steps),
            "train_loss_box": total_box_loss / max(1, steps),
            "train_loss_obj": total_obj_loss / max(1, steps)
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        self.model.eval()
        eval_cfg = self.config.get("evaluation", {})
        conf_thr = eval_cfg.get("conf_threshold", 0.05)
        iou_thr = eval_cfg.get("nms_iou_threshold", 0.50)

        all_predictions = []
        all_targets = []

        for images, targets in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            preds = self.model.predict(images, conf_threshold=conf_thr, nms_iou_threshold=iou_thr)
            all_predictions.extend(preds)
            all_targets.extend(targets)

        metrics = evaluate_detections(
            all_predictions,
            all_targets,
            num_classes=self.config.get("model", {}).get("num_classes", 18)
        )
        return metrics

    def fit(self):
        print(f"Starting training for {self.epochs} epochs on device '{self.device}'...")
        print(f"Checkpoints and logs will be saved to: {self.save_dir}")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            
            # Step learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            val_metrics = {}
            if (epoch + 1) % self.eval_interval == 0:
                val_metrics = self.validate()
                current_map = val_metrics["mAP_50_95"]
                is_best = current_map > self.best_map
                if is_best:
                    self.best_map = current_map

                epoch_time = time.time() - t0
                print(f"\n[Epoch {epoch + 1}/{self.epochs}] ({epoch_time:.1f}s) "
                      f"Train Loss: {train_metrics['train_loss']:.4f} | "
                      f"mAP@50: {val_metrics['mAP_50']:.4f} | "
                      f"mAP@75: {val_metrics['mAP_75']:.4f} | "
                      f"mAP@[50:95]: {val_metrics['mAP_50_95']:.4f} (Best: {self.best_map:.4f})")

                # Save checkpoint
                save_checkpoint(
                    state={
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "best_map": self.best_map,
                        "config": self.config
                    },
                    is_best=is_best,
                    checkpoint_dir=self.save_dir
                )

                # Log metrics
                log_entry = {
                    "epoch": epoch + 1,
                    **train_metrics,
                    "val_mAP_50": val_metrics["mAP_50"],
                    "val_mAP_75": val_metrics["val_mAP_75"] if "val_mAP_75" in val_metrics else val_metrics["mAP_75"],
                    "val_mAP_90": val_metrics["mAP_90"],
                    "val_mAP_50_95": val_metrics["mAP_50_95"],
                    "lr": self.optimizer.param_groups[0]["lr"]
                }
                self.logger.log(log_entry)

                if self.early_stopping(current_map):
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

        self.logger.close()
        print(f"Training completed. Best mAP@[50:95]: {self.best_map:.4f}")
