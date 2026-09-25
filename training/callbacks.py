"""
Training Callbacks:
Early stopping, CSV experiment metrics logging, and progress reporting.
"""

import os
import csv
from typing import Dict, Any


class CSVLogger:
    def __init__(self, log_dir: str, filename: str = "training_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = None
        self.file = None
        self.writer = None

    def log(self, metrics: Dict[str, Any]):
        if self.fieldnames is None:
            self.fieldnames = list(metrics.keys())
            self.file = open(self.filepath, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()

        self.writer.writerow(metrics)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
