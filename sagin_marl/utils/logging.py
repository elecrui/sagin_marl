from __future__ import annotations

import csv
import os
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self._init_csv()

    def _init_csv(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "episode_reward", "policy_loss", "value_loss", "entropy"])

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                metrics.get("episode_reward", 0.0),
                metrics.get("policy_loss", 0.0),
                metrics.get("value_loss", 0.0),
                metrics.get("entropy", 0.0),
            ])
        for key, val in metrics.items():
            self.writer.add_scalar(key, val, step)

    def close(self) -> None:
        self.writer.close()
