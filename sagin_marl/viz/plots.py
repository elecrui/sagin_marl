from __future__ import annotations

import csv
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(csv_path: str, out_path: str) -> None:
    steps = []
    rewards = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            rewards.append(float(row["episode_reward"]))
    plt.figure(figsize=(6, 4))
    plt.plot(steps, rewards, label="Avg Reward")
    plt.xlabel("Update")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_trajectories(gu_pos: np.ndarray, uav_traj: List[np.ndarray], out_path: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(gu_pos[:, 0], gu_pos[:, 1], s=10, c="tab:blue", label="GU")
    for i, traj in enumerate(uav_traj):
        plt.plot(traj[:, 0], traj[:, 1], label=f"UAV {i}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("UAV Trajectories")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
