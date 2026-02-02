from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, cfg):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, cfg.critic_hidden)
        self.fc2 = nn.Linear(cfg.critic_hidden, cfg.critic_hidden)
        self.v = nn.Linear(cfg.critic_hidden, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x).squeeze(-1)
