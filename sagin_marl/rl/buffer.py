from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class RolloutBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    logprobs: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    global_states: List[np.ndarray] = field(default_factory=list)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        reward: float,
        value: float,
        done: bool,
        global_state: np.ndarray,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        self.global_states.append(global_state)

    def as_arrays(self):
        return (
            np.stack(self.obs, axis=0),
            np.stack(self.actions, axis=0),
            np.stack(self.logprobs, axis=0),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.stack(self.global_states, axis=0),
        )
