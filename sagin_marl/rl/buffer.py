from __future__ import annotations

from typing import List, Optional

import numpy as np


class RolloutBuffer:
    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = int(capacity) if capacity is not None else None
        self._use_list = self.capacity is None
        self._idx = 0

        self._obs: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._logprobs: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._dones: Optional[np.ndarray] = None
        self._global_states: Optional[np.ndarray] = None

        if self._use_list:
            self.obs: List[np.ndarray] = []
            self.actions: List[np.ndarray] = []
            self.logprobs: List[np.ndarray] = []
            self.rewards: List[float] = []
            self.values: List[float] = []
            self.dones: List[bool] = []
            self.global_states: List[np.ndarray] = []

    def _allocate(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        global_state: np.ndarray,
    ) -> None:
        if self.capacity is None:
            return
        cap = self.capacity
        self._obs = np.empty((cap,) + obs.shape, dtype=np.float32)
        self._actions = np.empty((cap,) + actions.shape, dtype=np.float32)
        self._logprobs = np.empty((cap,) + logprobs.shape, dtype=np.float32)
        self._rewards = np.empty((cap,), dtype=np.float32)
        self._values = np.empty((cap,), dtype=np.float32)
        self._dones = np.empty((cap,), dtype=np.float32)
        self._global_states = np.empty((cap,) + global_state.shape, dtype=np.float32)

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
        if self._use_list:
            self.obs.append(obs)
            self.actions.append(actions)
            self.logprobs.append(logprobs)
            self.rewards.append(float(reward))
            self.values.append(float(value))
            self.dones.append(bool(done))
            self.global_states.append(global_state)
            return

        if self._obs is None:
            self._allocate(obs, actions, logprobs, global_state)

        if self.capacity is not None and self._idx >= self.capacity:
            raise IndexError("RolloutBuffer capacity exceeded.")

        self._obs[self._idx] = obs
        self._actions[self._idx] = actions
        self._logprobs[self._idx] = logprobs
        self._rewards[self._idx] = float(reward)
        self._values[self._idx] = float(value)
        self._dones[self._idx] = float(done)
        self._global_states[self._idx] = global_state
        self._idx += 1

    def as_arrays(self):
        if self._use_list:
            return (
                np.stack(self.obs, axis=0),
                np.stack(self.actions, axis=0),
                np.stack(self.logprobs, axis=0),
                np.array(self.rewards, dtype=np.float32),
                np.array(self.values, dtype=np.float32),
                np.array(self.dones, dtype=np.float32),
                np.stack(self.global_states, axis=0),
            )
        end = self._idx
        return (
            self._obs[:end],
            self._actions[:end],
            self._logprobs[:end],
            self._rewards[:end],
            self._values[:end],
            self._dones[:end],
            self._global_states[:end],
        )
