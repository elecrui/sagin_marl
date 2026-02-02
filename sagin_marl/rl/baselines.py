from __future__ import annotations

import numpy as np


def zero_accel_policy(num_agents: int) -> np.ndarray:
    return np.zeros((num_agents, 2), dtype=np.float32)
