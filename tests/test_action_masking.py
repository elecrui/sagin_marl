from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv


def test_action_masking_fixed_satellite():
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    cfg.fixed_satellite_strategy = True
    env = SaginParallelEnv(cfg)
    obs, _ = env.reset()
    actions = {agent: {"accel": np.random.uniform(-1, 1, size=2).astype(np.float32),
                       "bw_logits": np.random.randn(cfg.users_obs_max).astype(np.float32),
                       "sat_logits": np.random.randn(cfg.sats_obs_max).astype(np.float32)}
               for agent in env.agents}
    obs, rewards, terms, truncs, _ = env.step(actions)
    assert len(obs) == cfg.num_uav
