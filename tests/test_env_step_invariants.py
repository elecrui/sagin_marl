from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv


def test_env_step_invariants():
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    env = SaginParallelEnv(cfg)
    obs, _ = env.reset()
    actions = {agent: {"accel": np.zeros(2, dtype=np.float32),
                       "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
                       "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32)}
               for agent in env.agents}
    for _ in range(3):
        obs, rewards, terms, truncs, _ = env.step(actions)
        assert np.all(env.gu_queue >= 0)
        assert np.all(env.uav_queue >= 0)
        assert np.all(env.sat_queue >= 0)
        assert np.isfinite(env.gu_queue).all()
        assert np.isfinite(env.uav_queue).all()
        assert np.isfinite(env.sat_queue).all()
