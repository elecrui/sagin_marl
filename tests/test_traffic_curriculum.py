from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv


def test_traffic_level_controls_effective_arrival_rate():
    cfg = SaginConfig(num_uav=1, num_gu=4, num_sat=2, users_obs_max=4, sats_obs_max=2, nbrs_obs_max=1)
    cfg.task_arrival_rate = 1e5
    cfg.task_arrival_poisson = False
    cfg.traffic_level_nav_ratio = 0.08
    cfg.traffic_level_easy_ratio = 0.5
    cfg.traffic_level_hard_ratio = 1.0

    for level, ratio in ((0, 0.08), (1, 0.5), (2, 1.0)):
        cfg.traffic_level = level
        env = SaginParallelEnv(cfg)
        _, infos = env.reset()
        expected = cfg.task_arrival_rate * ratio
        assert np.isclose(env.effective_task_arrival_rate, expected)
        first_info = infos[env.agents[0]]
        assert first_info["traffic_level"] == level
        assert np.isclose(first_info["effective_task_arrival_rate"], expected)

        actions = {
            agent: {
                "accel": np.zeros(2, dtype=np.float32),
                "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
                "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
            }
            for agent in env.agents
        }
        env.step(actions)
        assert np.isclose(env.last_arrival_rate, expected)
