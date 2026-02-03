from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv


def test_sat_queue_tracking_consistency():
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    cfg.task_arrival_poisson = False
    env = SaginParallelEnv(cfg)
    env.reset()

    actions = {
        agent: {
            "accel": np.zeros(2, dtype=np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        for agent in env.agents
    }

    before = env.sat_queue.copy()
    env.step(actions)

    incoming = env.last_sat_incoming
    processed = env.last_sat_processed

    assert incoming.shape == (cfg.num_sat,)
    assert processed.shape == (cfg.num_sat,)
    assert np.all(incoming >= 0)
    assert np.all(processed >= 0)

    compute_cap = cfg.sat_cpu_freq / cfg.task_cycles_per_bit * cfg.tau0
    assert np.all(processed <= before + incoming + 1e-6)
    assert np.all(processed <= compute_cap + 1e-6)

    expected = np.maximum(before + incoming - compute_cap, 0.0)
    np.testing.assert_allclose(env.sat_queue, expected, rtol=1e-5, atol=1e-6)
