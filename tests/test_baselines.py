from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.baselines import queue_aware_policy, zero_accel_policy


def test_zero_accel_policy_shape_dtype():
    actions = zero_accel_policy(3)
    assert actions.shape == (3, 2)
    assert actions.dtype == np.float32
    assert np.all(actions == 0.0)


def test_queue_aware_policy_shapes():
    cfg = SaginConfig()
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False
    obs = {
        "own": np.zeros((7,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, 9), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["users_mask"][0] = 1.0
    obs["users"][0, 0:2] = np.array([0.2, -0.1], dtype=np.float32)
    obs["users"][0, 2] = 0.5
    obs["users"][0, 3] = 1.0
    obs["users"][0, 4] = 1.0
    obs["sats_mask"][0] = 1.0
    obs["sats"][0, 7] = 2.0
    obs["sats"][0, 8] = 0.1

    accel, bw_logits, sat_logits = queue_aware_policy([obs, obs], cfg)
    assert accel.shape == (2, 2)
    assert bw_logits.shape == (2, cfg.users_obs_max)
    assert sat_logits.shape == (2, cfg.sats_obs_max)
    assert accel.dtype == np.float32
    assert bw_logits.dtype == np.float32
    assert sat_logits.dtype == np.float32
    assert np.all(np.isfinite(accel))
    assert np.all(np.isfinite(bw_logits))
    assert np.all(np.isfinite(sat_logits))
    assert np.max(np.abs(accel)) <= 1.0 + 1e-6
    assert np.max(np.abs(bw_logits)) <= cfg.bw_logit_scale + 1e-6
    assert np.max(np.abs(sat_logits)) <= cfg.sat_logit_scale + 1e-6
