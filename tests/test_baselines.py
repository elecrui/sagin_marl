from __future__ import annotations

import numpy as np

from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.baselines import (
    centroid_accel_policy,
    queue_aware_policy,
    random_accel_policy,
    zero_accel_policy,
)
from sagin_marl.rl.policy import SAT_OBS_DIM


def test_zero_accel_policy_shape_dtype():
    actions = zero_accel_policy(3)
    assert actions.shape == (3, 2)
    assert actions.dtype == np.float32
    assert np.all(actions == 0.0)


def test_random_accel_policy_shape_dtype_and_range():
    actions = random_accel_policy(4, rng=np.random.default_rng(123))
    assert actions.shape == (4, 2)
    assert actions.dtype == np.float32
    assert np.max(actions) <= 1.0 + 1e-6
    assert np.min(actions) >= -1.0 - 1e-6


def test_centroid_accel_policy_points_to_users():
    obs = {
        "own": np.zeros((7,), dtype=np.float32),
        "users": np.zeros((3, 5), dtype=np.float32),
        "users_mask": np.zeros((3,), dtype=np.float32),
        "sats": np.zeros((1, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((1,), dtype=np.float32),
        "nbrs": np.zeros((1, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((1,), dtype=np.float32),
    }
    obs["users_mask"][0] = 1.0
    obs["users"][0, 0:2] = np.array([0.4, -0.2], dtype=np.float32)
    obs["users"][0, 2] = 1.0

    accel = centroid_accel_policy([obs], gain=2.0, queue_weighted=True)
    assert accel.shape == (1, 2)
    assert accel.dtype == np.float32
    assert accel[0, 0] > 0.0
    assert accel[0, 1] < 0.0
    assert np.max(np.abs(accel)) <= 1.0 + 1e-6


def test_queue_aware_policy_shapes():
    cfg = SaginConfig()
    cfg.enable_bw_action = True
    cfg.fixed_satellite_strategy = False
    obs = {
        "own": np.zeros((7,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
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


def test_queue_aware_sat_uses_load_bw_and_stay_features():
    cfg = SaginConfig(num_uav=3, sats_obs_max=2)
    cfg.fixed_satellite_strategy = False
    obs = {
        "own": np.zeros((7,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["sats_mask"][:] = 1.0
    obs["sats"][0, 7] = 5.0
    obs["sats"][0, 8] = 0.5
    obs["sats"][0, 9] = 2.0 / 3.0
    obs["sats"][0, 10] = 0.5
    obs["sats"][0, 11] = 0.0

    obs["sats"][1, 7] = 4.4
    obs["sats"][1, 8] = 0.0
    obs["sats"][1, 9] = 0.0
    obs["sats"][1, 10] = 1.0
    obs["sats"][1, 11] = 1.0

    _, _, sat_logits = queue_aware_policy([obs], cfg)
    assert sat_logits.shape == (1, cfg.sats_obs_max)
    assert sat_logits[0, 1] > sat_logits[0, 0]


def test_queue_aware_sat_prefers_current_sat_within_switch_margin():
    cfg = SaginConfig(num_uav=3, sats_obs_max=2, N_RF=1)
    cfg.fixed_satellite_strategy = False
    cfg.baseline_sat_switch_margin = 0.2
    obs = {
        "own": np.zeros((7,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    obs["sats_mask"][:] = 1.0
    obs["sats"][0, 7] = 4.8
    obs["sats"][0, 8] = 0.2
    obs["sats"][0, 9] = 1.0 / 3.0
    obs["sats"][0, 10] = 1.0
    obs["sats"][0, 11] = 1.0

    obs["sats"][1, 7] = 4.9
    obs["sats"][1, 8] = 0.15
    obs["sats"][1, 9] = 1.0 / 3.0
    obs["sats"][1, 10] = 0.5
    obs["sats"][1, 11] = 0.0

    _, _, sat_logits = queue_aware_policy([obs], cfg)
    assert sat_logits[0, 0] > sat_logits[0, 1]
