from __future__ import annotations

import numpy as np
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.policy import ActorNet, OWN_OBS_DIM, SAT_OBS_DIM, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_state_dict_forgiving


def _make_policy_obs(cfg: SaginConfig) -> dict[str, np.ndarray]:
    return {
        "own": np.zeros((OWN_OBS_DIM,), dtype=np.float32),
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }


def test_visible_sats_score_mode_tracks_raw_and_kept_candidates():
    cfg = SaginConfig(num_uav=1, num_gu=0, num_sat=4, users_obs_max=1, sats_obs_max=2, nbrs_obs_max=1)
    cfg.theta_min_deg = 0.0
    cfg.visible_sats_max = 2
    cfg.sat_candidate_mode = "score"
    env = SaginParallelEnv(cfg)

    elev = np.array([[0.9, 0.8, 0.7, -0.1]], dtype=np.float32)
    sat_pos = np.zeros((cfg.num_sat, 3), dtype=np.float32)

    def fake_elevation_matrix(sat_pos=None):
        return elev

    def fake_rank_data(u: int, sat_indices: np.ndarray, sat_pos: np.ndarray, elev_values=None):
        sat_idx = np.asarray(sat_indices, dtype=np.int32)
        elev_arr = np.asarray(elev_values, dtype=np.float32)
        rank_map = {0: -1.0, 1: 0.5, 2: 0.25, 3: 0.1}
        dist_map = {0: 0.2, 1: 0.4, 2: 0.8, 3: 1.2}
        se_map = {0: 0.9, 1: 0.8, 2: 0.3, 3: 0.1}
        queue_map = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
        rank_value = np.array([rank_map[int(idx)] for idx in sat_idx], dtype=np.float32)
        return {
            "elevation": elev_arr,
            "distance": np.array([dist_map[int(idx)] for idx in sat_idx], dtype=np.float32),
            "spectral_efficiency": np.array([se_map[int(idx)] for idx in sat_idx], dtype=np.float32),
            "queue_norm": np.array([queue_map[int(idx)] for idx in sat_idx], dtype=np.float32),
            "score": rank_value.copy(),
            "rank_value": rank_value,
        }

    env._get_elevation_matrix = fake_elevation_matrix
    env._sat_candidate_rank_data = fake_rank_data

    visible = env._visible_sats_sorted(sat_pos)

    assert visible == [[1, 2]]
    assert env.last_visible_raw_counts.tolist() == [3]
    assert env.last_visible_kept_counts.tolist() == [2]
    assert env.last_visible_raw_candidates[0] == [1, 2]
    assert env.last_visible_candidates[0] == [1, 2]
    assert np.isclose(env.last_visible_candidate_rank_gap_top1_top2[0], 0.25)
    assert np.isclose(env.last_visible_candidate_score_gap_top1_top2[0], 0.25)
    assert np.isclose(env.last_visible_stats["raw_visible_count_mean"], 3.0)
    assert np.isclose(env.last_visible_stats["kept_visible_count_mean"], 2.0)
    assert np.isclose(env.last_visible_stats["visible_truncation_fraction"], 1.0)


def test_cache_sat_obs_appends_link_load_features():
    cfg = SaginConfig(num_uav=2, num_gu=0, num_sat=3, users_obs_max=1, sats_obs_max=2, nbrs_obs_max=1)
    env = SaginParallelEnv(cfg)
    env.reset(seed=0)

    sat_pos, sat_vel = env._get_orbit_states()
    env.last_sat_selection = [[0], [0]]
    env.last_sat_connection_counts = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    env._cache_sat_obs(sat_pos, sat_vel, [[0, 1], [0, 1]])

    selected_feat = env._cached_sat_obs[0, 0]
    idle_feat = env._cached_sat_obs[0, 1]

    assert selected_feat.shape == (SAT_OBS_DIM,)
    np.testing.assert_allclose(selected_feat[9:12], np.array([1.0, 0.5, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(idle_feat[9:12], np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_cache_sat_obs_effective_bw_uses_projected_post_choice_share():
    cfg = SaginConfig(num_uav=3, num_gu=0, num_sat=2, users_obs_max=1, sats_obs_max=2, nbrs_obs_max=1)
    env = SaginParallelEnv(cfg)
    env.reset(seed=0)

    sat_pos, sat_vel = env._get_orbit_states()
    env.last_sat_selection = [[0], [], []]
    env.last_sat_connection_counts = np.array([1.0, 1.0], dtype=np.float32)
    env._cache_sat_obs(sat_pos, sat_vel, [[0, 1], [0, 1], [0, 1]])

    stay_feat = env._cached_sat_obs[0, 0]
    switch_feat = env._cached_sat_obs[0, 1]
    np.testing.assert_allclose(float(stay_feat[10]), 1.0, atol=1e-6)
    np.testing.assert_allclose(float(switch_feat[10]), 0.5, atol=1e-6)


def test_load_state_dict_forgiving_adapts_smaller_sat_encoder_inputs():
    torch.manual_seed(0)
    cfg = SaginConfig(users_obs_max=2, sats_obs_max=2, nbrs_obs_max=1)
    cfg.actor_encoder_type = "set_pool"
    cfg.actor_set_embed_dim = 8
    cfg.input_norm_enabled = False

    obs = _make_policy_obs(cfg)
    obs_dim = batch_flatten_obs([obs], cfg).shape[1]
    source_actor = ActorNet(obs_dim, cfg)
    target_actor = ActorNet(obs_dim, cfg)

    source_state = source_actor.state_dict()
    smaller_state = dict(source_state)
    smaller_state["sats_encoder.0.weight"] = source_state["sats_encoder.0.weight"][:, :9].clone()

    target_before = target_actor.state_dict()["sats_encoder.0.weight"].clone()
    info = load_state_dict_forgiving(target_actor, smaller_state)
    target_after = target_actor.state_dict()["sats_encoder.0.weight"]

    torch.testing.assert_close(target_after[:, :9], source_state["sats_encoder.0.weight"][:, :9])
    torch.testing.assert_close(target_after[:, 9:], target_before[:, 9:])
    assert any(key == "sats_encoder.0.weight" for key, _, _ in info["adapted_keys"])
