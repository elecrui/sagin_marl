from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def zero_accel_policy(num_agents: int) -> np.ndarray:
    return np.zeros((num_agents, 2), dtype=np.float32)


def random_accel_policy(
    num_agents: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.uniform(-1.0, 1.0, size=(num_agents, 2)).astype(np.float32)


def centroid_accel_policy(
    obs_list: List[Dict[str, np.ndarray]],
    gain: float = 2.0,
    queue_weighted: bool = True,
) -> np.ndarray:
    num_agents = len(obs_list)
    accel = np.zeros((num_agents, 2), dtype=np.float32)
    for i, obs in enumerate(obs_list):
        users = obs["users"]
        users_mask = obs["users_mask"] > 0.0
        if not np.any(users_mask):
            continue
        rel = users[users_mask, 0:2]
        vec = np.mean(rel, axis=0)
        if queue_weighted and users.shape[1] >= 3:
            q = np.clip(users[users_mask, 2], 0.0, None)
            q_sum = float(np.sum(q))
            if q_sum > 1e-6:
                vec = (rel * q[:, None]).sum(axis=0) / (q_sum + 1e-9)
        accel[i] = np.clip(vec * gain, -1.0, 1.0).astype(np.float32)
    return accel

def uniform_bw_policy(num_agents: int, users_obs_max: int) -> np.ndarray:
    return np.zeros((num_agents, users_obs_max), dtype=np.float32)

def random_bw_policy(
    num_agents: int,
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.uniform(-cfg.bw_logit_scale, cfg.bw_logit_scale, size=(num_agents, cfg.users_obs_max)).astype(np.float32)

def uniform_sat_policy(num_agents: int, sats_obs_max: int) -> np.ndarray:
    return np.zeros((num_agents, sats_obs_max), dtype=np.float32)

def random_sat_policy(
    num_agents: int,
    cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.uniform(-cfg.sat_logit_scale, cfg.sat_logit_scale, size=(num_agents, cfg.sats_obs_max)).astype(np.float32)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return vals
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    if v_max - v_min <= 1e-9:
        return np.zeros_like(vals, dtype=np.float32)
    return ((vals - v_min) / (v_max - v_min)).astype(np.float32, copy=False)


def _sat_heuristic_score(sats: np.ndarray, mask: np.ndarray, cfg) -> np.ndarray:
    score = np.zeros((cfg.sats_obs_max,), dtype=np.float32)
    if not np.any(mask):
        return score

    sat_feat = sats[mask]
    se = np.asarray(sat_feat[:, 7], dtype=np.float32)
    qsat = np.asarray(sat_feat[:, 8], dtype=np.float32)
    load_norm = np.asarray(sat_feat[:, 9], dtype=np.float32)
    bw_ratio = np.asarray(sat_feat[:, 10], dtype=np.float32)
    stay = np.asarray(sat_feat[:, 11], dtype=np.float32)

    se_weight = float(getattr(cfg, "baseline_sat_se_weight", 1.0) or 0.0)
    q_penalty = float(getattr(cfg, "baseline_sat_queue_penalty", 0.5) or 0.0)
    load_penalty = float(getattr(cfg, "baseline_sat_load_penalty", 1.0) or 0.0)
    bw_reward = float(getattr(cfg, "baseline_sat_bw_reward", 0.75) or 0.0)
    stay_bonus = float(getattr(cfg, "baseline_sat_stay_bonus", 0.25) or 0.0)
    switch_margin = max(float(getattr(cfg, "baseline_sat_switch_margin", 0.15) or 0.0), 0.0)

    projected_count = 1.0 / np.clip(bw_ratio, 1e-6, 1.0)
    projected_load_term = np.log1p(projected_count)

    se_norm = _minmax_normalize(se)
    q_norm = _minmax_normalize(qsat)
    load_term_norm = _minmax_normalize(projected_load_term)
    bw_norm = _minmax_normalize(bw_ratio)

    score_slice = (
        se_weight * se_norm
        - q_penalty * q_norm
        - load_penalty * load_term_norm
        + bw_reward * bw_norm
        + stay_bonus * stay
    ).astype(np.float32, copy=False)

    current_idx = np.flatnonzero(stay > 0.5)
    if cfg.N_RF == 1 and current_idx.size == 1:
        cur = int(current_idx[0])
        best = int(np.argmax(score_slice))
        if best != cur and score_slice[best] <= score_slice[cur] + switch_margin:
            score_slice[cur] = score_slice[best] + 1e-3

    score[mask] = np.clip(score_slice, -cfg.sat_logit_scale, cfg.sat_logit_scale)
    return score

def queue_aware_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Queue + channel aware heuristic baseline.

    - Accel: move toward weighted centroid of high-queue users.
    - BW: softmax weights ~ queue * (0.5 + eta), with mild assoc bonus.
    - Sat: score by link quality, backlog, expected contention, and stay bonus.
    - Safety: repel from nearby neighbors.
    - Energy: slow down when energy is low (if enabled).
    """

    num_agents = len(obs_list)
    accel = np.zeros((num_agents, 2), dtype=np.float32)
    bw_logits = np.zeros((num_agents, cfg.users_obs_max), dtype=np.float32)
    sat_logits = np.zeros((num_agents, cfg.sats_obs_max), dtype=np.float32)

    accel_gain = float(getattr(cfg, "baseline_accel_gain", 2.0))
    assoc_bonus = float(getattr(cfg, "baseline_assoc_bonus", 0.3))
    repulse_gain = float(getattr(cfg, "baseline_repulse_gain", 0.0))
    repulse_radius_factor = float(getattr(cfg, "baseline_repulse_radius_factor", 1.5))
    energy_low = float(getattr(cfg, "baseline_energy_low", 0.3))
    energy_weight = float(getattr(cfg, "baseline_energy_weight", 1.0))
    repulse_radius = float(cfg.d_safe) * repulse_radius_factor if repulse_radius_factor > 0 else 0.0

    for i, obs in enumerate(obs_list):
        accel_vec = np.zeros((2,), dtype=np.float32)
        users = obs["users"]
        users_mask = obs["users_mask"] > 0.0
        if np.any(users_mask):
            rel = users[users_mask, 0:2]
            q = users[users_mask, 2]
            eta = users[users_mask, 3]
            prev = users[users_mask, 4]
            weights = q * (0.5 + eta)
            if assoc_bonus > 0.0:
                weights = weights * (1.0 + assoc_bonus * prev)

            weight_sum = float(np.sum(weights))
            if weight_sum > 1e-6:
                vec = (rel * weights[:, None]).sum(axis=0) / (weight_sum + 1e-9)
                accel_vec = accel_vec + vec * accel_gain

            if cfg.enable_bw_action:
                logits = np.zeros((cfg.users_obs_max,), dtype=np.float32)
                logits_slice = np.log(weights + 1e-6)
                logits_slice = np.clip(logits_slice, -cfg.bw_logit_scale, cfg.bw_logit_scale)
                logits[users_mask] = logits_slice
                bw_logits[i] = logits

        if not cfg.fixed_satellite_strategy:
            sats = obs["sats"]
            sats_mask = obs["sats_mask"] > 0.0
            if np.any(sats_mask):
                sat_logits[i] = _sat_heuristic_score(sats, sats_mask, cfg)

        if repulse_gain > 0.0 and repulse_radius > 0.0:
            nbrs = obs["nbrs"]
            nbrs_mask = obs["nbrs_mask"] > 0.0
            if np.any(nbrs_mask):
                rel = nbrs[nbrs_mask, 0:2]
                dist_norm = np.linalg.norm(rel, axis=1)
                dist = dist_norm * cfg.map_size
                mask = (dist > 1e-6) & (dist < repulse_radius)
                if np.any(mask):
                    rel_sel = rel[mask]
                    dist_sel = dist[mask]
                    dist_norm_sel = dist_norm[mask]
                    direction = rel_sel / dist_norm_sel[:, None]
                    strength = (1.0 / dist_sel - 1.0 / repulse_radius)
                    accel_vec = accel_vec + repulse_gain * (direction * strength[:, None]).sum(axis=0)

        if cfg.energy_enabled and energy_weight > 0.0:
            energy_norm = float(obs["own"][4])
            if energy_norm < energy_low:
                vel = obs["own"][2:4].astype(np.float32)
                speed = float(np.linalg.norm(vel))
                if speed > 1e-6:
                    target_speed = min(cfg.uav_opt_speed / max(cfg.v_max, 1e-6), 1.0)
                    delta = target_speed - speed
                    if delta < 0.0:
                        scale = (energy_low - energy_norm) / max(energy_low, 1e-6)
                        accel_vec = accel_vec + energy_weight * scale * (vel / speed) * delta

        accel[i] = np.clip(accel_vec, -1.0, 1.0)

    return accel, bw_logits, sat_logits

def queue_aware_bw_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    _, bw_logits, _ = queue_aware_policy(obs_list, cfg)
    return bw_logits

def queue_aware_sat_policy(
    obs_list: List[Dict[str, np.ndarray]],
    cfg,
) -> np.ndarray:
    _, _, sat_logits = queue_aware_policy(obs_list, cfg)
    return sat_logits
