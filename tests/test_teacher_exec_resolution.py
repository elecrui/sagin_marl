from __future__ import annotations

import numpy as np
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.rl.mappo import _configure_actor_trainability, _resolve_frozen_teacher_exec_sources
from sagin_marl.rl.policy import ActorNet, OWN_OBS_DIM, SAT_OBS_DIM, batch_flatten_obs


def _make_obs(cfg: SaginConfig) -> dict[str, np.ndarray]:
    own = np.zeros((OWN_OBS_DIM,), dtype=np.float32)
    obs = {
        "own": own,
        "users": np.zeros((cfg.users_obs_max, 5), dtype=np.float32),
        "users_mask": np.zeros((cfg.users_obs_max,), dtype=np.float32),
        "sats": np.zeros((cfg.sats_obs_max, SAT_OBS_DIM), dtype=np.float32),
        "sats_mask": np.zeros((cfg.sats_obs_max,), dtype=np.float32),
        "nbrs": np.zeros((cfg.nbrs_obs_max, 4), dtype=np.float32),
        "nbrs_mask": np.zeros((cfg.nbrs_obs_max,), dtype=np.float32),
    }
    if cfg.danger_nbr_enabled:
        obs["danger_nbr"] = np.zeros((5,), dtype=np.float32)
    return obs


def _make_actor(cfg: SaginConfig) -> ActorNet:
    obs_dim = batch_flatten_obs([_make_obs(cfg)], cfg).shape[1]
    actor = ActorNet(obs_dim, cfg)
    actor.eval()
    return actor


def test_frozen_teacher_accel_exec_can_resolve_to_policy_when_backbone_matches():
    torch.manual_seed(0)
    cfg = SaginConfig(
        users_obs_max=3,
        sats_obs_max=2,
        nbrs_obs_max=2,
        enable_bw_action=True,
        fixed_satellite_strategy=False,
    )
    cfg.actor_encoder_type = "set_pool"
    cfg.actor_set_embed_dim = 16
    cfg.train_shared_backbone = False

    actor = _make_actor(cfg)
    teacher_actor = _make_actor(cfg)
    teacher_actor.load_state_dict(actor.state_dict())

    train_heads = {"accel": False, "bw": True, "sat": False}
    _configure_actor_trainability(actor, cfg, train_heads)

    with torch.no_grad():
        actor.mu_head.weight.add_(1.0)
        actor.log_std.add_(0.5)

    exec_accel_source, exec_bw_source, exec_sat_source, resolved, kept = _resolve_frozen_teacher_exec_sources(
        actor,
        teacher_actor,
        train_heads,
        "teacher",
        "policy",
        "zero",
    )

    assert exec_accel_source == "policy"
    assert exec_bw_source == "policy"
    assert exec_sat_source == "zero"
    assert resolved == ["accel"]
    assert kept == []
    torch.testing.assert_close(actor.mu_head.weight, teacher_actor.mu_head.weight)
    torch.testing.assert_close(actor.mu_head.bias, teacher_actor.mu_head.bias)
    torch.testing.assert_close(actor.log_std, teacher_actor.log_std)


def test_teacher_exec_stays_when_backbone_mismatches_even_if_head_is_frozen():
    torch.manual_seed(0)
    cfg = SaginConfig(
        users_obs_max=3,
        sats_obs_max=2,
        nbrs_obs_max=2,
        enable_bw_action=True,
        fixed_satellite_strategy=False,
    )
    cfg.actor_encoder_type = "set_pool"
    cfg.actor_set_embed_dim = 16
    cfg.train_shared_backbone = False

    actor = _make_actor(cfg)
    teacher_actor = _make_actor(cfg)
    teacher_actor.load_state_dict(actor.state_dict())

    train_heads = {"accel": False, "bw": True, "sat": False}
    _configure_actor_trainability(actor, cfg, train_heads)

    with torch.no_grad():
        first_backbone_module = actor.backbone_modules()[0]
        first_param = next(first_backbone_module.parameters())
        first_param.add_(1.0)

    exec_accel_source, _, _, resolved, kept = _resolve_frozen_teacher_exec_sources(
        actor,
        teacher_actor,
        train_heads,
        "teacher",
        "policy",
        "zero",
    )

    assert exec_accel_source == "teacher"
    assert resolved == []
    assert kept == ["accel(backbone_mismatch)"]
