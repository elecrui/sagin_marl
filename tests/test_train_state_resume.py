from __future__ import annotations

import numpy as np
import torch

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.mappo import _load_train_state, _save_train_state
from sagin_marl.utils.normalization import RunningMeanStd


def _step_linear(module: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor) -> None:
    loss = module(batch).pow(2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()


def _first_exp_avg(optim: torch.optim.Optimizer) -> torch.Tensor:
    state = optim.state_dict()["state"]
    assert state
    first = next(iter(state.values()))
    return first["exp_avg"].detach().clone()


def test_train_state_roundtrip_restores_optimizer_scheduler_and_stats(tmp_path):
    actor = torch.nn.Linear(3, 2)
    critic = torch.nn.Linear(4, 1)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=2e-3)
    actor_sched = torch.optim.lr_scheduler.LambdaLR(actor_optim, lr_lambda=lambda _: 1.0)
    critic_sched = torch.optim.lr_scheduler.LambdaLR(critic_optim, lr_lambda=lambda _: 1.0)

    _step_linear(actor, actor_optim, torch.randn(8, 3))
    _step_linear(critic, critic_optim, torch.randn(8, 4))
    actor_sched.step()
    critic_sched.step()

    reward_rms = RunningMeanStd()
    reward_rms.update(np.asarray([1.0, 2.0, 4.0], dtype=np.float64))

    _save_train_state(
        str(tmp_path),
        actor,
        critic,
        actor_optim,
        critic_optim,
        actor_sched,
        critic_sched,
        reward_rms,
        update=7,
        planned_total_updates=20,
        total_env_steps=123,
        reward_history=[1.0, 2.0, 3.0],
        best_ma=2.5,
        no_improve=1,
        checkpoint_eval_state={"prev_sat_drop_ratio": 0.2},
        checkpoint_eval_fixed_summary={"reward_sum": 3.0},
        total_time_sec=12.5,
    )

    actor_loaded = torch.nn.Linear(3, 2)
    critic_loaded = torch.nn.Linear(4, 1)
    actor_optim_loaded = torch.optim.Adam(actor_loaded.parameters(), lr=1e-3)
    critic_optim_loaded = torch.optim.Adam(critic_loaded.parameters(), lr=2e-3)
    actor_sched_loaded = torch.optim.lr_scheduler.LambdaLR(actor_optim_loaded, lr_lambda=lambda _: 1.0)
    critic_sched_loaded = torch.optim.lr_scheduler.LambdaLR(critic_optim_loaded, lr_lambda=lambda _: 1.0)
    reward_rms_loaded = RunningMeanStd()

    meta = _load_train_state(
        str(tmp_path / "train_state.pt"),
        actor=actor_loaded,
        critic=critic_loaded,
        actor_optim=actor_optim_loaded,
        critic_optim=critic_optim_loaded,
        actor_sched=actor_sched_loaded,
        critic_sched=critic_sched_loaded,
        reward_rms=reward_rms_loaded,
        device=torch.device("cpu"),
    )

    for p_saved, p_loaded in zip(actor.parameters(), actor_loaded.parameters()):
        assert torch.allclose(p_saved, p_loaded)
    for p_saved, p_loaded in zip(critic.parameters(), critic_loaded.parameters()):
        assert torch.allclose(p_saved, p_loaded)

    assert torch.allclose(_first_exp_avg(actor_optim), _first_exp_avg(actor_optim_loaded))
    assert torch.allclose(_first_exp_avg(critic_optim), _first_exp_avg(critic_optim_loaded))
    assert actor_sched_loaded.last_epoch == actor_sched.last_epoch
    assert critic_sched_loaded.last_epoch == critic_sched.last_epoch
    assert reward_rms_loaded.mean == reward_rms.mean
    assert reward_rms_loaded.var == reward_rms.var
    assert reward_rms_loaded.count == reward_rms.count

    assert meta["update"] == 7
    assert meta["planned_total_updates"] == 20
    assert meta["total_env_steps"] == 123
    assert meta["reward_history"] == [1.0, 2.0, 3.0]
    assert meta["best_ma"] == 2.5
    assert meta["no_improve"] == 1
    assert meta["checkpoint_eval_state"]["prev_sat_drop_ratio"] == 0.2
    assert meta["checkpoint_eval_fixed_summary"]["reward_sum"] == 3.0


def test_smoke_train_resume_uses_saved_train_state(tmp_path):
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    cfg.buffer_size = 5
    cfg.num_mini_batch = 1
    cfg.ppo_epochs = 1
    cfg.reward_norm_enabled = True

    env = SaginParallelEnv(cfg)
    try:
        train_dir = tmp_path / "resume_smoke"
        from sagin_marl.rl.mappo import train

        train(env, cfg, str(train_dir), total_updates=1)
    finally:
        env.close()

    train_state_path = train_dir / "train_state.pt"
    assert train_state_path.exists()

    env_resume = SaginParallelEnv(cfg)
    try:
        from sagin_marl.rl.mappo import train

        train(env_resume, cfg, str(train_dir), total_updates=1, resume_state_path=str(train_state_path))
    finally:
        env_resume.close()

    with (train_dir / "metrics.csv").open("r", encoding="utf-8") as f:
        rows = [line.strip().split(",")[0] for line in f.readlines()[1:]]
    assert rows[:2] == ["0", "1"]

    payload = torch.load(train_state_path, map_location="cpu")
    assert int(payload["update"]) == 2
