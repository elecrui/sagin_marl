from __future__ import annotations

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.vec_env import make_vec_env
from sagin_marl.rl.mappo import train


def test_smoke_train_vec_sync(tmp_path):
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    cfg.buffer_size = 5
    cfg.num_mini_batch = 1
    cfg.ppo_epochs = 1
    cfg.early_stop_enabled = False
    env = make_vec_env(cfg, num_envs=2, backend="sync")
    try:
        train(env, cfg, str(tmp_path), total_updates=1)
    finally:
        env.close()
