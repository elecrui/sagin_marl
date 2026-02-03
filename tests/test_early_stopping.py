from __future__ import annotations

import csv

from sagin_marl.env.config import SaginConfig
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.mappo import train


def test_early_stopping_triggers(tmp_path):
    cfg = SaginConfig(num_uav=2, num_gu=5, num_sat=3, users_obs_max=5, sats_obs_max=3, nbrs_obs_max=1)
    cfg.buffer_size = 5
    cfg.num_mini_batch = 1
    cfg.ppo_epochs = 1
    cfg.early_stop_enabled = True
    cfg.early_stop_min_updates = 1
    cfg.early_stop_window = 1
    cfg.early_stop_patience = 1
    cfg.early_stop_min_delta = 1e9

    env = SaginParallelEnv(cfg)
    train(env, cfg, str(tmp_path), total_updates=5)

    metrics_path = tmp_path / "metrics.csv"
    with open(metrics_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # header + updates; early stop should end before total_updates
    assert len(rows) - 1 < 5
