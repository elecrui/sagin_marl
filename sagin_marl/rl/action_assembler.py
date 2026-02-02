from __future__ import annotations

from typing import Dict

import numpy as np


def assemble_actions(cfg, agents, accel_actions, bw_logits=None, sat_logits=None) -> Dict[str, Dict]:
    actions = {}
    for i, agent in enumerate(agents):
        act = {
            "accel": accel_actions[i].astype(np.float32),
            "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
            "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
        }
        if cfg.enable_bw_action and bw_logits is not None:
            act["bw_logits"] = np.clip(
                bw_logits[i].astype(np.float32), -cfg.bw_logit_scale, cfg.bw_logit_scale
            )
        if not cfg.fixed_satellite_strategy and sat_logits is not None:
            act["sat_logits"] = np.clip(
                sat_logits[i].astype(np.float32), -cfg.sat_logit_scale, cfg.sat_logit_scale
            )
        actions[agent] = act
    return actions
