from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/phase1/actor.pt")
    parser.add_argument("--out", type=str, default="runs/phase1/episode.gif")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = SaginParallelEnv(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset()
    obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
    actor = ActorNet(obs_dim, cfg).to(device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=device))
    actor.eval()

    frames = []
    done = False
    while not done:
        frame = env.render(mode="rgb_array")
        frames.append(frame)

        obs_batch = batch_flatten_obs(list(obs.values()), cfg)
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            policy_out = actor.act(obs_tensor, deterministic=True)
        actions = {}
        for i, agent in enumerate(env.agents):
            act = {
                "accel": policy_out.action[i].cpu().numpy(),
                "bw_logits": np.zeros(cfg.users_obs_max, dtype=np.float32),
                "sat_logits": np.zeros(cfg.sats_obs_max, dtype=np.float32),
            }
            if cfg.enable_bw_action and policy_out.bw_logits is not None:
                act["bw_logits"] = policy_out.bw_logits[i].cpu().numpy()
            if not cfg.fixed_satellite_strategy and policy_out.sat_logits is not None:
                act["sat_logits"] = policy_out.sat_logits[i].cpu().numpy()
            actions[agent] = act
        obs, rewards, terms, truncs, _ = env.step(actions)
        done = list(terms.values())[0] or list(truncs.values())[0]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved render to {args.out}")


if __name__ == "__main__":
    main()
