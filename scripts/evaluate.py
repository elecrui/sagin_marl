from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import zero_accel_policy
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.progress import Progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/phase1/actor.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", type=str, default="runs/phase1/eval.csv")
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=["none", "zero_accel"],
        help="Use a baseline policy instead of a trained model.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = SaginParallelEnv(cfg)

    use_baseline = args.baseline != "none"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = None
    if not use_baseline:
        obs, _ = env.reset()
        obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
        actor = ActorNet(obs_dim, cfg).to(device)
        actor.load_state_dict(torch.load(args.checkpoint, map_location=device))
        actor.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "episode",
        "reward_sum",
        "steps",
        "episode_time_sec",
        "steps_per_sec",
        "gu_queue_mean",
        "uav_queue_mean",
        "sat_queue_mean",
        "gu_queue_max",
        "uav_queue_max",
        "sat_queue_max",
        "gu_drop_sum",
        "uav_drop_sum",
        "sat_processed_sum",
        "sat_incoming_sum",
        "energy_mean",
    ]
    progress = Progress(args.episodes, desc="Eval")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            reward_sum = 0.0
            steps = 0
            gu_queue_sum = 0.0
            uav_queue_sum = 0.0
            sat_queue_sum = 0.0
            gu_queue_max = 0.0
            uav_queue_max = 0.0
            sat_queue_max = 0.0
            gu_drop_sum = 0.0
            uav_drop_sum = 0.0
            sat_processed_sum = 0.0
            sat_incoming_sum = 0.0
            energy_mean_sum = 0.0
            ep_start = time.perf_counter()
            while not done:
                if use_baseline:
                    accel_actions = zero_accel_policy(len(env.agents))
                    actions = assemble_actions(cfg, env.agents, accel_actions)
                else:
                    obs_batch = batch_flatten_obs(list(obs.values()), cfg)
                    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        policy_out = actor.act(obs_tensor, deterministic=True)
                    accel_actions = policy_out.accel.cpu().numpy()
                    bw_logits = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
                    sat_logits = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None
                    actions = assemble_actions(cfg, env.agents, accel_actions, bw_logits=bw_logits, sat_logits=sat_logits)
                obs, rewards, terms, truncs, _ = env.step(actions)
                reward_sum += list(rewards.values())[0]
                done = list(terms.values())[0] or list(truncs.values())[0]
                steps += 1
                gu_queue_sum += float(np.mean(env.gu_queue))
                uav_queue_sum += float(np.mean(env.uav_queue))
                sat_queue_sum += float(np.mean(env.sat_queue))
                gu_queue_max = max(gu_queue_max, float(np.max(env.gu_queue)))
                uav_queue_max = max(uav_queue_max, float(np.max(env.uav_queue)))
                sat_queue_max = max(sat_queue_max, float(np.max(env.sat_queue)))
                gu_drop_sum += float(np.sum(env.gu_drop))
                uav_drop_sum += float(np.sum(env.uav_drop))
                if hasattr(env, "last_sat_processed"):
                    sat_processed_sum += float(np.sum(env.last_sat_processed))
                if hasattr(env, "last_sat_incoming"):
                    sat_incoming_sum += float(np.sum(env.last_sat_incoming))
                if cfg.energy_enabled:
                    energy_mean_sum += float(np.mean(env.uav_energy))
            ep_time = time.perf_counter() - ep_start
            steps = max(1, steps)
            writer.writerow(
                [
                    ep,
                    reward_sum,
                    steps,
                    ep_time,
                    steps / max(1e-9, ep_time),
                    gu_queue_sum / steps,
                    uav_queue_sum / steps,
                    sat_queue_sum / steps,
                    gu_queue_max,
                    uav_queue_max,
                    sat_queue_max,
                    gu_drop_sum,
                    uav_drop_sum,
                    sat_processed_sum,
                    sat_incoming_sum,
                    (energy_mean_sum / steps) if cfg.energy_enabled else 0.0,
                ]
            )
            progress.update(ep + 1)
    progress.close()
    print(f"Saved evaluation metrics to {args.out}")


if __name__ == "__main__":
    main()
