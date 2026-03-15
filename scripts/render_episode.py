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
from sagin_marl.rl.action_assembler import assemble_actions
from sagin_marl.rl.baselines import (
    centroid_accel_policy,
    queue_aware_policy,
    queue_aware_bw_policy,
    queue_aware_sat_policy,
    random_accel_policy,
    random_bw_policy,
    random_sat_policy,
    uniform_bw_policy,
    uniform_sat_policy,
    zero_accel_policy,
)
from sagin_marl.rl.policy import ActorNet, batch_flatten_obs
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving


def _resolve_render_paths(
    run_dir: str | None, checkpoint: str | None, out: str | None, baseline: str
) -> tuple[str | None, str]:
    use_baseline = baseline != "none"
    if run_dir:
        if not use_baseline:
            checkpoint = checkpoint or os.path.join(run_dir, "actor.pt")
        if out is None:
            filename = "episode.gif" if not use_baseline else f"episode_{baseline}.gif"
            out = os.path.join(run_dir, filename)
    else:
        if not use_baseline:
            checkpoint = checkpoint or "runs/phase1/actor.pt"
        if out is None:
            filename = "episode.gif" if not use_baseline else f"episode_{baseline}.gif"
            out = os.path.join("runs/phase1", filename)
    return checkpoint, out


def _baseline_actions(
    baseline: str,
    obs_list,
    cfg,
    num_agents: int,
    rng: np.random.Generator | None = None,
):
    if baseline in ("zero_accel", "fixed"):
        return zero_accel_policy(num_agents), None, None
    if baseline == "random_accel":
        return random_accel_policy(num_agents, rng=rng), None, None
    if baseline == "centroid":
        gain = float(getattr(cfg, "baseline_centroid_gain", 2.0))
        queue_weighted = bool(getattr(cfg, "baseline_centroid_queue_weighted", True))
        return centroid_accel_policy(obs_list, gain=gain, queue_weighted=queue_weighted), None, None
    if baseline == "uniform_bw":
        return zero_accel_policy(num_agents), uniform_bw_policy(num_agents, cfg.users_obs_max), None
    if baseline == "random_bw":
        return zero_accel_policy(num_agents), random_bw_policy(num_agents, cfg, rng=rng), None
    if baseline == "queue_aware_bw":
        return zero_accel_policy(num_agents), queue_aware_bw_policy(obs_list, cfg), None
    if baseline == "uniform_sat":
        return zero_accel_policy(num_agents), None, uniform_sat_policy(num_agents, cfg.sats_obs_max)
    if baseline == "random_sat":
        return zero_accel_policy(num_agents), None, random_sat_policy(num_agents, cfg, rng=rng)
    if baseline == "queue_aware_sat":
        return zero_accel_policy(num_agents), None, queue_aware_sat_policy(obs_list, cfg)
    if baseline == "queue_aware":
        return queue_aware_policy(obs_list, cfg)
    raise ValueError(f"Unknown baseline: {baseline}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory that contains checkpoints and render outputs.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=[
            "none",
            "fixed",
            "zero_accel",
            "random_accel",
            "centroid",
            "queue_aware",
            "uniform_bw",
            "random_bw",
            "queue_aware_bw",
            "uniform_sat",
            "random_sat",
            "queue_aware_sat",
        ],
        help="Render a baseline policy instead of a trained checkpoint.",
    )
    parser.add_argument(
        "--episode_seed",
        type=int,
        default=None,
        help="Reset seed for the rendered episode. Match evaluate.py --episode_seed_base + episode index to replay an eval rollout.",
    )
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    args.checkpoint, args.out = _resolve_render_paths(
        args.run_dir, args.checkpoint, args.out, args.baseline
    )

    cfg = load_config(args.config)
    env = SaginParallelEnv(cfg)
    use_baseline = args.baseline != "none"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs, _ = env.reset(seed=args.episode_seed)
    actor = None
    if not use_baseline:
        obs_dim = batch_flatten_obs(list(obs.values()), cfg).shape[1]
        actor = ActorNet(obs_dim, cfg).to(device)
        info = load_checkpoint_forgiving(actor, args.checkpoint, map_location=device, strict=True)
        if info.get("adapted_keys"):
            print(f"Loaded actor with adapted tensors from {args.checkpoint}: {len(info['adapted_keys'])}")
        actor.eval()

    frames = []
    done = False
    while not done:
        frame = env.render(mode="rgb_array")
        frames.append(frame)

        obs_list = list(obs.values())
        if use_baseline:
            accel_actions, bw_logits, sat_logits = _baseline_actions(
                args.baseline,
                obs_list,
                cfg,
                len(env.agents),
                rng=env.rng,
            )
        else:
            obs_batch = batch_flatten_obs(obs_list, cfg)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                policy_out = actor.act(obs_tensor, deterministic=True)
            accel_actions = policy_out.accel.cpu().numpy()
            bw_logits = policy_out.bw_logits.cpu().numpy() if policy_out.bw_logits is not None else None
            sat_logits = policy_out.sat_logits.cpu().numpy() if policy_out.sat_logits is not None else None
        actions = assemble_actions(
            cfg,
            env.agents,
            accel_actions,
            bw_logits=bw_logits,
            sat_logits=sat_logits,
        )
        obs, rewards, terms, truncs, _ = env.step(actions)
        done = list(terms.values())[0] or list(truncs.values())[0]

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved render to {args.out}")


if __name__ == "__main__":
    main()
