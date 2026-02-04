from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.mappo import train
from sagin_marl.utils.seeding import set_seed


def _resolve_log_dir(log_dir: str, run_dir: str | None, run_id: str | None) -> str:
    if run_dir:
        return run_dir
    if run_id:
        if run_id == "auto":
            run_id = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(log_dir, run_id)
    return log_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--log_dir", type=str, default="runs/phase1")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Full run directory. Overrides --log_dir/--run_id.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Subdirectory name under --log_dir. Use 'auto' for timestamp.",
    )
    parser.add_argument("--updates", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    env = SaginParallelEnv(cfg)
    log_dir = _resolve_log_dir(args.log_dir, args.run_dir, args.run_id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Run dir: {log_dir}")
    train(env, cfg, log_dir, total_updates=args.updates)


if __name__ == "__main__":
    main()
