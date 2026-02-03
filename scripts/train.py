from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv
from sagin_marl.rl.mappo import train
from sagin_marl.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--log_dir", type=str, default="runs/phase1")
    parser.add_argument("--updates", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    env = SaginParallelEnv(cfg)
    os.makedirs(args.log_dir, exist_ok=True)
    train(env, cfg, args.log_dir, total_updates=args.updates)


if __name__ == "__main__":
    main()
