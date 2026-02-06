from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml

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


def _save_config(log_dir: str, cfg, config_path: str) -> None:
    try:
        data = asdict(cfg)
        data["_config_source"] = config_path
        out_path = os.path.join(log_dir, "config.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        if os.path.isfile(config_path):
            shutil.copy2(config_path, os.path.join(log_dir, "config_source.yaml"))
    except Exception as exc:
        print(f"Warning: failed to save config in {log_dir}: {exc}")


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
    parser.add_argument("--init_actor", type=str, default=None, help="Init actor checkpoint path.")
    parser.add_argument("--init_critic", type=str, default=None, help="Init critic checkpoint path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    env = SaginParallelEnv(cfg)
    log_dir = _resolve_log_dir(args.log_dir, args.run_dir, args.run_id)
    os.makedirs(log_dir, exist_ok=True)
    _save_config(log_dir, cfg, os.path.abspath(args.config))
    print(f"Run dir: {log_dir}")
    train(
        env,
        cfg,
        log_dir,
        total_updates=args.updates,
        init_actor_path=args.init_actor,
        init_critic_path=args.init_critic,
    )


if __name__ == "__main__":
    main()
