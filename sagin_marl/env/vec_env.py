from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from multiprocessing.connection import Connection
from typing import Dict, List, Sequence

import numpy as np

from .config import SaginConfig
from .sagin_env import SaginParallelEnv


def _done_from_flags(terms: Dict[str, bool], truncs: Dict[str, bool]) -> bool:
    if not terms:
        return False
    first_agent = next(iter(terms))
    return bool(terms[first_agent] or truncs[first_agent])


def _collect_step_stats(env: SaginParallelEnv) -> Dict[str, object]:
    def _safe_mean(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.mean(a)) if a.size > 0 else 0.0

    def _safe_max(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.max(a)) if a.size > 0 else 0.0

    def _safe_sum(arr) -> float:
        a = np.asarray(arr, dtype=np.float32)
        return float(np.sum(a)) if a.size > 0 else 0.0

    num_agents = len(getattr(env, "agents", []))
    default_accel = np.zeros((num_agents, 2), dtype=np.float32)
    parts = getattr(env, "last_reward_parts", None) or {}
    sat_processed = getattr(env, "last_sat_processed", None)
    sat_incoming = getattr(env, "last_sat_incoming", None)

    return {
        "last_exec_accel": np.asarray(getattr(env, "last_exec_accel", default_accel), dtype=np.float32),
        "gu_queue_mean": _safe_mean(getattr(env, "gu_queue", 0.0)),
        "uav_queue_mean": _safe_mean(getattr(env, "uav_queue", 0.0)),
        "sat_queue_mean": _safe_mean(getattr(env, "sat_queue", 0.0)),
        "gu_queue_max": _safe_max(getattr(env, "gu_queue", 0.0)),
        "uav_queue_max": _safe_max(getattr(env, "uav_queue", 0.0)),
        "sat_queue_max": _safe_max(getattr(env, "sat_queue", 0.0)),
        "gu_drop_sum": _safe_sum(getattr(env, "gu_drop", 0.0)),
        "uav_drop_sum": _safe_sum(getattr(env, "uav_drop", 0.0)),
        "sat_processed_sum": _safe_sum(sat_processed) if sat_processed is not None else 0.0,
        "sat_incoming_sum": _safe_sum(sat_incoming) if sat_incoming is not None else 0.0,
        "energy_mean": _safe_mean(getattr(env, "uav_energy", 0.0)),
        "reward_parts": dict(parts),
    }


def _worker(remote: Connection, cfg: SaginConfig, seed_offset: int) -> None:
    env_cfg = replace(cfg, seed=int(cfg.seed) + int(seed_offset))
    env = SaginParallelEnv(env_cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                seed = None if data is None else data.get("seed")
                obs, infos = env.reset(seed=seed)
                remote.send((obs, infos))
            elif cmd == "step":
                actions = data["actions"]
                auto_reset = bool(data.get("auto_reset", True))
                obs, rewards, terms, truncs, infos = env.step(actions)
                stats = _collect_step_stats(env)
                if auto_reset and _done_from_flags(terms, truncs):
                    obs, _ = env.reset()
                remote.send((obs, rewards, terms, truncs, infos, stats))
            elif cmd == "get_state":
                remote.send(env.get_global_state())
            elif cmd == "close":
                remote.close()
                return
            else:
                raise ValueError(f"Unknown worker command: {cmd}")
    finally:
        try:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass


class SyncVecSaginEnv:
    def __init__(self, cfg: SaginConfig, num_envs: int):
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be >= 1")
        self.envs = [SaginParallelEnv(replace(cfg, seed=int(cfg.seed) + i)) for i in range(self.num_envs)]
        self.agents = list(self.envs[0].agents)
        self.last_step_stats: List[Dict[str, object]] = [{} for _ in range(self.num_envs)]

    def reset(self, seeds: Sequence[int] | None = None):
        obs_batch = []
        infos_batch = []
        for idx, env in enumerate(self.envs):
            seed = None if seeds is None else int(seeds[idx])
            obs, infos = env.reset(seed=seed)
            obs_batch.append(obs)
            infos_batch.append(infos)
        return obs_batch, infos_batch

    def step(self, action_batch, auto_reset: bool = True):
        if len(action_batch) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} action dicts, got {len(action_batch)}")
        obs_batch = []
        rewards_batch = []
        terms_batch = []
        truncs_batch = []
        infos_batch = []
        stats_batch: List[Dict[str, object]] = []
        for env, actions in zip(self.envs, action_batch):
            obs, rewards, terms, truncs, infos = env.step(actions)
            stats = _collect_step_stats(env)
            if auto_reset and _done_from_flags(terms, truncs):
                obs, _ = env.reset()
            obs_batch.append(obs)
            rewards_batch.append(rewards)
            terms_batch.append(terms)
            truncs_batch.append(truncs)
            infos_batch.append(infos)
            stats_batch.append(stats)
        self.last_step_stats = stats_batch
        return obs_batch, rewards_batch, terms_batch, truncs_batch, infos_batch

    def get_global_state_batch(self) -> np.ndarray:
        states = [env.get_global_state() for env in self.envs]
        return np.stack(states, axis=0).astype(np.float32, copy=False)

    def close(self) -> None:
        for env in self.envs:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()


class SubprocVecSaginEnv:
    def __init__(self, cfg: SaginConfig, num_envs: int, start_method: str = "spawn"):
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be >= 1")
        self.agents = [f"uav_{i}" for i in range(cfg.num_uav)]
        self.last_step_stats: List[Dict[str, object]] = [{} for _ in range(self.num_envs)]

        ctx = mp.get_context(start_method)
        self._remotes: List[Connection] = []
        self._procs: List[mp.Process] = []
        for idx in range(self.num_envs):
            remote, work_remote = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(work_remote, cfg, idx), daemon=True)
            proc.start()
            work_remote.close()
            self._remotes.append(remote)
            self._procs.append(proc)

    def reset(self, seeds: Sequence[int] | None = None):
        for idx, remote in enumerate(self._remotes):
            seed = None if seeds is None else int(seeds[idx])
            remote.send(("reset", {"seed": seed}))
        results = [remote.recv() for remote in self._remotes]
        obs_batch, infos_batch = zip(*results)
        return list(obs_batch), list(infos_batch)

    def step(self, action_batch, auto_reset: bool = True):
        if len(action_batch) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} action dicts, got {len(action_batch)}")
        for remote, actions in zip(self._remotes, action_batch):
            remote.send(("step", {"actions": actions, "auto_reset": bool(auto_reset)}))
        results = [remote.recv() for remote in self._remotes]
        obs_batch, rewards_batch, terms_batch, truncs_batch, infos_batch, stats_batch = zip(*results)
        self.last_step_stats = list(stats_batch)
        return list(obs_batch), list(rewards_batch), list(terms_batch), list(truncs_batch), list(infos_batch)

    def get_global_state_batch(self) -> np.ndarray:
        for remote in self._remotes:
            remote.send(("get_state", None))
        states = [remote.recv() for remote in self._remotes]
        return np.stack(states, axis=0).astype(np.float32, copy=False)

    def close(self) -> None:
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
        self._remotes.clear()
        self._procs.clear()


def make_vec_env(cfg: SaginConfig, num_envs: int, backend: str = "subproc"):
    if int(num_envs) <= 1:
        raise ValueError("num_envs must be > 1 for vectorized environments.")
    backend_l = str(backend).lower()
    if backend_l == "sync":
        return SyncVecSaginEnv(cfg, num_envs)
    if backend_l == "subproc":
        return SubprocVecSaginEnv(cfg, num_envs)
    raise ValueError(f"Unknown vec backend: {backend}")
