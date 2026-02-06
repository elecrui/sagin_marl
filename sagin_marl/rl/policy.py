from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class PolicyOutput:
    action: torch.Tensor
    logprob: torch.Tensor
    accel: torch.Tensor
    bw_logits: torch.Tensor | None = None
    sat_logits: torch.Tensor | None = None
    dist_out: Dict[str, torch.Tensor] | None = None


def flatten_obs(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    parts = [
        obs["own"].ravel(),
        obs["users"].ravel(),
        obs["users_mask"].ravel(),
        obs["sats"].ravel(),
        obs["sats_mask"].ravel(),
        obs["nbrs"].ravel(),
        obs["nbrs_mask"].ravel(),
    ]
    return np.concatenate(parts).astype(np.float32)


def batch_flatten_obs(obs_batch: Dict[str, np.ndarray], cfg) -> np.ndarray:
    # obs_batch is a list/dict of per-agent observations
    obs_list = [flatten_obs(obs, cfg) for obs in obs_batch]
    return np.stack(obs_list, axis=0)


def atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_action(z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.tanh(z) * scale


def _logprob_from_squashed(dist: Normal, action: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    eps = 1e-4
    t = action / scale
    t = torch.clamp(t, -1 + eps, 1 - eps)
    z = atanh(t)
    logprob = dist.log_prob(z) - torch.log(1 - t.pow(2) + eps)
    if scale != 1.0:
        logprob = logprob - math.log(scale)
    return logprob.sum(-1)


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.enable_bw = cfg.enable_bw_action
        self.enable_sat = not cfg.fixed_satellite_strategy
        self.bw_scale = float(cfg.bw_logit_scale)
        self.sat_scale = float(cfg.sat_logit_scale)

        self.obs_norm = nn.LayerNorm(obs_dim) if getattr(cfg, "input_norm_enabled", False) else nn.Identity()
        self.fc1 = nn.Linear(obs_dim, cfg.actor_hidden)
        self.fc2 = nn.Linear(cfg.actor_hidden, cfg.actor_hidden)

        self.mu_head = nn.Linear(cfg.actor_hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        if self.enable_bw:
            self.bw_head = nn.Linear(cfg.actor_hidden, cfg.users_obs_max)
            self.bw_log_std = nn.Parameter(torch.zeros(cfg.users_obs_max))
        else:
            self.bw_head = None
            self.bw_log_std = None

        if self.enable_sat:
            self.sat_head = nn.Linear(cfg.actor_hidden, cfg.sats_obs_max)
            self.sat_log_std = nn.Parameter(torch.zeros(cfg.sats_obs_max))
        else:
            self.sat_head = None
            self.sat_log_std = None

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.obs_norm(obs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        out = {"mu": mu}
        if self.bw_head is not None:
            out["bw_mu"] = self.bw_head(x)
        if self.sat_head is not None:
            out["sat_mu"] = self.sat_head(x)
        return out

    def _concat_actions(
        self,
        accel: torch.Tensor,
        bw: torch.Tensor | None,
        sat: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [accel]
        if self.enable_bw and bw is not None:
            parts.append(bw)
        if self.enable_sat and sat is not None:
            parts.append(sat)
        return torch.cat(parts, dim=-1)

    def _split_actions(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        idx = 0
        accel = action[:, idx : idx + 2]
        idx += 2
        bw = None
        sat = None
        if self.enable_bw:
            bw = action[:, idx : idx + self.cfg.users_obs_max]
            idx += self.cfg.users_obs_max
        if self.enable_sat:
            sat = action[:, idx : idx + self.cfg.sats_obs_max]
        return accel, bw, sat

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        out = self.forward(obs)
        mu = out["mu"]
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            z = mu
        else:
            z = dist.rsample()
        accel = _squash_action(z, scale=1.0)
        logprob = _logprob_from_squashed(dist, accel, scale=1.0)

        bw_logits = None
        sat_logits = None

        if self.enable_bw:
            bw_mu = out["bw_mu"]
            bw_log_std = torch.clamp(self.bw_log_std, -5.0, 2.0)
            bw_std = torch.exp(bw_log_std)
            bw_dist = Normal(bw_mu, bw_std)
            z_bw = bw_mu if deterministic else bw_dist.rsample()
            bw_logits = _squash_action(z_bw, scale=self.bw_scale)
            logprob = logprob + _logprob_from_squashed(bw_dist, bw_logits, scale=self.bw_scale)

        if self.enable_sat:
            sat_mu = out["sat_mu"]
            sat_log_std = torch.clamp(self.sat_log_std, -5.0, 2.0)
            sat_std = torch.exp(sat_log_std)
            sat_dist = Normal(sat_mu, sat_std)
            z_sat = sat_mu if deterministic else sat_dist.rsample()
            sat_logits = _squash_action(z_sat, scale=self.sat_scale)
            logprob = logprob + _logprob_from_squashed(sat_dist, sat_logits, scale=self.sat_scale)

        action = self._concat_actions(accel, bw_logits, sat_logits)
        return PolicyOutput(
            action=action,
            logprob=logprob,
            accel=accel,
            bw_logits=bw_logits,
            sat_logits=sat_logits,
            dist_out=out,
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        out: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.isfinite(obs).all():
            print("NaN/Inf detected in obs passed to evaluate_actions")
            raise ValueError("obs contains NaN/Inf")
        if out is None:
            out = self.forward(obs)
        accel_action, bw_action, sat_action = self._split_actions(action)

        mu = out["mu"]
        if not torch.isfinite(mu).all():
            print("NaN/Inf detected in actor mu inside evaluate_actions")
            raise ValueError("actor mu contains NaN/Inf")
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        if not torch.isfinite(std).all():
            print("NaN/Inf detected in actor std inside evaluate_actions")
            raise ValueError("actor std contains NaN/Inf")
        dist = Normal(mu, std)
        logprob = _logprob_from_squashed(dist, accel_action, scale=1.0)
        entropy = dist.entropy().sum(-1)

        if self.enable_bw and bw_action is not None:
            bw_mu = out["bw_mu"]
            bw_log_std = torch.clamp(self.bw_log_std, -5.0, 2.0)
            bw_std = torch.exp(bw_log_std)
            bw_dist = Normal(bw_mu, bw_std)
            logprob = logprob + _logprob_from_squashed(bw_dist, bw_action, scale=self.bw_scale)
            entropy = entropy + bw_dist.entropy().sum(-1)

        if self.enable_sat and sat_action is not None:
            sat_mu = out["sat_mu"]
            sat_log_std = torch.clamp(self.sat_log_std, -5.0, 2.0)
            sat_std = torch.exp(sat_log_std)
            sat_dist = Normal(sat_mu, sat_std)
            logprob = logprob + _logprob_from_squashed(sat_dist, sat_action, scale=self.sat_scale)
            entropy = entropy + sat_dist.entropy().sum(-1)

        return logprob, entropy
