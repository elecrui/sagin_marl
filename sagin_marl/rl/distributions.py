from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Gamma, Normal


def atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squash_action(z: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.tanh(z) * scale


def squashed_logprob(dist: Normal, action: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    eps = 1e-4
    t = action / scale
    t = torch.clamp(t, -1 + eps, 1 - eps)
    z = atanh(t)
    logprob = dist.log_prob(z) - torch.log(1 - t.pow(2) + eps)
    if scale != 1.0:
        logprob = logprob - torch.log(torch.full_like(logprob, scale))
    return logprob.sum(-1)


class MaskedDirichlet:
    def __init__(self, alpha: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
        self.alpha = alpha.clamp_min(eps)
        self.mask = mask > 0.5
        self.eps = eps
        self.mask_f = self.mask.to(self.alpha.dtype)
        self.valid_count = self.mask_f.sum(dim=-1)

    def _masked_alpha(self) -> torch.Tensor:
        return torch.where(self.mask, self.alpha, torch.ones_like(self.alpha))

    def sample(self) -> torch.Tensor:
        gamma = Gamma(self.alpha, torch.ones_like(self.alpha)).sample()
        gamma = gamma * self.mask_f
        denom = gamma.sum(dim=-1, keepdim=True)
        action = torch.where(denom > self.eps, gamma / denom.clamp_min(self.eps), torch.zeros_like(gamma))
        single_mask = self.valid_count == 1
        if torch.any(single_mask):
            action = torch.where(single_mask.unsqueeze(-1), self.mask_f, action)
        no_mask = self.valid_count <= 0
        if torch.any(no_mask):
            action = torch.where(no_mask.unsqueeze(-1), torch.zeros_like(action), action)
        return action

    def mode(self) -> torch.Tensor:
        alpha_masked = self.alpha * self.mask_f
        denom = alpha_masked.sum(dim=-1, keepdim=True)
        action = torch.where(denom > self.eps, alpha_masked / denom.clamp_min(self.eps), torch.zeros_like(alpha_masked))
        single_mask = self.valid_count == 1
        if torch.any(single_mask):
            action = torch.where(single_mask.unsqueeze(-1), self.mask_f, action)
        no_mask = self.valid_count <= 0
        if torch.any(no_mask):
            action = torch.where(no_mask.unsqueeze(-1), torch.zeros_like(action), action)
        return action

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        masked_alpha = self._masked_alpha()
        action_safe = torch.where(self.mask, action.clamp_min(self.eps), torch.ones_like(action))
        alpha0 = (self.alpha * self.mask_f).sum(dim=-1).clamp_min(self.eps)
        logprob = (
            torch.lgamma(alpha0)
            - torch.lgamma(masked_alpha).sum(dim=-1)
            + ((masked_alpha - 1.0) * torch.log(action_safe)).sum(dim=-1)
        )
        return torch.where(self.valid_count >= 2, logprob, torch.zeros_like(logprob))

    def entropy(self) -> torch.Tensor:
        masked_alpha = self._masked_alpha()
        alpha0 = (self.alpha * self.mask_f).sum(dim=-1).clamp_min(self.eps)
        k_valid = self.valid_count
        log_beta = torch.lgamma(masked_alpha).sum(dim=-1) - torch.lgamma(alpha0)
        entropy = (
            log_beta
            + (alpha0 - k_valid) * torch.digamma(alpha0)
            - ((masked_alpha - 1.0) * torch.digamma(masked_alpha)).sum(dim=-1)
        )
        return torch.where(self.valid_count >= 2, entropy, torch.zeros_like(entropy))


@dataclass
class MaskedSequentialCategoricalSample:
    indices: torch.Tensor
    select_mask: torch.Tensor


class MaskedSequentialCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor, k: int = 2):
        self.logits = logits
        self.mask = mask > 0.5
        self.k = max(int(k), 0)
        self.num_choices = int(logits.shape[-1])
        self._bit_weights = (1 << torch.arange(self.num_choices, device=logits.device, dtype=torch.int64))

    def _safe_logits(self, mask: torch.Tensor) -> torch.Tensor:
        return self.logits.masked_fill(~mask, -1e9)

    def _indices_to_select_mask(self, indices: torch.Tensor) -> torch.Tensor:
        select_mask = torch.zeros((*indices.shape[:-1], self.num_choices), dtype=self.logits.dtype, device=self.logits.device)
        valid_choice_mask = indices >= 0
        if torch.any(valid_choice_mask):
            one_hot = F.one_hot(indices.clamp_min(0), num_classes=self.num_choices).to(select_mask.dtype)
            one_hot = one_hot * valid_choice_mask.unsqueeze(-1).to(select_mask.dtype)
            select_mask = one_hot.sum(dim=-2)
        return select_mask

    def _pad_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] >= self.k:
            return indices
        pad = torch.full(
            (*indices.shape[:-1], self.k - indices.shape[-1]),
            -1,
            dtype=indices.dtype,
            device=indices.device,
        )
        return torch.cat([indices, pad], dim=-1)

    def _empty_result(self) -> MaskedSequentialCategoricalSample:
        batch_shape = self.logits.shape[:-1]
        return MaskedSequentialCategoricalSample(
            indices=torch.full((*batch_shape, self.k), -1, dtype=torch.int64, device=self.logits.device),
            select_mask=torch.zeros((*batch_shape, self.num_choices), dtype=self.logits.dtype, device=self.logits.device),
        )

    def sample(self) -> MaskedSequentialCategoricalSample:
        if self.k <= 0:
            return self._empty_result()
        max_k = min(self.k, self.num_choices)
        if max_k <= 0:
            return self._empty_result()
        valid_count = self.mask.sum(dim=-1)
        safe_logits = self.logits.masked_fill(~self.mask, float("-inf"))
        gumbel_u = torch.rand_like(self.logits).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(gumbel_u))
        sampled_scores = safe_logits + gumbel
        topk = sampled_scores.topk(k=max_k, dim=-1).indices
        rank_idx = torch.arange(max_k, device=topk.device).view(*([1] * (topk.ndim - 1)), -1)
        valid_rank = rank_idx < valid_count.unsqueeze(-1)
        indices = torch.where(valid_rank, topk, torch.full_like(topk, -1))
        indices = self._pad_indices(indices)
        select_mask = self._indices_to_select_mask(indices)
        return MaskedSequentialCategoricalSample(indices=indices, select_mask=select_mask)

    def mode(self) -> MaskedSequentialCategoricalSample:
        if self.k <= 0:
            return self._empty_result()
        safe_logits = self._safe_logits(self.mask)
        topk = safe_logits.topk(k=min(self.k, self.num_choices), dim=-1).indices
        valid_count = self.mask.sum(dim=-1)
        rank_idx = torch.arange(topk.shape[-1], device=topk.device).view(*([1] * (topk.ndim - 1)), -1)
        valid_rank = rank_idx < valid_count.unsqueeze(-1)
        indices = torch.where(valid_rank, topk, torch.full_like(topk, -1))
        indices = self._pad_indices(indices)
        select_mask = self._indices_to_select_mask(indices)
        return MaskedSequentialCategoricalSample(indices=indices, select_mask=select_mask)

    def log_prob(self, indices: torch.Tensor) -> torch.Tensor:
        if self.k <= 0:
            return torch.zeros(self.logits.shape[:-1], dtype=self.logits.dtype, device=self.logits.device)
        flat_logits = self.logits.reshape(-1, self.num_choices)
        flat_mask = self.mask.reshape(-1, self.num_choices)
        flat_indices = indices.reshape(-1, self.k)
        logprob = torch.zeros((flat_logits.shape[0],), dtype=flat_logits.dtype, device=flat_logits.device)
        current_mask = flat_mask.clone()
        for step in range(self.k):
            remaining = current_mask.sum(dim=-1)
            active = remaining > 1
            if torch.any(active):
                safe_logits = flat_logits.masked_fill(~current_mask, -1e9)
                safe_logits = torch.where(active.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits))
                step_indices = flat_indices[:, step].clamp_min(0)
                step_log_probs = F.log_softmax(safe_logits, dim=-1)
                gathered = step_log_probs.gather(1, step_indices.unsqueeze(-1)).squeeze(-1)
                step_valid = active & (flat_indices[:, step] >= 0) & current_mask.gather(
                    1, step_indices.unsqueeze(-1)
                ).squeeze(-1)
                logprob = logprob + torch.where(step_valid, gathered, torch.zeros_like(logprob))
            step_valid_idx = flat_indices[:, step] >= 0
            if torch.any(step_valid_idx):
                chosen_mask = torch.zeros_like(current_mask)
                chosen_mask[step_valid_idx, flat_indices[step_valid_idx, step].long()] = True
                current_mask = current_mask & ~chosen_mask
        return logprob.reshape(self.logits.shape[:-1])

    def _mask_key(self, mask: torch.Tensor) -> int:
        return int(torch.sum(mask.to(torch.int64) * self._bit_weights).item())

    def _entropy_exact_row(
        self,
        row_logits: torch.Tensor,
        row_mask: torch.Tensor,
        steps_left: int,
        memo: dict[tuple[int, int], torch.Tensor],
    ) -> torch.Tensor:
        valid_count = int(row_mask.sum().item())
        if steps_left <= 0 or valid_count <= 1:
            return row_logits.new_zeros(())
        key = (steps_left, self._mask_key(row_mask))
        if key in memo:
            return memo[key]
        safe_logits = row_logits.masked_fill(~row_mask, float("-inf"))
        log_probs = F.log_softmax(safe_logits, dim=-1)
        probs = log_probs.exp()
        row_entropy = -(probs[row_mask] * log_probs[row_mask]).sum()
        if steps_left == 1:
            memo[key] = row_entropy
            return row_entropy
        future_entropy = row_logits.new_zeros(())
        valid_indices = torch.nonzero(row_mask, as_tuple=False).flatten()
        for idx in valid_indices:
            next_mask = row_mask.clone()
            next_mask[idx] = False
            future_entropy = future_entropy + probs[idx] * self._entropy_exact_row(
                row_logits,
                next_mask,
                steps_left - 1,
                memo,
            )
        total_entropy = row_entropy + future_entropy
        memo[key] = total_entropy
        return total_entropy

    def entropy(self) -> torch.Tensor:
        batch_shape = self.logits.shape[:-1]
        if self.k <= 0:
            return torch.zeros(batch_shape, dtype=self.logits.dtype, device=self.logits.device)
        safe_logits = self._safe_logits(self.mask)
        has_any = self.mask.any(dim=-1)
        safe_logits = torch.where(has_any.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits))
        dist1 = Categorical(logits=safe_logits)
        entropy = torch.where(has_any, dist1.entropy(), torch.zeros_like(dist1.entropy()))
        if self.k == 1:
            return entropy
        if self.k != 2:
            flat_logits = self.logits.reshape(-1, self.num_choices)
            flat_mask = self.mask.reshape(-1, self.num_choices)
            flat_entropy = []
            for row_logits, row_mask in zip(flat_logits, flat_mask):
                memo: dict[tuple[int, int], torch.Tensor] = {}
                flat_entropy.append(
                    self._entropy_exact_row(
                        row_logits,
                        row_mask,
                        min(self.k, int(row_mask.sum().item())),
                        memo,
                    )
                )
            return torch.stack(flat_entropy, dim=0).reshape(batch_shape)
        probs1 = dist1.probs
        expected_h2 = torch.zeros_like(entropy)
        for choice in range(self.num_choices):
            choice_mask = self.mask.clone()
            choice_mask[..., choice] = False
            remaining = choice_mask.sum(dim=-1)
            active = self.mask[..., choice] & (remaining > 0)
            if not torch.any(active):
                continue
            logits2 = self._safe_logits(choice_mask)
            logits2 = torch.where(active.unsqueeze(-1), logits2, torch.zeros_like(logits2))
            dist2 = Categorical(logits=logits2)
            h2 = torch.where(active, dist2.entropy(), torch.zeros_like(entropy))
            expected_h2 = expected_h2 + probs1[..., choice] * h2
        return entropy + expected_h2


@dataclass
class HybridActionSample:
    env_action: torch.Tensor
    accel: torch.Tensor | None
    bw_action: torch.Tensor | None
    sat_indices: torch.Tensor | None
    sat_select_mask: torch.Tensor | None
    logprob_parts: Dict[str, torch.Tensor]
    entropy_parts: Dict[str, torch.Tensor]


def _normalize_requested_heads(heads: Iterable[str] | None) -> set[str] | None:
    if heads is None:
        return None
    allowed = {"accel", "bw", "sat"}
    out = {str(head).strip().lower() for head in heads}
    return out & allowed


class HybridActionDist:
    def __init__(
        self,
        accel_mu: torch.Tensor | None,
        accel_log_std: torch.Tensor | None,
        bw_alpha: torch.Tensor | None = None,
        bw_mask: torch.Tensor | None = None,
        sat_logits: torch.Tensor | None = None,
        sat_mask: torch.Tensor | None = None,
        sat_num_select: int = 2,
    ):
        batch_source = accel_mu if accel_mu is not None else bw_alpha
        if batch_source is None:
            batch_source = sat_logits
        if batch_source is None:
            raise ValueError("HybridActionDist requires at least one action head.")
        self.batch_shape = tuple(batch_source.shape[:-1])
        self.dtype = batch_source.dtype
        self.device = batch_source.device
        self.accel_mu = accel_mu
        self.accel_log_std = None if accel_log_std is None else torch.clamp(accel_log_std, -5.0, 2.0)
        self.accel_std = None if self.accel_log_std is None else torch.exp(self.accel_log_std)
        self.accel_dist = (
            None if self.accel_mu is None or self.accel_std is None else Normal(self.accel_mu, self.accel_std)
        )
        self.bw_dist = None if bw_alpha is None or bw_mask is None else MaskedDirichlet(bw_alpha, bw_mask)
        self.sat_dist = (
            None
            if sat_logits is None or sat_mask is None
            else MaskedSequentialCategorical(sat_logits, sat_mask, k=sat_num_select)
        )

    def _zero_batch(self) -> torch.Tensor:
        return torch.zeros(self.batch_shape, dtype=self.dtype, device=self.device)

    def _head_enabled(self, head_set: set[str] | None, head: str) -> bool:
        return head_set is None or head in head_set

    def sample(
        self,
        deterministic: bool = False,
        compute_logprob: bool = False,
        compute_entropy: bool = False,
        stat_heads: Iterable[str] | None = None,
    ) -> HybridActionSample:
        stat_head_set = _normalize_requested_heads(stat_heads)
        logprob_parts: Dict[str, torch.Tensor] = {}
        entropy_parts: Dict[str, torch.Tensor] = {}
        env_parts: list[torch.Tensor] = []
        accel = None
        if self.accel_dist is not None and self.accel_mu is not None:
            accel_z = self.accel_mu if deterministic else self.accel_dist.rsample()
            accel = squash_action(accel_z, scale=1.0)
            env_parts.append(accel)
            if compute_logprob and self._head_enabled(stat_head_set, "accel"):
                logprob_parts["accel"] = squashed_logprob(self.accel_dist, accel, scale=1.0)
            if compute_entropy and self._head_enabled(stat_head_set, "accel"):
                entropy_parts["accel"] = self.accel_dist.entropy().sum(dim=-1)
        bw_action = None
        if self.bw_dist is not None:
            bw_action = self.bw_dist.mode() if deterministic else self.bw_dist.sample()
            env_parts.append(bw_action)
            if compute_logprob and self._head_enabled(stat_head_set, "bw"):
                logprob_parts["bw"] = self.bw_dist.log_prob(bw_action)
            if compute_entropy and self._head_enabled(stat_head_set, "bw"):
                entropy_parts["bw"] = self.bw_dist.entropy()
        sat_indices = None
        sat_select_mask = None
        if self.sat_dist is not None:
            sat_sample = self.sat_dist.mode() if deterministic else self.sat_dist.sample()
            sat_indices = sat_sample.indices
            sat_select_mask = sat_sample.select_mask
            env_parts.append(sat_select_mask)
            if compute_logprob and self._head_enabled(stat_head_set, "sat"):
                logprob_parts["sat"] = self.sat_dist.log_prob(sat_indices)
            if compute_entropy and self._head_enabled(stat_head_set, "sat"):
                entropy_parts["sat"] = self.sat_dist.entropy()
        env_action = torch.cat(env_parts, dim=-1) if env_parts else torch.zeros(
            (*self.batch_shape, 0), dtype=self.dtype, device=self.device
        )
        return HybridActionSample(
            env_action=env_action,
            accel=accel,
            bw_action=bw_action,
            sat_indices=sat_indices,
            sat_select_mask=sat_select_mask,
            logprob_parts=logprob_parts,
            entropy_parts=entropy_parts,
        )

    def log_prob(
        self,
        accel: torch.Tensor,
        bw_action: torch.Tensor | None = None,
        sat_indices: torch.Tensor | None = None,
        heads: Iterable[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        head_set = _normalize_requested_heads(heads)
        out: Dict[str, torch.Tensor] = {}
        if self.accel_dist is not None and self._head_enabled(head_set, "accel"):
            out["accel"] = squashed_logprob(self.accel_dist, accel, scale=1.0)
        if self.bw_dist is not None and bw_action is not None and self._head_enabled(head_set, "bw"):
            out["bw"] = self.bw_dist.log_prob(bw_action)
        if self.sat_dist is not None and sat_indices is not None and self._head_enabled(head_set, "sat"):
            out["sat"] = self.sat_dist.log_prob(sat_indices)
        return out

    def entropy(self, heads: Iterable[str] | None = None) -> Dict[str, torch.Tensor]:
        head_set = _normalize_requested_heads(heads)
        out: Dict[str, torch.Tensor] = {}
        if self.accel_dist is not None and self._head_enabled(head_set, "accel"):
            out["accel"] = self.accel_dist.entropy().sum(dim=-1)
        if self.bw_dist is not None and self._head_enabled(head_set, "bw"):
            out["bw"] = self.bw_dist.entropy()
        if self.sat_dist is not None and self._head_enabled(head_set, "sat"):
            out["sat"] = self.sat_dist.entropy()
        return out
