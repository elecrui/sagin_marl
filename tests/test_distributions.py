from __future__ import annotations

import itertools
import math

import torch

from sagin_marl.rl.distributions import MaskedSequentialCategorical


def _manual_seq_logprob(logits: list[float], mask: list[bool], indices: list[int]) -> float:
    remaining = [idx for idx, valid in enumerate(mask) if valid]
    logprob = 0.0
    steps = min(len(indices), len(remaining))
    for step in range(steps):
        if len(remaining) <= 1:
            break
        chosen = indices[step]
        if chosen < 0 or chosen not in remaining:
            break
        max_logit = max(logits[idx] for idx in remaining)
        weights = [math.exp(logits[idx] - max_logit) for idx in remaining]
        denom = sum(weights)
        choice_pos = remaining.index(chosen)
        logprob += math.log(weights[choice_pos] / denom)
        remaining.remove(chosen)
    return logprob


def _manual_seq_entropy(logits: list[float], mask: list[bool], k: int) -> float:
    valid = [idx for idx, ok in enumerate(mask) if ok]
    draw_k = min(k, len(valid))
    if draw_k <= 1:
        return 0.0
    total = 0.0
    for ordered in itertools.permutations(valid, draw_k):
        prob = 1.0
        remaining = valid.copy()
        for step, choice in enumerate(ordered):
            if len(remaining) <= 1:
                break
            max_logit = max(logits[idx] for idx in remaining)
            weights = [math.exp(logits[idx] - max_logit) for idx in remaining]
            denom = sum(weights)
            choice_pos = remaining.index(choice)
            prob *= weights[choice_pos] / denom
            remaining.remove(choice)
        if prob > 0.0:
            total -= prob * math.log(prob)
    return total


def test_masked_sequential_categorical_sample_returns_unique_valid_indices():
    torch.manual_seed(0)
    logits = torch.tensor([[1.2, -0.4, 0.3, 2.0], [0.1, 0.2, -1.0, 0.7]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.float32)
    dist = MaskedSequentialCategorical(logits, mask, k=2)

    sample = dist.sample()

    assert sample.indices.shape == (2, 2)
    assert sample.select_mask.shape == (2, 4)
    for row in range(sample.indices.shape[0]):
        chosen = sample.indices[row][sample.indices[row] >= 0].tolist()
        assert len(chosen) == len(set(chosen))
        for idx in chosen:
            assert mask[row, idx].item() > 0.5
        assert int(sample.select_mask[row].sum().item()) == len(chosen)


def test_masked_sequential_categorical_log_prob_matches_manual_k2():
    logits = torch.tensor([[0.7, -0.2, 0.1], [0.2, 1.1, -0.3]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    indices = torch.tensor([[2, 0], [1, 0]], dtype=torch.int64)
    dist = MaskedSequentialCategorical(logits, mask, k=2)

    log_prob = dist.log_prob(indices)
    expected = torch.tensor(
        [
            _manual_seq_logprob(logits[0].tolist(), [True, True, True], [2, 0]),
            _manual_seq_logprob(logits[1].tolist(), [True, True, False], [1, 0]),
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(log_prob, expected, rtol=1e-6, atol=1e-6)


def test_masked_sequential_categorical_entropy_matches_manual_k3():
    logits = torch.tensor([[0.4, -0.3, 1.0], [0.2, 0.5, -0.1]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    dist = MaskedSequentialCategorical(logits, mask, k=3)

    entropy = dist.entropy()
    expected = torch.tensor(
        [
            _manual_seq_entropy(logits[0].tolist(), [True, True, True], 3),
            _manual_seq_entropy(logits[1].tolist(), [True, True, True], 3),
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(entropy, expected, rtol=1e-6, atol=1e-6)
