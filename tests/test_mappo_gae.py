import numpy as np

from sagin_marl.rl.mappo import compute_gae


def test_compute_gae_true_terminal_uses_zero_bootstrap():
    rewards = np.asarray([1.0], dtype=np.float32)
    values = np.asarray([0.2], dtype=np.float32)
    bootstrap_values = np.asarray([0.0], dtype=np.float32)
    episode_boundaries = np.asarray([True])

    adv, rets = compute_gae(rewards, values, bootstrap_values, episode_boundaries, gamma=0.9, lam=0.95)

    assert np.allclose(adv, np.asarray([0.8], dtype=np.float32))
    assert np.allclose(rets, np.asarray([1.0], dtype=np.float32))


def test_compute_gae_time_limit_truncation_bootstraps_last_value():
    rewards = np.asarray([1.0], dtype=np.float32)
    values = np.asarray([0.2], dtype=np.float32)
    bootstrap_values = np.asarray([0.5], dtype=np.float32)
    episode_boundaries = np.asarray([True])

    adv, rets = compute_gae(rewards, values, bootstrap_values, episode_boundaries, gamma=0.9, lam=0.95)

    assert np.allclose(adv, np.asarray([1.25], dtype=np.float32))
    assert np.allclose(rets, np.asarray([1.45], dtype=np.float32))


def test_compute_gae_rollout_cutoff_bootstraps_final_step_without_breaking_recursion():
    rewards = np.asarray([1.0, 2.0], dtype=np.float32)
    values = np.asarray([0.1, 0.2], dtype=np.float32)
    bootstrap_values = np.asarray([0.2, 0.4], dtype=np.float32)
    episode_boundaries = np.asarray([False, False])

    adv, rets = compute_gae(rewards, values, bootstrap_values, episode_boundaries, gamma=0.9, lam=0.95)

    expected_adv = np.asarray([2.9268, 2.16], dtype=np.float32)
    expected_rets = expected_adv + values
    assert np.allclose(adv, expected_adv, atol=1e-5)
    assert np.allclose(rets, expected_rets, atol=1e-5)