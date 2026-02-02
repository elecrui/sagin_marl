from __future__ import annotations

import numpy as np

from sagin_marl.rl.baselines import zero_accel_policy


def test_zero_accel_policy_shape_dtype():
    actions = zero_accel_policy(3)
    assert actions.shape == (3, 2)
    assert actions.dtype == np.float32
    assert np.all(actions == 0.0)
