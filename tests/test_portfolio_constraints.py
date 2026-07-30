import numpy as np

from portfolio_optimizer import cap_and_normalize_weights
from quant.models.calibration import map_scores_to_expected_returns


def test_position_caps_are_respected():
    weights = cap_and_normalize_weights(np.array([0.9, 0.05, 0.05]), max_single_weight=0.6)
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.6 + 1e-9


def test_oos_calibration_is_monotonic_for_monotonic_data():
    mapped = map_scores_to_expected_returns(
        [0.15, 0.85],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [-0.02, -0.01, 0.0, 0.01, 0.02],
        buckets=5,
    )
    assert mapped[1] > mapped[0]
