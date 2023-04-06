
import numpy as np
import pytest
from numpy.testing import assert_allclose

from little_helpers.rl_helpers import get_future_rewards

# Test cases
test_data = [
    (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0, np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
    (np.array([3, 0, 10]), 1, np.array([13, 10, 10])),
    (
        np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        0.99,
        np.array([0.96059601, 0.970299, 0.9801, 0.99, 1.0]),
    ),
    (
        np.array([-1.0, -1.0, -1.0, -1.0, 0.0]),
        0.99,
        np.array([-3.940399, -2.9701, -1.99, -1.0, 0.0]),
    ),
]


@pytest.mark.parametrize("rewards, gamma, expected", test_data)
def test_get_future_rewards(rewards: np.ndarray, gamma: float, expected: np.ndarray):
    result = get_future_rewards(rewards, gamma)
    assert_allclose(result, expected, atol=1e-4)


def test_get_future_rewards_type_error():
    with pytest.raises(TypeError):
        get_future_rewards([1, 2, 3], gamma="1")
