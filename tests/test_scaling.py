import numpy as np

from sleekit.scaling import *


def test_scaling_axis():
    data = np.array(
        [
            [0.0, 10.0],
            [5.0, 5.0],
        ]
    )
    # Axis 0
    sc = compute_norm_scaling(data, 0)
    scaled = apply_scaling(data, sc, 0)
    exp_sc = np.array([10.0 / np.sqrt(2), 5.0])
    expected = np.array([[0.0, np.sqrt(2)], [1.0, 1.0]])
    assert np.allclose(sc, exp_sc)
    assert np.allclose(scaled, expected)
    assert np.allclose(apply_scaling(scaled, 1 / sc, 0), data)
    # Axis 1
    sc = compute_norm_scaling(data, 1)
    scaled = apply_scaling(data, sc, 1)
    exp_sc = np.array([5.0 / np.sqrt(2), np.sqrt(125 / 2)])
    expected = np.array(
        [[0.0, 10.0 / np.sqrt(125 / 2)], [np.sqrt(2), 5.0 / np.sqrt(125 / 2)]]
    )
    assert np.allclose(sc, exp_sc)
    assert np.allclose(scaled, expected)
    assert np.allclose(apply_scaling(scaled, 1 / sc, 1), data)
