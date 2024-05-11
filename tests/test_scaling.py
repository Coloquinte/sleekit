import numpy as np

from sleekit.codebook import Codebook
from sleekit.obq import random_psd_matrix
from sleekit.scaling import *


def test_scaling_axis():
    data = np.array(
        [
            [0.0, 10.0],
            [5.0, 5.0],
        ],
        dtype=np.float32,
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


def test_non_saturating_scaling():
    data = np.array(
        [
            [0.0, 10.0, -20.0, 15.0],
            [5.0, 5.0, 10.0, -10.0],
            [1.0, 2.0, -4.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 10.0, 100.0, 1000.0],
            [-1.0, 10.0, 100.0, 1000.0],
        ],
        dtype=np.float32,
    )
    cb = Codebook([-1.0, 0.0, 10.0, 20.0])
    sc0 = compute_non_saturating_scaling(data, cb, 0)
    sc1 = compute_non_saturating_scaling(data, cb, 1)
    assert np.allclose(sc0, [1.0, 0.5, 0.2, 0.5e-18, 50.0, 50.0])
    assert np.allclose(sc1, [0.25, 0.5, 5.0, 50.0])


def test_min_mse_scaling():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = Codebook.uniform(9, -2, 2)
    compute_min_mse_scaling(data, cb, 0)


def test_min_mse_scaling_diag_hessian():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = Codebook.uniform(9, -2, 2)
    H = np.random.rand(50)
    compute_min_mse_scaling(data, cb, 0, H=H)


def test_min_mse_scaling_hessian():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = Codebook.uniform(9, -2, 2)
    H = random_psd_matrix(50, 10)
    compute_min_mse_scaling(data, cb, 0, H=H)
