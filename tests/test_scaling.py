import numpy as np

from sleekit.codebook import Codebook
from sleekit.obq import random_psd_matrix
from sleekit.scaling import (
    compute_norm_scaling,
    apply_scaling,
    compute_non_saturating_scaling,
    compute_min_mse_scaling,
)


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


def test_scaling_axis_high_dim():
    data = np.random.randn(10, 20, 30, 40).astype(np.float32)
    sc = compute_norm_scaling(data, 0)
    assert len(sc) == 10
    sc = compute_norm_scaling(data, 1)
    assert len(sc) == 20
    sc = compute_norm_scaling(data, 2)
    assert len(sc) == 30
    sc = compute_norm_scaling(data, 3)
    assert len(sc) == 40


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


def test_non_saturating_scaling_high_dim():
    data = np.random.randn(10, 20, 30, 40).astype(np.float32)
    cb = Codebook.uniform(9, -2, 2)
    sc = compute_non_saturating_scaling(data, cb, 0)
    assert len(sc) == 10
    sc = compute_non_saturating_scaling(data, cb, 1)
    assert len(sc) == 20
    sc = compute_non_saturating_scaling(data, cb, 2)
    assert len(sc) == 30
    sc = compute_non_saturating_scaling(data, cb, 3)
    assert len(sc) == 40


def test_min_mse_scaling():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = Codebook.uniform(9, -2, 2)
    sc = compute_min_mse_scaling(data, cb, 0)
    assert len(sc) == 20
    sc = compute_min_mse_scaling(data, cb, 1)
    assert len(sc) == 50


def test_min_mse_scaling_diag_hessian():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = Codebook.uniform(9, -2, 2)
    H = np.random.rand(50)
    sc = compute_min_mse_scaling(data, cb, 0, H=H)
    assert len(sc) == 20
    H = np.random.rand(20)
    sc = compute_min_mse_scaling(data, cb, 1, H=H)
    assert len(sc) == 50


def test_min_mse_scaling_hessian():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = Codebook.uniform(9, -2, 2)
    H = random_psd_matrix(50, 10)
    sc = compute_min_mse_scaling(data, cb, 0, H=H)
    assert len(sc) == 20
    H = random_psd_matrix(20, 10)
    sc = compute_min_mse_scaling(data, cb, 1, H=H)
    assert len(sc) == 50


def test_min_mse_scaling_obq():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = Codebook.uniform(9, -2, 2)
    H = random_psd_matrix(50, 10, damp=1.0e-6)
    sc = compute_min_mse_scaling(data, cb, 0, H=H, obq=True)
    assert len(sc) == 20
    H = random_psd_matrix(20, 10, damp=1.0e-6)
    sc = compute_min_mse_scaling(data, cb, 1, H=H, obq=True)
    assert len(sc) == 50
