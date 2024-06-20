import numpy as np

from sleekit.codebook import Codebook, UniformCodebook
from sleekit.obq import quantization_error, random_psd_matrix
from sleekit.scaling import (
    compute_norm_scaling,
    apply_scaling,
    compute_non_saturating_scaling,
    compute_min_mse_scaling,
    compute_obq_scaling,
    compute_scaling,
    quantize_with_scaling,
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
    assert np.allclose(sc0, [20, 10, 4, 1e-16, 50, 50])
    assert np.allclose(sc1, [1, 0.5, 20, 50])


def test_non_saturating_scaling_high_dim():
    data = np.random.randn(10, 20, 30, 40).astype(np.float32)
    cb = UniformCodebook(9, -2, 2)
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
    cb = UniformCodebook(9, -2, 2)
    sc = compute_min_mse_scaling(data, cb, 0)
    assert len(sc) == 20
    sc = compute_min_mse_scaling(data, cb, 1)
    assert len(sc) == 50


def test_min_mse_scaling_diag_hessian():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = UniformCodebook(9, -2, 2)
    H = np.random.rand(50)
    sc = compute_min_mse_scaling(data, cb, 0, H=H)
    assert len(sc) == 20
    H = np.random.rand(20)
    sc = compute_min_mse_scaling(data, cb, 1, H=H)
    assert len(sc) == 50


def test_min_mse_scaling_hessian():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = UniformCodebook(9, -2, 2)
    H = random_psd_matrix(50, 10)
    sc = compute_min_mse_scaling(data, cb, 0, H=H)
    assert len(sc) == 20
    H = random_psd_matrix(20, 10)
    sc = compute_min_mse_scaling(data, cb, 1, H=H)
    assert len(sc) == 50


def test_min_mse_scaling_obq():
    data = np.random.randn(20, 50).astype(np.float32)
    cb = UniformCodebook(9, -2, 2)
    H = random_psd_matrix(50, 10, damp=1.0e-6)
    sc = compute_obq_scaling(data, cb, 0, H=H)
    assert len(sc) == 20
    H = random_psd_matrix(20, 10, damp=1.0e-6)
    sc = compute_obq_scaling(data, cb, 1, H=H)
    assert len(sc) == 50


def test_min_mse_scaling_quality():
    size = 100
    data = np.random.randn(20, size).astype(np.float32)
    cb = UniformCodebook(9, -3, 3)
    H = random_psd_matrix(size, 10, damp=1.0e-6)
    sc_base = compute_min_mse_scaling(data, cb, 0)
    sc_diag = compute_min_mse_scaling(data, cb, 0, H=H.diagonal())
    sc_hessian = compute_min_mse_scaling(data, cb, 0, H=H)
    sc_obq = compute_obq_scaling(data, cb, 0, H=H)
    q_base = quantize_with_scaling(data, sc_base, cb)
    q_diag = quantize_with_scaling(data, sc_diag, cb)
    q_hessian = quantize_with_scaling(data, sc_hessian, cb)
    q_obq = quantize_with_scaling(data, sc_obq, cb, H=H)
    err_base = quantization_error(q_base, data, H)
    err_diag = quantization_error(q_diag, data, H)
    err_hessian = quantization_error(q_hessian, data, H)
    err_obq = quantization_error(q_obq, data, H)
    assert err_hessian <= err_base
    assert err_hessian <= err_diag
    assert err_obq <= err_hessian


def test_scaling_modes():
    size = 20
    data = np.random.randn(20, size).astype(np.float32)
    cb = UniformCodebook(9, -3, 3)
    H = random_psd_matrix(size, 10, damp=1.0e-6)
    compute_scaling(data, cb, H, mode="norm")
    compute_scaling(data, cb, H, mode="max")
    compute_scaling(data, cb, H, mode="mse")
    compute_scaling(data, cb, H, mode="diag")
    compute_scaling(data, cb, H, mode="hessian")
    compute_scaling(data, cb, H, mode="diag1")
    compute_scaling(data, cb, H, mode="hessian1")
    compute_scaling(data, cb, H, mode="diag1.8")
    compute_scaling(data, cb, H, mode="hessian1.8")
