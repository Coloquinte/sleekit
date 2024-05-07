import numpy as np

from sleekit.obq import (
    random_psd_matrix,
    compute_hessian_chol,
    quantize_opt,
    quantization_error,
)


def test_psd():
    H = random_psd_matrix(10, 20)
    assert H.shape == (10, 10)
    assert np.all(np.linalg.eigvalsh(H) >= 0)
    assert np.linalg.matrix_rank(H) == 10


def test_hessian():
    H = random_psd_matrix(4, 6, 1.0e-6)

    # GPTQ way of computing it
    gptq = np.linalg.inv(H)
    gptq = np.linalg.cholesky(gptq).T

    # Custom faster way
    U = compute_hessian_chol(H)

    assert np.allclose(U, gptq)
    assert np.allclose(np.linalg.inv(U.T @ U), H)


def test_obq():
    size = 1000
    damp = 1.0e-6
    H = random_psd_matrix(size, 2, damp)
    W = 10.0 * np.random.randn(10, size)
    quantizer = lambda x: np.round(x)
    Q_ordered = quantize_opt(W, H, quantizer, act_order=True, block_size=size)
    Q_unordered = quantize_opt(W, H, quantizer, act_order=False, block_size=size)
    assert Q_unordered.shape == W.shape
    assert Q_ordered.shape == W.shape
    error_direct = quantization_error(W, quantizer(W), H)
    error_ordered = quantization_error(W, Q_ordered, H)
    error_unordered = quantization_error(W, Q_unordered, H)
    # We may be unlucky but given the sizes involved we should be fine
    assert error_unordered <= error_direct
    assert error_ordered <= error_unordered


def test_blockobq():
    size = 100
    damp = 1.0e-6
    H = random_psd_matrix(size, 2, damp)
    W = 10.0 * np.random.randn(1, size)
    quantizer = lambda x: np.round(x)
    Q1 = quantize_opt(W, H, quantizer, block_size=size, act_order=False)
    Q2 = quantize_opt(W, H, quantizer, block_size=1, act_order=False)
    # We may be unlucky for close to values close to a half-integer, but should be fine
    assert np.allclose(Q1, Q2)


def test_reopt():
    size = 1000
    damp = 1.0e-6
    H = random_psd_matrix(size, 2, damp)
    W = 10.0 * np.random.randn(10, size)
    quantizer = lambda x: np.round(x)
    Q_noopt = quantize_opt(W, H, quantizer, reopt=False)
    Q_reopt = quantize_opt(W, H, quantizer, reopt=True)
    error_noopt = quantization_error(W, Q_noopt, H)
    error_reopt = quantization_error(W, Q_reopt, H)
    # We may be unlucky but given the sizes involved we should be fine
    assert error_reopt <= error_noopt
