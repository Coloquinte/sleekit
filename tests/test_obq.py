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
    H = random_psd_matrix(4, 6)

    # GPTQ way of computing it
    gptq = np.linalg.inv(H)
    gptq = np.linalg.cholesky(gptq).T

    # Custom faster way
    L = compute_hessian_chol(H)

    assert np.allclose(L, gptq)


def test_obq():
    H = random_psd_matrix(10, 2) + 1.0e-6 * np.eye(10)
    W = 10.0 * np.random.randn(2, 10)
    quantizer = lambda x: np.round(x)
    Q = quantize_opt(W, H, quantizer)
    assert Q.shape == W.shape
