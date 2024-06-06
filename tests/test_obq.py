import numpy as np

from sleekit.obq import (
    random_psd_matrix,
    compute_hessian_chol,
    quantize_opt,
    quantization_error,
    remove_input_bias,
    compute_gain,
    channelwise_error,
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
    Q_ordered = quantize_opt(W, H, quantizer, act_order="diag", min_block_size=size)
    Q_unordered = quantize_opt(W, H, quantizer, act_order="none", min_block_size=size)
    Q_cholesky = quantize_opt(W, H, quantizer, act_order="pivot", min_block_size=size)
    assert Q_unordered.shape == W.shape
    assert Q_ordered.shape == W.shape
    assert Q_cholesky.shape == W.shape
    error_direct = quantization_error(W, quantizer(W), H)
    error_ordered = quantization_error(W, Q_ordered, H)
    error_unordered = quantization_error(W, Q_unordered, H)
    error_cholesky = quantization_error(W, Q_cholesky, H)
    # We may be unlucky but given the sizes involved we should be fine
    assert error_unordered <= error_direct
    assert error_ordered <= error_unordered
    assert error_cholesky <= error_unordered


def test_blockobq():
    size = 64
    damp = 1.0e-6
    H = random_psd_matrix(size, 2, damp)
    W = 10.0 * np.random.randn(1, size)
    quantizer = lambda x: np.round(x)
    Q = quantize_opt(W, H, quantizer, min_block_size=1, act_order="none")
    for b in [3, 4, 7, 8, 63, 64]:
        for n in [2, 4]:
            Q_block = quantize_opt(
                W, H, quantizer, min_block_size=b, num_blocks=n, act_order="none"
            )
            # We may be unlucky for close to values close to a half-integer, but should be fine
            assert np.allclose(Q, Q_block)


def test_input_bias_removal():
    size = 16
    samples = 32

    # Bias removal when the samples are summed
    X = np.random.randn(size, samples)
    H = np.matmul(X, X.T)
    input_bias = np.mean(X, axis=1)
    # First way of computing bias removal
    removed_H = H - samples * np.outer(input_bias, input_bias)
    # Second way of computing bias removal
    unbiased_X = X - np.expand_dims(input_bias, 1)
    unbiased_H = np.matmul(unbiased_X, unbiased_X.T)
    # Check that they are similar
    assert np.allclose(removed_H, unbiased_H)
    # Check that they are positive semidefinite
    assert np.all(np.linalg.eigvalsh(H) >= 0)
    assert np.all(np.linalg.eigvalsh(removed_H) >= 0)

    # Bias removal when the samples are averaged
    X = np.random.randn(size, samples)
    H = np.matmul(X, X.T) / samples
    input_bias = np.mean(X, axis=1)
    # First way of computing bias removal
    removed_H = H - np.outer(input_bias, input_bias)
    # Second way of computing bias removal
    unbiased_X = X - np.expand_dims(input_bias, 1)
    unbiased_H = np.matmul(unbiased_X, unbiased_X.T) / samples
    # Check that they are similar
    assert np.allclose(removed_H, unbiased_H)
    # Check that they are positive semidefinite
    assert np.all(np.linalg.eigvalsh(H) >= 0)
    assert np.all(np.linalg.eigvalsh(removed_H) >= 0)

    # Library implementation is for averaged samples
    library_H = remove_input_bias(H, input_bias)
    assert np.allclose(removed_H, library_H)


def compute_gain_exhaustive(W, Q, H, candidates):
    """
    Compute the gain of changing the quantized weights to the candidates
    """
    assert W.ndim == 2
    assert H.ndim == 2
    assert Q.shape == W.shape
    assert candidates.shape == W.shape
    assert H.shape[0] == H.shape[1]
    assert H.shape[0] == W.shape[1]
    gains = np.zeros_like(Q)
    err = channelwise_error(W, Q, H)
    for i in range(W.shape[1]):
        cur = Q.copy()
        cur[:, i] = candidates[:, i]
        gains[:, i] = err - channelwise_error(W, cur, H)
    return gains


def test_gain_computation():
    sz = 16
    channels = 10
    W = np.random.randn(channels, sz)
    H = random_psd_matrix(sz, 10)
    Q = np.round(W)
    Q_next = Q + np.square(np.random.randn(channels, sz))
    gain_1 = compute_gain_exhaustive(W, Q, H, Q_next)
    gain_2 = compute_gain(W, Q, H, Q_next)
    assert np.allclose(gain_1, gain_2)
