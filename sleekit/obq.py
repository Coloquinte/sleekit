import numpy as np


def random_psd_matrix(size, rank, damp=0.0):
    """
    Generate a random positive semidefinite matrix of a given size from the Wishart distribution.
    """
    A = np.random.randn(size, rank).astype(np.float32)
    H = np.matmul(A, A.T)
    dampval = damp * np.linalg.norm(H, ord=2, axis=1)
    return H + dampval * np.eye(size)


def remove_input_bias(H, input_bias):
    """
    Remove the effect of the input bias from the hessian, as it can be removed by bias correction.

    The input bias is the mean of the input samples associated with each weight.
    Mathematically, removing the bias B from the weight error E applies (I - diag(B)) to it.
    We apply this factor to the hessian.
    """
    assert H.ndim == 2
    assert input_bias.ndim == 1
    sz = H.shape[0]
    assert sz == H.shape[1]
    assert sz == input_bias.shape[0]
    scale = np.ones(sz, dtype=np.float32) - input_bias
    scaled_H = np.expand_dims(scale, 0) * H * np.expand_dims(scale, 1)
    return scaled_H


def compute_hessian_chol(H):
    """
    Compute an upper cholesky factor of the inverse hessian.
    """
    # We want to compute the Cholesky factors of the inverse
    # GPTQ first inverses the matrix, then computes the Cholesky factor
    # We do an equivalent transformation that should be slightly faster when optimized

    # Work in reverse order
    H = np.flip(H)
    # Compute the Cholesky factor
    # TODO: handle semi positive definite matrices by catching LinAlgError
    H = np.linalg.cholesky(H)
    # Invert the Cholesky factor
    H = np.linalg.inv(H)
    # Return to the original order
    H = np.flip(H)
    return np.ascontiguousarray(H)


def quantization_error(W, Q, H):
    """
    Compute the error between two weight matrices given the hessian
    """
    E = W - Q
    return np.einsum("ij,...i,...j", H, E, E).mean()


def _quantize_opt_core(W, Hinv, quantizer):
    """
    Core of the quantization algorithm, without block operations.
    """
    Q = W.copy()
    E = np.zeros(W.shape, dtype=np.float32)
    for i in range(W.shape[1]):
        # Quantize the column
        w = Q[:, i]
        q = quantizer(w)
        err = (w - q) / Hinv[i, i]
        E[:, i] = err
        Q[:, i] = q
        # Now correct the error
        Q[:, i + 1 :] -= np.outer(err, Hinv[i, i + 1 :])
    return (Q, E)


def _quantize_opt_block(W, Hinv, quantizer, min_block_size, num_blocks):
    """
    Quantization algorithm using block operations for speed.
    """
    size = W.shape[1]
    if size <= min_block_size:
        return _quantize_opt_core(W, Hinv, quantizer)
    Q = W.copy()
    E = np.zeros(W.shape, dtype=np.float32)
    block_size = max((size + num_blocks - 1) // num_blocks, min_block_size)
    for i in range(0, W.shape[1], block_size):
        # Quantize the block
        e = min(i + block_size, size)
        w = Q[:, i:e]
        q, err = _quantize_opt_block(
            w, Hinv[i:e, i:e], quantizer, min_block_size, num_blocks
        )
        E[:, i:e] = err
        Q[:, i:e] = q
        # Now correct the error
        Q[:, e:] -= err @ Hinv[i:e, e:]
    return (Q, E)


def _quantize_opt_ordered(W, H, quantizer, order, min_block_size, num_blocks):
    """
    Apply quantization optimization with a given ordering.
    """
    # Apply reordering
    W = W[:, order]
    H = H[order][:, order]

    # Apply the algorithm itself
    Hinv = compute_hessian_chol(H)
    Q, _ = _quantize_opt_block(W, Hinv, quantizer, min_block_size, num_blocks)

    # Reverse reordering
    Q = Q[:, np.argsort(order)]
    return Q


def quantize_opt(W, H, quantizer, act_order=True, min_block_size=32, num_blocks=8):
    """
    Quantize the weights with the given quantizer, minimizing the squared error using a GPTQ-like algorithm.

    :param W: Weights (2D array)
    :param H: Hessian of the error (2D array)
    :param quantizer: Function that quantizes its argument and returns it
    :param act_order: Whether to use GPTQ's ordering heuristic
    :returns: Quantized weights
    """
    assert W.ndim == 2
    assert H.ndim == 2
    assert H.shape[0] == H.shape[1]
    assert H.shape[0] == W.shape[1]
    assert min_block_size >= 1
    W = W.astype(np.float32)
    H = H.astype(np.float32)

    # Reorder if required, by order of decreasing diagonal elements
    if act_order:
        order = np.argsort(-np.diag(H))
    else:
        order = np.arange(W.shape[1])

    Q = _quantize_opt_ordered(W, H, quantizer, order, min_block_size, num_blocks)

    return Q
