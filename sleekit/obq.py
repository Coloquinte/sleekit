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

    The hessian is the mean of the hessian, and the input bias is the mean of the input samples.
    The error due to this bias can be compensated in the bias of the layer, a technique known as bias correction.
    """
    assert H.ndim == 2
    assert input_bias.ndim == 1
    assert H.shape[0] == H.shape[1]
    assert H.shape[0] == input_bias.shape[0]
    return H - np.outer(input_bias, input_bias)


def remove_dead_values(H, W, pcdamp=0.01):
    """
    Make the Hessian diagonal non-zero, add dampening term, and zero out dead weights.
    """
    mean_diag = np.mean(np.diag(H))

    # Remove dead elements
    dead = np.diag(H) == 0
    H[dead, dead] = mean_diag
    W[:, dead] = 0

    # Add dampening term
    H += np.diag(np.full(H.shape[0], 0.01 * pcdamp * mean_diag))


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


def channelwise_error(W, Q, H):
    """
    Compute the channel error between two weight matrices given the hessian
    """
    E = W - Q
    # Einsum is much too slow: np.einsum("ij,...i,...j", H, E, E).mean()
    return ((E @ H) * E).sum(axis=-1)


def quantization_error(W, Q, H):
    """
    Compute the mean channel error between two weight matrices given the hessian
    """
    # Einsum is much too slow: np.einsum("ij,...i,...j", H, E, E).mean()
    return channelwise_error(W, Q, H).mean()


def compute_gain(W, Q, H, candidates):
    """
    Compute the gain of changing the quantized weights to the candidates.

    With D the change between Q and the candidate (with a single non-zero), this gain is
    (Q - W) @ H @ (Q - W) - (Q + D - W) @ H @ (Q + D - W).
    Simplifying it yields - 2 * (Q - W) @ H @ D - D @ H @ D, which we can compute for all
    candidates in a single matrix operation.
    """
    delta = Q - W
    D = candidates - Q
    return -np.square(D) * np.diag(H) - 2 * (delta @ H) * D


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


def _cholesky_ordering(H):
    """
    Compute a good ordering of the matrix based on pivoted Cholesky decomposition.
    The diagonal of the Cholesky decomposition gives the conditional variance with respect to the previous variables.
    We pick a greedy ordering that maximizes the diagonal elements of the Cholesky decomposition.
    This is better than the GPTQ ordering, which is based on the diagonal of the Hessian, as this factors prior decisions in the ordering.
    """
    n = H.shape[0]
    H = H.copy()
    L = H.copy()
    order = np.arange(n)
    for k in range(n):
        # Find the pivot
        pivot = np.argmax(np.abs(np.diag(L)[k:])) + k
        # Swap rows and columns
        L[[k, pivot], :] = L[[pivot, k], :]
        L[:, [k, pivot]] = L[:, [pivot, k]]
        order[[k, pivot]] = order[[pivot, k]]
        # Update the matrix
        b = L[k, k + 1 :]
        L[k + 1 :, k + 1 :] -= np.outer(b, b) / L[k, k]
        L[k, k] = np.sqrt(L[k, k])
        L[k, k + 1 :] = 0
        L[k + 1 :, k] /= L[k, k]
    # Should work in general despite imprecisions
    assert np.allclose(L @ L.T, H[order][:, order], rtol=1.0e-4, atol=1.0e-4)
    return order


def quantize_opt(
    W, H, quantizer, act_order="diag", min_block_size=32, num_blocks=8, nb_ls_moves=0
):
    """
    Quantize the weights with the given quantizer, minimizing the squared error using a GPTQ-like algorithm.

    :param W: Weights (2D array)
    :param H: Hessian of the error (2D array)
    :param quantizer: Function that quantizes its argument and returns it
    :param act_order: Ordering heuristic to use
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
    if act_order == "err":
        Q = quantizer(W)
        err = np.abs(Q - W).sum(axis=0)
        order = np.argsort(-np.diag(H) * err)
    elif act_order == "sqerr":
        Q = quantizer(W)
        sqerr = np.square(Q - W).sum(axis=0)
        order = np.argsort(-np.diag(H) * sqerr)
    elif act_order == "combined_diag":
        order = np.argsort(-np.diag(H) / np.diag(np.linalg.inv(H)))
    elif act_order == "inv_diag":
        order = np.argsort(np.diag(np.linalg.inv(H)))
    elif act_order == "pivot":
        order = _cholesky_ordering(H)
    elif act_order == "diag":
        order = np.argsort(-np.diag(H))
    elif act_order == "none":
        order = np.arange(W.shape[1])
    else:
        raise RuntimeError(f"Invalid act_order value {act_order}")

    Q = _quantize_opt_ordered(W, H, quantizer, order, min_block_size, num_blocks)
    Q = quantize_local_search(W, Q, H, quantizer, nb_ls_moves)
    return Q


class LocalSearchQuantizer:
    def __init__(self, W, Q, H, quantizer):
        assert W.ndim == 2
        assert H.ndim == 2
        assert H.shape[0] == H.shape[1]
        assert H.shape[0] == W.shape[1]
        assert Q.shape == W.shape
        self.W = W
        self.Q = Q.copy()
        self.H = H
        self.quantizer = quantizer
        self.recompute_error()
        self.recompute_candidates()
        self.recompute_gains()

    @property
    def nchannels(self):
        return self.W.shape[0]

    def recompute_error(self):
        self.err = channelwise_error(self.W, self.Q, self.H)

    def recompute_candidates(self):
        self.Q_up = self.quantizer.quantize_up(self.Q)
        self.Q_down = self.quantizer.quantize_down(self.Q)

    def recompute_gains(self):
        self.gain_up = compute_gain(self.W, self.Q, self.H, self.Q_up)
        self.gain_down = compute_gain(self.W, self.Q, self.H, self.Q_down)

    def do_move(self):
        change_up = self.gain_up.max(axis=1)
        change_down = self.gain_down.max(axis=1)
        change = np.maximum(change_up, change_down)
        change = np.maximum(change, 0)
        is_up = (change_up > change_down) & (change_up > 0)
        is_down = ~is_up & (change_down > 0)

        # Pick the new values
        inds_up = np.arange(self.nchannels)[is_up]
        best_up = self.gain_up.argmax(axis=1)[is_up]
        inds_down = np.arange(self.nchannels)[is_down]
        best_down = self.gain_down.argmax(axis=1)[is_down]
        self.Q[inds_up, best_up] = self.Q_up[inds_up, best_up]
        self.Q[inds_down, best_down] = self.Q_down[inds_down, best_down]
        self.err -= change

        # Update the candidates and gain
        # TODO: Should be incremental
        self.recompute_candidates()
        self.recompute_gains()


def quantize_local_search(W, Q, H, quantizer, nb_moves):
    """
    Perform a local search to improve the quantization error.
    """
    if nb_moves == 0:
        return Q
    ls = LocalSearchQuantizer(W, Q, H, quantizer)
    for _ in range(nb_moves):
        ls.do_move()
    return ls.Q
