import numpy as np


def random_psd_matrix(size, rank, damp=0.0):
    """
    Generate a random positive semidefinite matrix of a given size from the Wishart distribution.
    """
    A = np.random.randn(size, rank)
    return np.matmul(A, A.T) + damp * np.eye(size)


def compute_hessian_chol(H):
    """
    Compute an upper cholesky factor of the inverse hessian.
    """
    # GPTQ first inverses the matrix, then computes the Cholesky factor
    # We do the inversion on only the triangular factor instead
    H = np.flip(H)
    H = np.linalg.cholesky(H)
    H = np.linalg.inv(H)
    H = np.flip(H)
    return np.ascontiguousarray(H)


def quantization_error(W, Q, H):
    """
    Compute the error between two weight matrices given the hessian
    """
    E = W - Q
    return np.einsum("ij,...i,...j", H, E, E).mean()


def quantize_opt(W, H, quantizer, act_order=True):
    """
    Quantize the weights with the given quantizer, minimizing the squared error using a GPTQ-like algorithm.

    :param W: Weights (2D array)
    :param H: Hessian of the error (2D array)
    :param quantizer: function that quantizes its argument and returns it
    :param act_order: whether to use GPTQ's ordering heuristic
    """
    assert W.ndim == 2
    assert H.ndim == 2
    assert H.shape[0] == H.shape[1]
    assert H.shape[0] == W.shape[1]

    # Reorder if required, by order of decreasing diagonal elements
    if act_order:
        order = np.argsort(-np.diag(H))
        W = W[:, order]
        H = H[order][:, order]

    # Apply the algorithm itself
    Hinv = compute_hessian_chol(H)
    Q = W.copy()
    for i in range(W.shape[1]):
        # Quantize the column
        w = Q[:, i : i + 1]
        q = quantizer(w)
        err = (w - q) / Hinv[i, i]
        Q[:, i : i + 1] = q
        # Now correct the error
        Q[:, i + 1 :] -= err @ Hinv[i : i + 1, i + 1 :]

    # Reorder if required
    if act_order:
        Q = Q[:, np.argsort(order)]

    return Q
