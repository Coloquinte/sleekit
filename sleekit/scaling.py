import numpy as np

from sleekit.obq import quantize_opt


def _broadcast_scaling(data, scale, axis):
    """
    Broadcast a 1D scale to multiply alng a given axis
    """
    assert scale.ndim == 1
    new_shape = [1 for i in data.shape]
    new_shape[axis] = -1
    return scale.reshape(new_shape)


def apply_scaling(data, scale, axis=0):
    """
    Apply a scaling factor over this axis.
    """
    return data / _broadcast_scaling(data, scale, axis)


def apply_scaling_in_place(data, scale, axis=0):
    """
    Apply a scaling factor over this axis in place.
    """
    data /= _broadcast_scaling(data, scale, axis)


def compute_norm_scaling(data, axis=0):
    """
    Compute a scaling factor over this axis to have squared average equal to 1.
    """
    other_axes = tuple(i for i in range(data.ndim) if i != axis)
    sqnorm = np.mean(np.square(data), axis=other_axes)
    return np.sqrt(np.maximum(sqnorm, 1.0e-16))


def compute_non_saturating_scaling(data, codebook, axis=0):
    """
    Compute a scaling factor over this axis to have no saturation.
    """
    if codebook.min() >= 0 or codebook.max() <= 0:
        raise RuntimeError("Codebook should have both negative and positive values.")
    other_axes = tuple(i for i in range(data.ndim) if i != axis)
    mindata, maxdata = data.min(axis=other_axes), data.max(axis=other_axes)
    mincode, maxcode = codebook.min(), codebook.max()
    scale = np.maximum(maxdata / maxcode, mindata / mincode)
    scale = np.maximum(scale, np.float32(1.0e-16))
    return scale


def quantize_with_scaling(
    data, scale, quantizer, H=None, act_order="diag", nb_ls_moves=0
):
    """
    Quantize the weights after applying a scaling factor.

    :param data: Weights (2D array)
    :param quantizer: Function that quantizes its argument and returns it
    :param H: Hessian of the error (optional 2D array)
    :param act_order: Ordering heuristic to use
    :returns: Quantized weights
    """
    assert data.ndim == 2
    assert scale.ndim == 1
    assert data.shape[0] == scale.size
    quant = apply_scaling(data, scale, 0)
    if H is not None:
        quant = quantize_opt(
            quant, H, quantizer, act_order=act_order, nb_ls_moves=nb_ls_moves
        )
    else:
        quant = quantizer(quant)
    quant = apply_scaling(quant, 1 / scale, 0)
    return quant


def _compute_mse(H, E):
    if H is None:
        return np.sum(np.square(E), axis=1)
    if H.ndim == 1:
        # Diagonal hessian
        assert E.shape[1] == H.shape[0]
        return np.sum(np.expand_dims(H, 0) * np.square(E), axis=1)
    # Full hessian
    assert H.ndim == 2
    assert E.shape[1] == H.shape[0]
    assert H.shape[1] == H.shape[0]
    return ((E @ H) * E).sum(axis=-1)


def compute_min_mse_scaling(
    data,
    codebook,
    axis=0,
    H=None,
    obq=False,
    min_factor=0.05,
    max_factor=1.0,
    grid_size=100,
):
    """
    Compute a scaling factor to minimize the squared error with respect to the codebook.

    :param data: Data to quantize
    :param codebook: Codebook to be used
    :param axis: Axis to apply the scale to
    :param H: Hessian matrix
    :param min_factor: Minimum scaling factor compared to a non-saturating scaling
    :param max_factor: Maximum scaling factor compared to a non-saturating scaling
    :param grid_size: Number of points for the grid search
    """
    # First flatten to 2D with scaling on the first dimension
    other_axes = tuple(i for i in range(data.ndim) if i != axis)
    flat_data = np.transpose(data, [axis, *other_axes])

    # Compute a non-saturating scaling
    initial_scale = compute_non_saturating_scaling(flat_data, codebook, 0)

    # Search for the best scaling on a grid
    # TODO: implement a golden section search to make this a lot faster
    scales = np.linspace(min_factor, max_factor, grid_size, dtype=np.float32)
    best_choice = np.full(initial_scale.size, np.inf, dtype=np.float32)
    best_error = np.full(initial_scale.size, np.inf, dtype=np.float32)
    for s in scales:
        scale = s * initial_scale
        # TODO: do the inverse hessian computation only once for OBQ
        quant = quantize_with_scaling(
            flat_data, scale, codebook, H if obq and H.ndim == 2 else None
        )
        error = _compute_mse(H, quant - flat_data)
        better = error < best_error
        best_error[better] = error[better]
        best_choice[better] = s
    return initial_scale * best_choice


def compute_scaling(
    data,
    codebook,
    H,
    mode="mse",
    axis=0,
    min_factor=0.05,
    max_factor=1.0,
    grid_size=100,
):
    if mode == "max":
        return compute_non_saturating_scaling(data, codebook, axis)
    if mode == "norm":
        return compute_norm_scaling(data, axis)
    obq = False
    if mode == "mse":
        H = None
    elif mode == "hessian":
        pass
    elif mode == "diag":
        H = H.diagonal()
    elif mode == "diag1":
        H = H.diagonal() + 0.01 * H.diagonal().mean()
    elif mode == "diag3":
        H = H.diagonal() + 0.03 * H.diagonal().mean()
    elif mode == "diag10":
        H = H.diagonal() + 0.1 * H.diagonal().mean()
    elif mode == "obq":
        obq = True
    else:
        raise RuntimeError(f"Unknown scaling mode {mode}")
    return compute_min_mse_scaling(
        data,
        codebook,
        axis,
        H=H,
        obq=obq,
        grid_size=grid_size,
        min_factor=min_factor,
        max_factor=max_factor,
    )
