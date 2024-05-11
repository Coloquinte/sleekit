import numpy as np


def _broadcast_scaling(data, scale, axis):
    """
    Broadcast a 1D scale to multiply alng a given axis
    """
    assert scale.ndim == 1
    new_shape = [1 for i in data.shape]
    new_shape[axis] = -1
    return np.reshape(scale, new_shape)


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
    return np.sqrt(np.mean(np.square(data), axis=other_axes))


def compute_non_saturating_scaling(data, codebook, axis=0):
    """
    Compute a scaling factor over this axis to have no saturation.

    This yields a non-saturating scaling factor, but not necessarily the tightest one for non-symmetric codebook and data.
    """
    maxcode = np.maximum(np.abs(codebook.values).max(), np.float32(1.0e-16))
    other_axes = tuple(i for i in range(data.ndim) if i != axis)
    maxdata = np.maximum(np.abs(data).max(axis=other_axes), np.float32(1.0e-16))
    return maxdata / maxcode


def _quantize(data, scale, codebook):
    assert data.ndim == 2
    assert scale.ndim == 1
    assert data.shape[0] == scale.size
    quant = apply_scaling(data, scale, 0)
    quant = codebook(quant)
    quant = apply_scaling(quant, 1 / scale, 0)
    return quant


def _compute_mse(data, scale, codebook, H=None):
    err = _quantize(data, scale, codebook) - data
    if H is None:
        return np.sum(np.square(err), axis=1)
    elif H.ndim == 1:
        # Diagonal hessian
        assert err.shape[1] == H.shape[0]
        return np.sum(np.expand_dims(H, 0) * np.square(err - data), axis=1)
    else:
        # Full hessian
        assert H.ndim == 2
        assert err.shape[1] == H.shape[0]
        assert H.shape[1] == H.shape[0]
        return np.einsum("ij,...i,...j", H, err, err)


def compute_min_mse_scaling(
    data, codebook, axis=0, H=None, min_factor=0.3, max_factor=1.0, grid_size=100
):
    """
    Compute a scaling factor to minimize the squared error with respect to the codebook.
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
        error = _compute_mse(flat_data, scale, codebook, H)
        better = error < best_error
        best_error[better] = error[better]
        best_choice[better] = s
    return initial_scale * best_choice
