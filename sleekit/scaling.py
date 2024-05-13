import torch


def _broadcast_scaling(data, scale, axis):
    """
    Broadcast a 1D scale to multiply alng a given axis
    """
    assert scale.ndim == 1
    new_shape = [1 for i in data.shape]
    new_shape[axis] = -1
    return torch.reshape(scale, new_shape)


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
    return torch.sqrt(torch.mean(torch.square(data), axis=other_axes))


def compute_non_saturating_scaling(data, codebook, axis=0):
    """
    Compute a scaling factor over this axis to have no saturation.

    This yields a non-saturating scaling factor, but not necessarily the tightest one for non-symmetric codebook and data.
    """
    maxcode = torch.maximum(torch.abs(codebook.values).max(), torch.tensor(1.0e-16, dtype=data.dtype))
    other_axes = tuple(i for i in range(data.ndim) if i != axis)
    maxdata = torch.maximum(torch.abs(data).amax(dim=other_axes), torch.tensor(1.0e-16, dtype=data.dtype))
    return maxdata / maxcode


def quantize_with_scaling(data, scale, codebook):
    assert data.ndim == 2
    assert scale.ndim == 1
    assert data.shape[0] == len(scale)
    quant = apply_scaling(data, scale, 0)
    quant = codebook(quant)
    quant = apply_scaling(quant, 1 / scale, 0)
    return quant


def _compute_mse(data, scale, codebook, H=None):
    err = quantize_with_scaling(data, scale, codebook) - data
    if H is None:
        return torch.sum(torch.square(err), axis=1)
    elif H.ndim == 1:
        # Diagonal hessian
        assert err.shape[1] == H.shape[0]
        return torch.sum(torch.expand_dims(H, 0) * torch.square(err - data), axis=1)
    else:
        # Full hessian
        assert H.ndim == 2
        assert err.shape[1] == H.shape[0]
        assert H.shape[1] == H.shape[0]
        return torch.einsum("ij,...i,...j", H, err, err)


def compute_min_mse_scaling(
    data, codebook, axis=0, H=None, min_factor=0.3, max_factor=1.0, grid_size=100
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
    flat_data = torch.permute(data, [axis, *other_axes])

    # Compute a non-saturating scaling
    initial_scale = compute_non_saturating_scaling(flat_data, codebook, 0)

    # Search for the best scaling on a grid
    # TODO: implement a golden section search to make this a lot faster
    scales = torch.linspace(min_factor, max_factor, grid_size, dtype=torch.float32)
    best_choice = torch.full_like(initial_scale, torch.inf)
    best_error = torch.full_like(initial_scale, torch.inf)
    for s in scales:
        scale = s * initial_scale
        error = _compute_mse(flat_data, scale, codebook, H)
        better = error < best_error
        best_error[better] = error[better]
        best_choice[better] = s
    return initial_scale * best_choice
