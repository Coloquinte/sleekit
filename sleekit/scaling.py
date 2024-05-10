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
    maxcode = np.maximum(np.abs(codebook.values).max(), 1.0e-16)
    other_axes = tuple(i for i in range(data.ndim) if i != axis)
    maxdata = np.maximum(np.abs(data).max(axis=other_axes), 1.0e-16)
    return maxdata / maxcode
