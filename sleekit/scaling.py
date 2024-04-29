import numpy as np


def scale_norm(data, axis):
    """
    Rescale the data in-place to have squared average equal to 1. Return a 1D array of the scaling factors.
    """
    other_axes = tuple(i for i in range(data.ndim) if i != axis)
    scale = np.sqrt(np.mean(np.square(data), axis=other_axes, keepdims=True))
    data /= scale
    return np.reshape(scale, (-1,))
