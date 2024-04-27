import numpy as np


class Codebook:
    """
    A codebook gives a list of codebook values (sorted) and a list
    """

    def __init__(self, values, limits=None):
        """
        Create a codebook from a list of values, and optionally a list of limits between bins.
        """
        self.values = np.array(values)
        if limits is not None:
            self.limits = np.array(limits)
        else:
            self.values.sort()
            self.limits = (self.values[:-1] + self.values[1:]) / 2
        self.check()

    def check(self):
        assert self.values.ndim == 1
        assert self.values.size > 0
        assert sorted(self.values)
        assert self.limits.ndim == 1
        assert self.limits.size == self.values.size - 1
        assert (self.limits >= self.values[:-1]).all()
        assert (self.limits <= self.values[1:]).all()

    def quantize_index(self, data):
        """
        Quantize data to their index in the codebook.
        """
        return np.digitize(data, self.limits)

    def quantize_value(self, data):
        """
        Quantize data to their value in the codebook.
        """
        return self.values[self.quantize_index(data)]

    def probabilities(self, data):
        """
        Return the probability of each codeword in the data.
        """
        quant = self.quantize_index(data)
        return np.bincount(quant, minlength=len(self.values)) / len(data)

    def entropy(self, data):
        """
        Return the entropy of the data with this codebook.
        """
        probs = self.probabilities(data)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def mse(self, data):
        """
        Return the mean squared error of the data with this codebook.
        """
        quant = self.quantize_value(data)
        return np.mean(np.square(data - quant))

    def centroids(self, data):
        """
        Return the centroid for each codeword in the data.
        """
        labels = self.quantize_index(data)
        ret = []
        for k, v in enumerate(self.values):
            d = data[labels == k]
            ret.append(v if len(d) == 0 else d.mean())
        return np.array(ret)


def lloyd(data, codebook_size, lagrange_mult=0.0, max_iter=100, tol=1e-6):
    """
    Lloyd's algorithm for scalar quantization.
    Returns a codebook that minimizes a combination of the mse and the etropy.
    """
    assert data.ndim == 1
    # Initialize the codebook
    data_range = data.max() - data.min()
    codebook = Codebook(np.linspace(data.min(), data.max(), codebook_size))
    for i in range(max_iter):
        # Assign each data point to the nearest codeword
        labels = codebook.quantize_index(data)
        # Update each codeword to the centroid or its datapoints
        new_codebook = Codebook(codebook.centroids(data))
        if np.allclose(new_codebook.values, codebook.values, atol=tol * data_range):
            break
        codebook = new_codebook
    return codebook
