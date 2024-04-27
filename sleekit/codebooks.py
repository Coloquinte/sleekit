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
