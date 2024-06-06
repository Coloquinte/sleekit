import numpy as np


class Codebook:
    """
    A codebook gives a list of codebook values (sorted) and a list
    """

    def __init__(self, values, limits=None):
        """
        Create a codebook from a list of values, and optionally a list of limits between bins.
        """
        self.values = np.array(values, dtype=np.float32)
        if limits is not None:
            self.thresholds = np.array(limits, dtype=np.float32)
        else:
            self.values.sort()
            self.thresholds = (self.values[:-1] + self.values[1:]) / 2
        self.check()

    def clone(self):
        """
        Return a copy of the codebook.
        """
        return Codebook(self.values.copy(), self.thresholds.copy())

    def check(self):
        """
        Run consistency check
        """
        assert self.values.ndim == 1
        assert self.values.size > 0
        assert np.isfinite(self.values).all()
        assert (self.values[1:] > self.values[:-1]).all()
        assert self.thresholds.ndim == 1
        assert self.thresholds.size == self.values.size - 1
        assert np.isfinite(self.thresholds).all()
        assert (self.thresholds[1:] > self.thresholds[:-1]).all()
        assert (self.thresholds >= self.values[:-1]).all()
        assert (self.thresholds <= self.values[1:]).all()

    def __len__(self):
        return len(self.values)

    def min(self):
        """
        Minimum value in the codebook
        """
        return self.values[0]

    def max(self):
        """
        Maximum value in the codebook
        """
        return self.values[-1]

    def quantize_index(self, data):
        """
        Quantize data to their index in the codebook.
        """
        vals = np.digitize(data, self.thresholds)
        if len(self) <= 2**8:
            return vals.astype(np.uint8)
        if len(self) <= 2**16:
            return vals.astype(np.uint16)
        return vals.astype(np.uint32)

    def quantize_value(self, data):
        """
        Quantize data to their value in the codebook.
        """
        return self.values[self.quantize_index(data)]

    def quantize_up(self, data):
        """
        Quantize data to the value above in the codebook
        """
        ind = np.digitize(data, self.thresholds)
        vals = np.concatenate((np.delete(self.values, 0), [np.inf]))
        return vals[ind]

    def quantize_down(self, data):
        """
        Quantize data to the value below in the codebook
        """
        ind = np.digitize(data, self.thresholds)
        vals = np.concatenate(([-np.inf], np.delete(self.values, -1)))
        return vals[ind]

    def __call__(self, data):
        """
        Quantize data to their value in the codebook.
        """
        return self.quantize_value(data)

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
        Return the centroid for each codeword in the data based on the current bin limits.
        """
        labels = self.quantize_index(data)
        ret = []
        for k in range(len(self.values)):
            d = data[labels == k]
            if len(d) != 0:
                # Typical case: we use the centroid
                ret.append(d.mean())
            else:
                # Edge case: no data point in the bin
                if k == 0:
                    ret.append(self.thresholds[0] - 1.0e-6)
                elif k == len(self.values) - 1:
                    ret.append(self.thresholds[-1] + 1.0e-6)
                else:
                    ret.append((self.thresholds[k - 1] + self.thresholds[k]) / 2)
        return np.array(ret)

    def remove_unused(self, data):
        """
        Remove unused codewords from the codebook; the old limits are kept and will be suboptimal.
        """
        quant = self.quantize_index(data)
        counts = np.bincount(quant, minlength=len(self.values))
        if (counts == 0).any():
            self.values = self.values[counts != 0]
            # Remove limits where the left bin has been removed
            self.thresholds = self.thresholds[counts[:-1] != 0]
            # Remove the last limit if needed
            if counts[-1] == 0:
                self.thresholds = self.thresholds[:-1]
            self.check()

    def improve(self, data, lagrange_mult=0.0):
        """
        Perform one round of codebook improvement using the Lloyd-Max algorithm.
        """
        if lagrange_mult != 0.0:
            self.remove_unused(data)
            v = self.values
            l = -np.log2(self.probabilities(data))
            penalty = (l[1:] - l[:-1]) / (v[1:] - v[:-1])
            # Assign each data point to the nearest codeword plus a penalty to gear towards low entropy
            self.thresholds = (v[:-1] + v[1:]) / 2 + lagrange_mult * penalty / 2
            # Workaround when the penalty throws the ordering away
            self.thresholds.sort()
        else:
            v = self.values
            # Assign each data point to the nearest codeword
            self.thresholds = (v[:-1] + v[1:]) / 2
        # Update each codeword to the centroid or its datapoints
        self.values = self.centroids(data)
        self.check()

    def close_to(self, other, tol=1.0e-6):
        """
        Returns whether two codebooks are very similar.
        """
        if len(self) != len(other):
            return False
        data_range = max(self.values.max() - self.values.min(), 1.0e-10)
        return np.allclose(self.values, other.values, atol=tol * data_range)

    @staticmethod
    def random(data, codebook_size):
        """
        Create a codebook from random sampling.
        """
        values = np.unique(data)
        return Codebook(
            np.random.choice(values, min(codebook_size, values.size), replace=False)
        )

    @staticmethod
    def uniform(codebook_size, min_val, max_val):
        """
        Create a uniform codebook.
        """
        assert min_val <= max_val
        return Codebook(np.linspace(min_val, max_val, codebook_size))

    @staticmethod
    def nf4():
        """
        Create the codebook for the NormalFloat4 datatype
        """
        return Codebook(
            [
                -1.0,
                -0.6961928009986877,
                -0.5250730514526367,
                -0.39491748809814453,
                -0.28444138169288635,
                -0.18477343022823334,
                -0.09105003625154495,
                0.0,
                0.07958029955625534,
                0.16093020141124725,
                0.24611230194568634,
                0.33791524171829224,
                0.44070982933044434,
                0.5626170039176941,
                0.7229568362236023,
                1.0,
            ]
        )

    @staticmethod
    def equiprobable(data, codebook_size):
        """
        Create a codebook where each codeword is equiprobable.
        """
        parts = np.array_split(np.sort(data), codebook_size)
        parts = [p for p in parts if len(p) > 0]
        limits = [(parts[k][-1] + parts[k + 1][0]) / 2 for k in range(len(parts) - 1)]
        # Temporary values
        values = [p.mean() for p in parts]
        cb = Codebook(values, limits)
        # Reset to the centroids
        cb.values = cb.centroids(data)
        return cb


def lloyd_max(
    data,
    codebook_size,
    lagrange_mult=0.0,
    max_iter=100,
    tol=1e-6,
    random_init=False,
    sample_count=None,
):
    """
    Lloyd-Max algorithm for scalar quantization.
    Returns a codebook that minimizes a combination of the mse and the entropy.
    """
    data = np.reshape(data, (-1,))
    if sample_count is not None:
        nsamples = codebook_size * sample_count
        if nsamples < len(data):
            data = np.random.choice(data, nsamples, replace=False)
    data = np.sort(data)
    if random_init:
        codebook = Codebook.random(data, codebook_size)
    else:
        codebook = Codebook.equiprobable(data, codebook_size)
    for i in range(max_iter):
        new_codebook = codebook.clone()
        new_codebook.improve(data, lagrange_mult)
        if new_codebook.close_to(codebook, tol):
            break
        codebook = new_codebook
    return codebook
