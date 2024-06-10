import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sleekit:
    """
    Statistics of a layer, with an API compatible with GPTQ.
    """

    def __init__(self, layer):
        self.layer = layer
        weight = layer.weight
        if isinstance(self.layer, nn.Conv1d):
            weight = weight.flatten(1)
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        # TODO: handle transformers.Conv1D
        n = weight.shape[1]
        self.mean = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.hessian = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        self.count = 0

    @property
    def device(self):
        return self.layer.weight.device

    def _prepare_input(self, inp):
        """
        Reshape the inputs so they are 2D
        """
        if isinstance(self.layer, nn.Linear):
            inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        elif isinstance(self.layer, nn.Conv2d):
            inp = F.unfold(
                inp,
                self.layer.kernel_size,
                self.layer.dilation,
                self.layer.padding,
                self.layer.stride,
            )
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        elif isinstance(self.layer, nn.Conv1d):
            # Torch does not support 1D unfold so we do it manually
            inp = torch.unsqueeze(inp, -1)
            inp = F.unfold(
                inp,
                (self.layer.kernel_size, 1),
                (self.layer.dilation, 1),
                (self.layer.padding, 0),
                (self.layer.stride, 1),
            )
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        else:
            raise RuntimeError("Unsupported layer type")
        assert inp.ndim == 2
        inp = inp.float()
        return inp

    def add_batch(self, inp, out):
        """
        Add a new batch to the statistics
        """
        inp = self._prepare_input(inp)
        added = inp.shape[1]
        factor = self.count / (self.count + added)
        self.count += added
        self.mean *= factor
        self.hessian *= factor
        self.mean += inp.sum(dim=1) / self.count
        self.hessian += inp @ inp.t() / self.count

    def export(self, path, npy_format=False):
        """
        Export the statistics as .pt or .npy files
        """
        os.makedirs(path, exist_ok=True)
        if npy_format:
            import numpy as np

            np.save(os.path.join(path, "bias.npy"), self.layer.bias.cpu().numpy())
            np.save(os.path.join(path, "weight.npy"), self.layer.weight.cpu().numpy())
            np.save(os.path.join(path, "mean.npy"), self.mean.cpu().numpy())
            np.save(os.path.join(path, "hessian.npy"), self.hessian.cpu().numpy())
        else:
            torch.save(self.layer.bias.cpu(), os.path.join(path, "bias.pt"))
            torch.save(self.layer.weight.cpu(), os.path.join(path, "weight.pt"))
            torch.save(self.mean.cpu(), os.path.join(path, "mean.pt"))
            torch.save(self.hessian.cpu(), os.path.join(path, "hessian.pt"))

    def quantize(self, cb):
        """
        Quantize the layer using the given codebook
        """
        raise RuntimeError("Quantization is not implemented yet")

    def free(self):
        """
        Free internal tensors
        """
        self.layer = None
        self.mean = None
        self.hessian = None
        self.count = 0
