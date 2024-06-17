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
        if not isinstance(self.layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            raise ValueError(f"Unsupported layer type {type(self.layer)}")
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d)):
            weight = weight.flatten(1)
        # TODO: handle transformers.Conv1D
        n = weight.shape[1]
        # Mean of the inputs
        self.mean = torch.zeros(n, dtype=torch.float32, device=self.device)
        # Mean of the hessian
        self.hessian = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        # Number of samples received
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
            if inp.ndim == 3:
                inp = torch.unsqueeze(inp, 0)
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
            if inp.ndim == 2:
                inp = torch.unsqueeze(inp, 0)
            # Torch does not support 1D unfold so we do it manually
            inp = torch.unsqueeze(inp, -1)
            inp = F.unfold(
                inp,
                (self.layer.kernel_size[0], 1),
                (self.layer.dilation[0], 1),
                (self.layer.padding[0], 0),
                (self.layer.stride[0], 1),
            )
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        else:
            raise RuntimeError(f"Unsupported layer type {type(self.layer)}")
        assert inp.ndim == 2
        inp = inp.float()
        return inp

    def add_batch(self, inp, out=None):
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

    def quantize(self, cb, scaling_mode="mse", order_mode="diag", bias_correction=False, damp=0.01, nb_ls_moves=0):
        """
        Quantize the layer using the given codebook.
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
