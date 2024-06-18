import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleekit.codebook import UniformCodebook
from sleekit.obq import remove_input_bias
from sleekit.scaling import compute_scaling, quantize_with_scaling


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

    def quantize_basic(self, nbits):
        """
        A typical quantization method, without our improvements.
        """
        return self.quantize(
            nbits,
            scaling_mode="mse",
            order_mode="diag",
            bias_correction=False,
            damp=0.01,
            nb_ls_moves=0,
        )

    def quantize_sleekit_light(self, nbits):
        """
        Sleekit "light" version, with no computational cost
        """
        return self.quantize(
            nbits,
            scaling_mode="diag",
            order_mode="sqerr",
            bias_correction=True,
            damp=0.03,
            nb_ls_moves=0,
        )

    def quantize_sleekit_heavy(self, nbits):
        """
        Sleekit "heavy" version, with more intensive computations
        """
        return self.quantize(
            nbits,
            scaling_mode="hessian",
            order_mode="sqerr",
            bias_correction=True,
            damp=0.03,
            nb_ls_moves=1000,
        )

    def quantize(
        self,
        nbits,
        scaling_mode="mse",
        order_mode="diag",
        bias_correction=False,
        damp=0.01,
        nb_ls_moves=0,
        grid_size=100,
        min_factor=0.05,
        max_factor=1.0,
    ):
        """
        Quantize the layer to the required precision.
        """
        cb = UniformCodebook(2**nbits, -1, 1)
        H = self.hessian.numpy()
        mean = self.mean.numpy()
        if bias_correction:
            H = remove_input_bias(H, mean)
        weight = self.layer.weight.data.flatten(1).numpy()
        sc = compute_scaling(
            weight,
            cb,
            H=H,
            mode=scaling_mode,
            grid_size=grid_size,
            min_factor=min_factor,
            max_factor=max_factor,
        )
        quant_weight = quantize_with_scaling(
            weight,
            sc,
            cb,
            H=H,
            act_order=order_mode,
            damp=damp,
            nb_ls_moves=nb_ls_moves,
        )

        self.layer.weight.data = torch.from_numpy(quant_weight).reshape(self.layer.weight.shape)
        if bias_correction:
            err = torch.from_numpy(weight - quant_weight).flatten(1)
            delta = (err * self.mean).sum(axis=1)
            self.layer.bias.data += delta

    def free(self):
        """
        Free internal tensors
        """
        self.layer = None
        self.mean = None
        self.hessian = None
        self.count = 0
