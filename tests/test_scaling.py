import math
import torch

from sleekit.codebook import Codebook
from sleekit.obq import random_psd_matrix
from sleekit.scaling import (
    compute_norm_scaling,
    apply_scaling,
    compute_non_saturating_scaling,
    compute_min_mse_scaling,
)


def test_scaling_axis():
    data = torch.tensor(
        [
            [0.0, 10.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    # Axis 0
    sc = compute_norm_scaling(data, 0)
    scaled = apply_scaling(data, sc, 0)
    exp_sc = torch.tensor([10.0 / math.sqrt(2), 5.0])
    expected = torch.tensor([[0.0, math.sqrt(2)], [1.0, 1.0]])
    assert torch.allclose(sc, exp_sc)
    assert torch.allclose(scaled, expected)
    assert torch.allclose(apply_scaling(scaled, 1 / sc, 0), data)
    # Axis 1
    sc = compute_norm_scaling(data, 1)
    scaled = apply_scaling(data, sc, 1)
    exp_sc = torch.tensor([5.0 / math.sqrt(2), math.sqrt(125 / 2)])
    expected = torch.tensor(
        [[0.0, 10.0 / math.sqrt(125 / 2)], [math.sqrt(2), 5.0 / math.sqrt(125 / 2)]]
    )
    assert torch.allclose(sc, exp_sc)
    assert torch.allclose(scaled, expected)
    assert torch.allclose(apply_scaling(scaled, 1 / sc, 1), data)


def test_non_saturating_scaling():
    data = torch.tensor(
        [
            [0.0, 10.0, -20.0, 15.0],
            [5.0, 5.0, 10.0, -10.0],
            [1.0, 2.0, -4.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 10.0, 100.0, 1000.0],
            [-1.0, 10.0, 100.0, 1000.0],
        ],
        dtype=torch.float32,
    )
    cb = Codebook([-1.0, 0.0, 10.0, 20.0])
    sc0 = compute_non_saturating_scaling(data, cb, 0)
    sc1 = compute_non_saturating_scaling(data, cb, 1)
    assert torch.allclose(sc0, torch.tensor([1.0, 0.5, 0.2, 0.5e-18, 50.0, 50.0]))
    assert torch.allclose(sc1, torch.tensor([0.25, 0.5, 5.0, 50.0]))


def test_min_mse_scaling():
    data = torch.randn(20, 50).to(torch.float32)
    cb = Codebook.uniform(9, -2, 2)
    compute_min_mse_scaling(data, cb, 0)


def test_min_mse_scaling_diag_hessian():
    data = torch.randn(20, 50).to(torch.float32)
    cb = Codebook.uniform(9, -2, 2)
    H = torch.rand(50)
    compute_min_mse_scaling(data, cb, 0, H=H)


def test_min_mse_scaling_hessian():
    data = torch.randn(20, 50).to(torch.float32)
    cb = Codebook.uniform(9, -2, 2)
    H = random_psd_matrix(50, 10)
    compute_min_mse_scaling(data, cb, 0, H=H)
