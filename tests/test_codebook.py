import torch
from sleekit.codebook import Codebook
from sleekit.codebook import lloyd_max


def test_quant():
    values = torch.tensor([0.0, 0.5, 1.0])
    cb = Codebook(values)
    assert torch.allclose(cb.values, values)
    assert torch.allclose(cb.thresholds, torch.tensor([0.25, 0.75]))
    data = torch.tensor([0.1, 0.6, 0.9, -10, 0.4, 2])
    expected = [0, 1, 2, 0, 1, 2]
    assert list(cb.quantize_index(data)) == expected
    data = torch.tensor([0.1, 0.6, 0.9, -10, 0.4, 2])
    expected = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    assert torch.allclose(cb.quantize_value(data), expected)
    assert len(cb) == 3


def test_dim_quant():
    cb = Codebook([-1, 0, 1])
    data = torch.tensor([[-2, 1, 0], [0.8, 0.1, -1]], dtype=torch.float32)
    expected = torch.tensor([[-1, 1, 0], [1, 0, -1]], dtype=torch.float32)
    assert torch.allclose(cb.quantize_value(data), expected)


def test_repeated():
    values = [0.0, 0.5, 1.0]
    cb = Codebook(values)
    data = torch.randn(1000)
    quant = cb.quantize_value(data)
    assert torch.all(quant == cb.quantize_value(quant))


def test_uniform():
    values = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11], dtype=torch.float32)
    cb = Codebook.uniform(3, values.min(), values.max())
    assert torch.allclose(cb.values, torch.tensor([1.0, 6.0, 11.0]))


def test_equiprobable():
    values = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11], dtype=torch.float32)
    cb = Codebook.equiprobable(values, 4)
    assert torch.allclose(cb.values, torch.tensor([1.5, 4.5, 7.5, 10.5]))
    assert torch.allclose(cb.thresholds, torch.tensor([3.0, 6.0, 9.0]))


def test_lloyd():
    data = torch.randn(1000)
    cb = lloyd_max(data, 8)
    assert len(cb.values) == 8
    assert len(cb.thresholds) == 7


def test_lloyd_max():
    data = torch.randn(1000)
    cb = lloyd_max(data, 8, 0.01)
    assert len(cb.values) == 8
    assert len(cb.thresholds) == 7


def test_lloyd_rand():
    data = torch.randn(1000)
    cb = lloyd_max(data, 8, random_init=True)
    assert len(cb.values) == 8
    assert len(cb.thresholds) == 7
