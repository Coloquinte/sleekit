import numpy as np
from sleekit.codebook import Codebook
from sleekit.codebook import lloyd_max


def test_quant():
    values = [0.0, 0.5, 1.0]
    cb = Codebook(values)
    assert np.allclose(cb.values, values)
    assert np.allclose(cb.thresholds, [0.25, 0.75])
    assert list(cb.quantize_index([0.1, 0.6, 0.9, -10, 0.4, 2])) == [0, 1, 2, 0, 1, 2]
    assert np.allclose(
        cb.quantize_value([0.1, 0.6, 0.9, -10, 0.4, 2]), [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    )
    assert len(cb) == 3


def test_quant_up():
    values = [-1.0, 0.0, 1.0]
    cb = Codebook(values)
    lst = [-10.0, -1.0, -0.6, -0.4, 0.4, 0.6, 1.0, 10.0]
    exp_up = [0, 0, 0, 1, 1, 1, 1, 1]
    exp_down = [-1, -1, -1, -1, -1, 0, 0, 0]
    assert list(cb.quantize_up(lst)) == exp_up
    assert list(cb.quantize_down(lst)) == exp_down


def test_dim_quant():
    cb = Codebook([-1, 0, 1])
    data = np.array([[-2, 1, 0], [0.8, 0.1, -1]])
    expected = np.array([[-1, 1, 0], [1, 0, -1]])
    assert np.allclose(cb.quantize_value(data), expected)


def test_repeated():
    values = [0.0, 0.5, 1.0]
    cb = Codebook(values)
    data = np.random.randn(1000)
    quant = cb.quantize_value(data)
    assert np.all(quant == cb.quantize_value(quant))


def test_uniform():
    values = np.array([1, 2, 4, 5, 7, 8, 10, 11], dtype=np.float32)
    cb = Codebook.uniform(3, values.min(), values.max())
    assert np.allclose(cb.values, [1.0, 6.0, 11.0])


def test_equiprobable():
    values = np.array([1, 2, 4, 5, 7, 8, 10, 11], dtype=np.float32)
    cb = Codebook.equiprobable(values, 4)
    assert np.allclose(cb.values, [1.5, 4.5, 7.5, 10.5])
    assert np.allclose(cb.thresholds, [3, 6, 9])


def test_lloyd():
    data = np.random.randn(1000)
    cb = lloyd_max(data, 8)
    assert len(cb.values) == 8
    assert len(cb.thresholds) == 7


def test_lloyd_max():
    data = np.random.randn(1000)
    cb = lloyd_max(data, 8, 0.01)
    assert len(cb.values) == 8
    assert len(cb.thresholds) == 7


def test_lloyd_rand():
    data = np.random.randn(1000)
    cb = lloyd_max(data, 8, random_init=True)
    assert len(cb.values) == 8
    assert len(cb.thresholds) == 7
