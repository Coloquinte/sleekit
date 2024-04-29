import numpy as np
from sleekit.codebooks import Codebook
from sleekit.codebooks import lloyd_max


def test_quant():
    values = [0.0, 0.5, 1.0]
    cb = Codebook(values)
    assert np.allclose(cb.values, values)
    assert np.allclose(cb.limits, [0.25, 0.75])
    assert list(cb.quantize_index([0.1, 0.6, 0.9, -10, 0.4, 2])) == [0, 1, 2, 0, 1, 2]
    assert np.allclose(
        cb.quantize_value([0.1, 0.6, 0.9, -10, 0.4, 2]), [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    )
    assert len(cb) == 3


def test_uniform():
    values = np.array([1, 2, 4, 5, 7, 8, 10, 11], dtype=np.float32)
    cb = Codebook.uniform(values, 3)
    assert np.allclose(cb.values, [1.0, 6.0, 11.0])


def test_equiprobable():
    values = np.array([1, 2, 4, 5, 7, 8, 10, 11], dtype=np.float32)
    cb = Codebook.equiprobable(values, 4)
    assert np.allclose(cb.values, [1.5, 4.5, 7.5, 10.5])
    assert np.allclose(cb.limits, [3, 6, 9])


def test_lloyd():
    data = np.random.randn(1000)
    cb = lloyd_max(data, 8)
    assert len(cb.values) == 8
    assert len(cb.limits) == 7


def test_lloyd_max():
    data = np.random.randn(1000)
    cb = lloyd_max(data, 8, 0.01)
    assert len(cb.values) == 8
    assert len(cb.limits) == 7


def test_lloyd_rand():
    data = np.random.randn(1000)
    cb = lloyd_max(data, 8, random_init=True)
    assert len(cb.values) == 8
    assert len(cb.limits) == 7
