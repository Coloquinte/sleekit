import numpy as np
from sleekit.codebooks import Codebook
from sleekit.codebooks import lloyd


def test_quant():
    values = [0.0, 0.5, 1.0]
    cb = Codebook(values)
    assert np.allclose(cb.values, values)
    assert np.allclose(cb.limits, [0.25, 0.75])
    assert list(cb.quantize_index([0.1, 0.6, 0.9, -10, 0.4, 2])) == [0, 1, 2, 0, 1, 2]
    assert np.allclose(
        cb.quantize_value([0.1, 0.6, 0.9, -10, 0.4, 2]), [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    )


def test_lloyd():
    data = np.random.randn(1000)
    cb = lloyd(data, 8)
    assert len(cb.values) == 8
    assert len(cb.limits) == 7
