import numpy as np

from sleekit.scaling import scale_norm


def test_scaling_axis():
    # Axis 0
    initial_data = np.array(
        [
            [
                0.0,
                10.0,
            ],
            [5.0, 5.0],
        ]
    )
    data = initial_data.copy()
    sc = scale_norm(data, 0)
    assert np.allclose(data, np.array([[0.0, np.sqrt(2)], [1.0, 1.0]]))
    assert np.allclose(np.reshape(sc, (-1, 1)) * data, initial_data)
    # Axis 1
    data = initial_data.copy()
    sc = scale_norm(data, 1)
    assert np.allclose(
        data,
        np.array(
            [[0.0, 10.0 / np.sqrt(125 / 2)], [np.sqrt(2), 5.0 / np.sqrt(125 / 2)]]
        ),
    )
    assert np.allclose(np.reshape(sc, (1, -1)) * data, initial_data)
