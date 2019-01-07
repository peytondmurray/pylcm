import pytest
import util
import numpy as np
import hypothesis


def test_rotate_rodrigues_vector2d():
    vector = np.random.rand(1000, 3)
    vector[:, 2] = 0
    axis = np.array([0, 0, 1])
    result = util.rotate_rodrigues(vector, axis, np.pi/2)
    assert np.all(np.isclose(np.sum(vector*result, axis=1), np.zeros(result.shape[0])))
    return


def test_rotate_rodrigues_vector1d():
    vector = np.random.rand(1000, 3)
    vector[:, 2] = 0
    axis = np.array([0, 0, 1])
    result = util.rotate_rodrigues(vector, axis, np.pi/2)
    assert np.all(np.isclose(np.sum(vector*result), np.zeros(result.shape[0])))
    return


if __name__ == '__main__':
    # test_rotate_rodrigues_vector2d()
    test_rotate_rodrigues_vector1d()