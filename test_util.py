import pytest
import util
import numpy as np
import hypothesis
import matplotlib.pyplot as plt
import plotting


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


def test_fill_locations(show=False):
    lattice_vectors = np.array([[1, 0, 0], [-0.5, 0.5*np.sqrt(3), 0], [0, 0, 1]])
    size = np.array([10, 10, 1])
    indices = util.fill_locations(None, lattice_vectors, size)
    locations = (np.outer(indices[:, 0], lattice_vectors[0])
                 + np.outer(indices[:, 1], lattice_vectors[1])
                 + np.outer(indices[:, 2], lattice_vectors[2]))

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(locations[:, 0], locations[:, 1], '.k')
        ax.plot([0, 0, size[0], size[0], 0], [0, size[1], size[1], 0, 0], '-k')
        plotting.xy_lattice_vecs(ax, lattice_vectors[:2], loc=np.zeros(3))
        plt.show()

    return

if __name__ == '__main__':
    # test_rotate_rodrigues_vector2d()
    # test_rotate_rodrigues_vector1d()
    test_fill_locations(show=True)
