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
    vecs = np.array([[1, 0, 0], [-0.5, 0.5*np.sqrt(3), 0], [0, 0, 1]])
    size = np.array([10, 10, 1])
    indices = util.fill_locations(vecs, size)
    locations = util.lattice_from_uvw(vecs, indices)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(locations[:, 0], locations[:, 1], '.k')
        plotting.xy_boundary(ax, size)
        plotting.xy_lattice_vecs(ax, vecs[:2], loc=np.zeros(3))
        plt.show()

    return


def test_generate_uvw(show=False):
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    size = np.array([10, 10, 1])
    uvw = util.generate_uvw(vecs, size)
    xyz = util.lattice_from_uvw(vecs, uvw)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plotting.xy_lattice(ax, xyz)
        plotting.xy_boundary(ax, size)
        plt.show()

    return


if __name__ == '__main__':
    # test_rotate_rodrigues_vector2d()
    # test_rotate_rodrigues_vector1d()
    test_fill_locations(show=True)
    # test_generate_uvw(show=True)
