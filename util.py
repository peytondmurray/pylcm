import numpy as np
import numba as nb


def rotate_rodrigues(vector: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate a vector around an axis by the specified angle. See wikipedia entry for description.

    Parameters
    ----------
    vector : np.ndarray
        Vector to rotate. If Nx3 element array is specified, each row is considered a separate vector.
    axis : np.ndarray
        Axis of rotation
    angle : np.ndarray or float
        Angle of rotation

    Returns
    -------
    np.ndarray
        Rotated vector
    """

    # If given a 2d input vector, return a 2d vector. If given a 1d input vector, return a 1d vector.
    if len(vector.shape) == 2:
        return _rotate_rodrigues(vector, axis, angle)
    elif len(vector.shape) == 1:
        return _rotate_rodrigues(np.atleast_2d(vector), axis, angle)[0]


def _rotate_rodrigues(vector, axis, angle):
    axis = axis.reshape(3)  # Ensure axis is 1D to make it broadcast correctly across vector, which is 2D

    # Calculates the dot product of each row, and stores the result in a column vector
    row_wise_dot_product = np.sum(vector*axis, axis=1).reshape((-1, 1))
    return vector*np.cos(angle) + np.cross(axis, vector)*np.sin(angle) + (1-np.cos(angle))*axis*row_wise_dot_product


@nb.jit(nopython=True)
def spherical_to_cartesian(rpt: np.ndarray) -> np.ndarray:
    """Transform spherical r, phi, theta coordinates to cartesian x, y, z coordinates.

    Parameters
    ----------
    rpt : np.ndarray
        Input (r, phi, theta) array of shape (N, 3). Phi is the azimuthal angle and theta is the polar angle

    Returns
    -------
    np.ndarray
        [description]
    """

    ret = np.empty_like(rpt)

    ret[:, 0] = rpt[:, 0]*np.sin(rpt[:, 2])*np.cos(rpt[:, 1])   # x
    ret[:, 1] = rpt[:, 0]*np.sin(rpt[:, 2])*np.sin(rpt[:, 1])   # y
    ret[:, 2] = rpt[:, 0]*np.cos(rpt[:, 2])                     # z

    return ret


@nb.jit(nopython=True)
def cartesian_to_spherical(xyz: np.ndarray) -> np.ndarray:
    """Transform cartesian x, y, z coordinates to spherical r, phi, theta coordinates.

    Parameters
    ----------
    xyz : np.ndarray
        Input array of shape (N, 3)

    Returns
    -------
    np.ndarray
        Spherical coordinates (r, phi, theta) of shape (N, 3). Phi is the azimuthal angle and theta is the polar angle
    """

    r = np.sqrt(np.sum(xyz**2, axis=1))
    return r, np.arctan2(xyz[:, 1], xyz[:, 0]), np.arccos(xyz[:, 2]/r)


@nb.jit(nopython=True)
def generate_random_vectors(N: int) -> np.ndarray:
    """Generate randomly oriented unit vectors.

    Parameters
    ----------
    N : int
        Number of vectors to generate

    Returns
    -------
    np.ndarray
        N-length xyz array of randomly oriented vectors
    """

    rpt = np.zeros((N, 3))
    rpt[:, 0] = 1
    rpt[:, 1] = 2*np.pi*np.random.rand((N))
    rpt[:, 2] = np.pi*np.random.rand((N))

    return spherical_to_cartesian(rpt)


def n_uc_fill(vecs: np.ndarray, size: np.ndarray, kind='project') -> np.ndarray:
    """Given a set of lattice vectors, estimate the number of unit cells needed to fill a box of size simulation_size.

    Parameters
    ----------
    vecs : np.ndarray
        Array containing three lattice vectors. These define the unit cell of the crystal system.
    size : np.ndarray
        Array containing the size of the simulation region, (size_x, size_y, size_z)

    Returns
    -------
    np.ndarray
        Number of unit cells needed to fill the simulation box

    """

    # TODO: improve this estimate. This is super crude.
    # Calculate the volume of the unit cell, and divide by the simulation size to get the number of unit cells
    # to create.
    if kind == 'volume':
        uc_volume = vecs[2, :]@np.cross(vecs[0, :], vecs[1, :])
        n_uc = np.prod(size)/uc_volume
    elif kind == 'project':
        xyz = np.eye(3)
        projections_along_xyz = np.sum(np.abs(xyz@vecs.T), axis=1)
        n_uc = size/projections_along_xyz

    return n_uc.astype(int)


def fill_locations(center: np.ndarray, vecs: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Given a set of lattice vectors and a simulation size, find all unit cell locations needed to fill the simulation
    space.

    Parameters
    ----------
    center: np.ndarray
        3 element array containing a vector to the xyz coordinates of (hkl) = (000)
    vecs : np.ndarray
        3x3 element array containing 3 lattice vectors

    Returns
    -------
    np.ndarray
        Array of [i, j, k] 3-tuples corresponding to allowed number of unit cells along each lattice vector needed
        to fill the simulation region.
    """

    S = np.vstack((size, size, size))
    n_uc = np.sum(np.abs(S*vecs), axis=1)   # Project x, y, z along each lattice vector.

    # return n_uc
    return np.array(np.meshgrid(np.arange(n_uc[0]), np.arange(n_uc[1]), np.arange(n_uc[2]))).T.reshape((-1, 3))
