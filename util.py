import numpy as np
import numba as nb

@nb.jit
def rotate_rodrigues(vector: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate a vector around an axis by the specified angle. See wikipedia entry for description.

    Parameters
    ----------
    vector : np.ndarray
        Vector to rotate. If Nx3 element array is specified, each row is considered a separate vector.
    axis : np.ndarray
        Axis of rotation
    angle : np.ndarray
        Angle of rotation

    Returns
    -------
    np.ndarray
        Rotated vector
    """

    if len(angle.shape) == 1:
        angle = angle.reshape((-1, 1))
    elif len(angle.shape) > 2:
        raise ValueError('Invalid shape for input angle.')

    # Calculates the dot product of each row, and stores the result in a column vector
    row_wise_dot_product = np.einsum('ij,ij->i', axis, vector).reshape((-1, 1))

    return vector*np.cos(angle) + np.cross(axis, vector)*np.sin(angle) + (1-np.cos(angle))*axis*row_wise_dot_product


@nb.jit
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

    return np.ndarray([rpt[:, 0]*np.sin(rpt[:, 2])*np.cos(rpt[:, 1]),
                       rpt[:, 0]*np.sin(rpt[:, 2])*np.sin(rpt[:, 1]),
                       rpt[:, 0]*np.cos(rpt[:, 2])])


@nb.jit
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


@nb.jit
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


def n_uc_fill(simulation_size: np.ndarray, lattice_vectors: np.ndarray, kind='project') -> np.ndarray:
    """Given a set of lattice vectors, estimate the number of unit cells needed to fill a box of size simulation_size.

    Parameters
    ----------
    simulation_size : np.ndarray
        Array containing the size of the simulation region, (size_x, size_y, size_z)
    lattice_vectors : np.ndarray
        Array containing three lattice vectors. These define the unit cell of the crystal system.

    """

    # TODO: improve this estimate. This is super crude.
    # Calculate the volume of the unit cell, and divide by the simulation size to get the number of unit cells
    # to create.
    if kind == 'volume':
        uc_volume = lattice_vectors[2, :]@np.cross(lattice_vectors[0, :], lattice_vectors[1, :])
        n_uc = np.prod(simulation_size)/uc_volume
    elif kind == 'project':
        xyz = np.eye(3)
        projections_along_xyz = np.sum(np.abs(xyz@lattice_vectors.T, axis=1))
        n_uc = simulation_size/projections_along_xyz

    return n_uc.astype(int)
