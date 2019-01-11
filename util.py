import numpy as np
import numba as nb
import scipy.spatial as ssp
import itertools as it


def pfun(func, *args, **kwargs):

    def f(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)

    return f


@pfun
@nb.jit
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


@pfun
def _rotate_rodrigues(vector, axis, angle):
    axis = axis.reshape(3)  # Ensure axis is 1D to make it broadcast correctly across vector, which is 2D

    # Calculates the dot product of each row, and stores the result in a column vector
    row_wise_dot_product = np.sum(vector*axis, axis=1).reshape((-1, 1))
    return vector*np.cos(angle) + np.cross(axis, vector)*np.sin(angle) + (1-np.cos(angle))*axis*row_wise_dot_product


@pfun
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


@pfun
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


@pfun
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


@pfun
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


@pfun
def fill_locations(vecs: np.ndarray, sim_size: np.ndarray):
    """Given a set of lattice vectors and a simulation size, find all unit cell locations needed to fill the simulation
    space.

    Parameters
    ----------
    vecs : np.ndarray
        3x3 element array containing 3 lattice vectors
    sim_size : np.ndarray
        3 element array containing the size of the simulation box in along the (x, y, z) axes.

    Returns
    -------
    np.ndarray
        Array of [u, v, w] 3-tuples corresponding to allowed number of unit cells along each lattice vector needed
        to fill the simulation region.
    """
    return _fill_locations(vecs, sim_size)


@pfun
def _fill_locations(vecs: np.ndarray, sim_size: np.ndarray, generate_size=None) -> np.ndarray:
    """Given a set of lattice vectors and a simulation size, find all unit cell locations needed to fill the simulation
    space.

    Parameters
    ----------
    vecs : np.ndarray
        3x3 element array containing 3 lattice vectors
    sim_size : np.ndarray
        3 element array containing the size of the simulation box in along the (x, y, z) axes.
    generate_size : np.ndarray
        3 element array containing the size of the region lattice sites are generated in. This can be different
        from the simulation size; for instance, if the lattice vectors do not align with the corners of the box, the
        region where you will need to generate lattice sites is larger, otherwise the corners of the sim box will not be
        filled. Defaults to sim_size.

    Returns
    -------
    np.ndarray
        Array of [u, v, w] 3-tuples corresponding to allowed number of unit cells along each lattice vector needed
        to fill the simulation region.
    """

    sim_center = 0.5*sim_size

    if generate_size is None:
        generate_size = sim_size.copy()

    # One liner finds the vertices of the simulation cube
    corners = np.array(list(it.product(*[[0, s] for s in sim_size])))

    uvw = generate_uvw(vecs, generate_size)                  # Generate lattice sites

    for i in reversed(range(len(corners))):
        is_within_one_uc = within_one_uc(vecs, uvw, corners[i])
        if not is_within_one_uc:
            return _fill_locations(vecs, sim_size, 2*generate_size)

    return uvw


@pfun
def nn_distance(vecs: np.ndarray):

    uvw = generate_uvw(vecs, np.sum(3*vecs, axis=1))    # Generate uvw tuples for a small volume of space
    xyz = lattice_from_uvw(vecs, uvw)                   # Generate lattice points for each uvw tuple

    kdt = ssp.cKDTree(xyz)
    dists, _ = kdt.query(xyz, 2)
    return dists[0, 1]


# @nb.jit(nopython=True)
def within_one_uc(vecs: np.ndarray, uvw: np.ndarray, point: np.ndarray) -> bool:
    """For a given set of lattice vectors and coefficients, check wether any of the lattice sites fall within 1
    unit cell of the given point.

    Parameters
    ----------
    vecs : np.ndarray
        Lattice vectors
    uvw : np.ndarray
        Coefficients of the lattice vectors to check
    point : np.ndarray
        Reference point; distance between each lattice point given by uvw and this point is checked. This is a 3
        element array containing the cartesian coordinates of the point

    Returns
    -------
    bool
        True if one of the lattice sites is within 1 unit cell of the point, False otherwise.
    """

    # Find NN distance for a set of lattice vectors; use that instead

    nn_dist = nn_distance(vecs)

    for i in range(uvw.shape[0]):
        if np.all(np.sqrt(np.sum((lattice_point(vecs, uvw[i]) - point)**2)) < nn_dist*1.5):
            return True

    return False


@pfun
# @nb.jit(nopython=True)
def generate_uvw(vecs: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Generate (u, v, w) coefficient 3-tuples which fill the simulation space given by size, given an input array
    of lattice vectors.

    Parameters
    ----------
    vecs : np.ndarray
        3x3 array of lattice vectors (a, b, c), with xyz cartesian components in the following arrangement:

            ax ay az
            bx by bz
            cx cy cz

    size : np.ndarray
        3 element array containing the sizes along the (x, y, z) directions

    Returns
    -------
    np.ndarray
        Mx3 array of lattice site coefficients. The actual lattice sites can be calculated from the product of these
        coefficients and the lattice vectors; each lattice site is given by

            T = u*a + v*b + w*c
    """

    S = np.vstack((size, size, size))
    n_uc = np.sum(np.abs(S*vecs), axis=1)   # Project x, y, z along each lattice vector.

    return np.array(np.meshgrid(np.arange(n_uc[0]), np.arange(n_uc[1]), np.arange(n_uc[2]))).T.reshape((-1, 3))


@pfun
@nb.jit(nopython=True)
def lattice_from_uvw(vecs: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    """Given a Mx3 set of lattice vectors (a, b, c) and a Nx3 array of integers (u, v, w), generate N (x, y, z)
    lattice sites according to

        u*a + v*b + w*c

    While this seems like something numpy can do in a one liner, I couldn't find a good way to make it readable so
    I'm doing it this way. Probably you can do it with numpy.einsum().

    Parameters
    ----------
    vecs : np.ndarray
        Lattice vectors of shape Mx3
    uvw : np.ndarray
        Array of shape Nx3 containing lattice vector multiplicative factors

    Returns
    -------
    np.ndarray
        Nx3 array of (x, y, z) locations of each lattice site.
    """

    return np.outer(uvw[:, 0], vecs[0]) + np.outer(uvw[:, 1], vecs[1]) + np.outer(uvw[:, 2], vecs[2])


@pfun
@nb.jit(nopython=True)
def lattice_point(vecs: np.ndarray, uvw: np.ndarray) -> np.ndarray:
    """Calculate the lattice position given a set of lattice vectors (a, b, c) and a 3-tuple of coefficients (u, v, w):

        (ax, ay, az) * u
        (bx, by, bz) * v
        (cx, cy, cz) * w

    Parameters
    ----------
    vecs : np.ndarray
        3x3 element array containing the lattice vectors
    uvw : np.ndarray
        3 element array containing the coefficients

    Returns
    -------
    np.ndarray
        3 element array containing the lattice position
    """

    ret = np.empty_like(vecs)
    for i in range(vecs.shape[0]):
        ret[i] = vecs[i]*uvw[i]

    return np.sum(ret, axis=0)
