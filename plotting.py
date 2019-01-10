import matplotlib
import numpy as np


# TODO Arrows drawn at an angle are distorted for some reason. WHYYYY MPL
def xy_lattice_vecs(ax: matplotlib.axes.Axes, vecs: np.ndarray, loc: np.ndarray=None):
    """Draw a set of lattice vectors as arrows on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the vectors on
    vecs : np.ndarray
        Nx2 element array containing the lattice vectors to draw. Columns in vec[:, 2:] are ignored.
    loc : np.ndarray, optional
        Origin of the vectors to be drawn. Defaults to (x, y) = (0, 0).

    """

    for vec in vecs:
        ax.arrow(loc[0], loc[1], vec[0], vec[1], color='k', alpha=0.6, width=0.05, length_includes_head=True)
    return


def xy_boundary(ax: matplotlib.axes.Axes, size: np.ndarray):
    """Draw the boundary of a simulation region in the xy plane

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    size : np.ndarray
        2 (or 3+)-element array containing the size of the simulation region in each direction. Z-direction is ignored.

    """

    ax.plot([0, 0, size[0], size[0], 0], [0, size[1], size[1], 0, 0], '-k')
    return


def xy_lattice(ax: matplotlib.axes.Axes, xyz: np.ndarray):
    """Plot the location of all the lattice sites in the xy plane

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    xyz : np.ndarray
        Nx3 element array containing the (x, y, z) locations of lattice sites

    """

    ax.plot(xyz[:, 0], xyz[:, 1], '.k')

    return
