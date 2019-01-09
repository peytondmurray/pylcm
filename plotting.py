import matplotlib
import numpy as np


#TODO Arrows drawn at an angle are distorted for some reason. WHYYYY MPL
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
