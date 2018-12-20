import numpy as np


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
