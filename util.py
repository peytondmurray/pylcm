import numpy as np


# rtmp = vector to be rotated
# rotv = axis of rotation
def rotate_rodrigues(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a vector around an axis by the specified angle.
    
    Parameters
    ----------
    vector : np.ndarray
        Vector to rotate
    axis : np.ndarray
        Axis of rotation
    angle : float
        Angle of rotation
    
    Returns
    -------
    np.ndarray
        Rotated vector
    """

    return vector*np.cos(angle) + np.cross(axis, vector)*np.sin(angle) + (1-np.cos(angle))*axis*(axis@vector)
