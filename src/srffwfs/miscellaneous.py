import numpy as np


def pad_array(array: np.ndarray, factor: int) -> np.ndarray:
    """
    Pad an array with zeros on all sides by a given factor.
    Dimensions agnostic, supports 1D and 2D arrays.

    Parameters
    ----------
    array : ndarray
        Input array to pad.
    factor : int
        Factor by which to pad the array.
    Returns
    -------
    ndarray
        Padded array.
    """
    pad = [((factor - 1) * s // 2,) * 2 for s in array.shape]
    return np.pad(array, pad)
