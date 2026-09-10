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


def orthonormalize_basis(basis_flat: np.ndarray) -> np.ndarray:
    """
    Orthonormalize a basis using the Gram-Schmidt process.

    Parameters
    ----------
    basis : ndarray
        Input basis to orthonormalize. Each row is a basis vector.

    Returns
    -------
    ndarray
        Orthonormalized basis.
    """
    basis_flat_ortho, _ = np.linalg.qr(basis_flat)
    basis_flat_ortho -= basis_flat_ortho.mean(axis=0)  # zero mean
    basis_flat_ortho /= basis_flat_ortho.std(axis=0)  # unitary std
    return basis_flat_ortho
