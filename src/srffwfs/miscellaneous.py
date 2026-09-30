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


def crop_array(
    arr: np.ndarray,
    new_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Crop an array around the null frequency (center) of an fftshifted FFT grid:
        [-N/2 ... -1 | 0 | 1 ... N/2-1]
    Dimensions agnostic, supports 1D and 2D arrays.

    Parameters
    ----------
    arr : ndarray
        Input array to crop.
    new_shape : tuple[int, ...] | int
        Desired output shape (must have same number of dimensions).
        If int is provided, the same value will be used for all dimensions.

    Returns
    -------
    ndarray
        Center-cropped array.
    """

    if isinstance(new_shape, int):
        new_shape = (new_shape,) * arr.ndim

    if arr.ndim != len(new_shape):
        raise ValueError("new_shape must have the same number of dimensions as arr")

    slices = []
    for size, new_size in zip(arr.shape, new_shape):

        if new_size > size:
            raise ValueError("new_shape must be smaller than arr.shape")

        center = size // 2
        start = center - new_size // 2
        stop = start + new_size

        slices.append(slice(start, stop))

    return arr[tuple(slices)]
