import numpy as np

import numpy as np


def bin_2d(array, bin_factor):
    """Average non-overlapping square blocks of a 2D or 3D array.

    For a 2D array of shape (ny, nx), returns (ny_b, nx_b).
    For a 3D array of shape (nmod, ny, nx), returns (nmod, ny_b, nx_b).
    """
    if array.ndim == 2:
        ny, nx = array.shape

        ny_b = ny // bin_factor
        nx_b = nx // bin_factor

        trimmed = array[
            : ny_b * bin_factor,
            : nx_b * bin_factor,
        ]

        return trimmed.reshape(
            ny_b,
            bin_factor,
            nx_b,
            bin_factor,
        ).mean(axis=(1, 3))

    if array.ndim == 3:
        nmod, ny, nx = array.shape

        ny_b = ny // bin_factor
        nx_b = nx // bin_factor

        trimmed = array[
            :,
            : ny_b * bin_factor,
            : nx_b * bin_factor,
        ]

        return trimmed.reshape(
            nmod,
            ny_b,
            bin_factor,
            nx_b,
            bin_factor,
        ).mean(axis=(2, 4))

    raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}")
