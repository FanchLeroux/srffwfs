import numpy as np

import numpy as np


def bin_2d(array, bin_factor):
    """Average non-overlapping square blocks of a 2D array."""
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
