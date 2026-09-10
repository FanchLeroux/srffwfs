import numpy as np


def get_labels(dimension: int, remove_piston: bool = False):
    labels = []

    half = dimension // 2
    nyquist = dimension % 2 == 0

    freq_x = np.arange(-half + 1, half + 1)
    freq_y = np.arange(0, half + 1)

    for nu_x in freq_x:
        for nu_y in freq_y:

            # -------------------
            # Special cases
            # -------------------
            special = (
                (nu_x == 0 and nu_y == 0)
                or (nyquist and nu_x == 0 and nu_y == half)
                or (nyquist and nu_x == half and nu_y == 0)
                or (nyquist and nu_x == half and nu_y == half)
            )

            if special:
                if nu_x == 0 and nu_y == 0:
                    if not remove_piston:
                        labels.append((0, 0, "piston"))
                else:
                    labels.append((nu_x, nu_y, "cos"))

                continue

            # -------------------
            # General case
            # -------------------
            if (nu_y != 0 or nu_x >= 0) and (nu_y != half or nu_x >= 0):

                labels.append((nu_x, nu_y, "cos"))
                labels.append((nu_x, nu_y, "sin"))

    return labels


def compute_fourier_mode(
    n_pixels: int, nu_x: int, nu_y: int, kind="cos", pupil_mask=None
):
    coords = np.arange(-n_pixels // 2, n_pixels // 2)
    X, Y = np.meshgrid(coords, coords)

    phase = 2 * np.pi * (nu_x * X + nu_y * Y) / n_pixels

    if kind == "piston":
        mode = np.ones(phase.shape)
    elif kind == "sin":
        mode = np.sin(phase)
    elif kind == "cos":
        mode = np.cos(phase)
    else:
        raise ValueError("kind must be 'sin' or 'cos' or 'piston'")

    if pupil_mask is None:
        pupil_mask = np.ones(mode.shape, dtype=bool)

    mode *= pupil_mask

    if kind != "piston":
        mode[pupil_mask] -= mode[pupil_mask].mean()
        mode[pupil_mask] /= mode[pupil_mask].std()

    return mode


def compute_fourier_basis(
    n_pixels: int,
    labels: list | None = None,
    remove_piston: bool = False,
    return_labels: bool = False,
):
    basis = []

    if labels is None:
        labels = get_labels(n_pixels, remove_piston=remove_piston)

    for nu_x, nu_y, kind in labels:
        if kind == "piston" and remove_piston:
            continue
        mode = compute_fourier_mode(n_pixels, nu_x, nu_y, kind)
        basis.append(mode)

    basis = np.array(basis)

    if return_labels:
        return basis, labels
    return basis
