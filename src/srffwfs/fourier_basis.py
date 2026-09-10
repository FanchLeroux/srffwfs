import numpy as np


def compute_fourier_mode(
    n_pixels: int, nu_x: int, nu_y: int, kind="cos", pupil_mask=None
):
    coords = np.arange(-n_pixels // 2, n_pixels // 2)
    X, Y = np.meshgrid(coords, coords)

    phase = 2 * np.pi * (nu_x * X + nu_y * Y) / n_pixels

    if kind == "sin":
        mode = np.sin(phase)
    elif kind == "cos":
        mode = np.cos(phase)
    else:
        raise ValueError("kind must be 'sin' or 'cos'")

    if pupil_mask is None:
        pupil_mask = np.ones(mode.shape, dtype=bool)

    mode *= pupil_mask

    mode[pupil_mask] -= mode[pupil_mask].mean()
    mode[pupil_mask] /= mode[pupil_mask].std()

    return mode


def compute_fourier_basis(n_pixels: int, remove_piston: bool = False):
    basis = []
    labels = []

    half = n_pixels // 2
    nyquist = n_pixels % 2 == 0

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
                        piston = np.ones((n_pixels, n_pixels))
                        piston /= np.sqrt(np.mean(piston**2))
                        basis.append(piston)
                        labels.append((0, 0, "piston"))
                else:
                    cos_mode = compute_fourier_mode(n_pixels, nu_x, nu_y, "cos")
                    basis.append(cos_mode)
                    labels.append((nu_x, nu_y, "cos"))

                continue

            # -------------------
            # General case
            # -------------------
            if (nu_y != 0 or nu_x >= 0) and (nu_y != half or nu_x >= 0):

                cos_mode = compute_fourier_mode(n_pixels, nu_x, nu_y, "cos")
                sin_mode = compute_fourier_mode(n_pixels, nu_x, nu_y, "sin")

                basis.append(cos_mode)
                labels.append((nu_x, nu_y, "cos"))

                basis.append(sin_mode)
                labels.append((nu_x, nu_y, "sin"))

    basis = np.array(basis)

    expected = n_pixels**2 if not remove_piston else n_pixels**2 - 1

    if basis.shape[0] != expected:
        raise RuntimeError(
            f"Basis incomplete: got {basis.shape[0]}, expected {expected}"
        )

    return basis, labels
