# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from srffwfs.fourier_basis import get_labels, compute_fourier_basis
from srffwfs.pattern import get_circular_pupil
from srffwfs.miscellaneous import orthonormalize_basis
from srffwfs.miscellaneous import pad_array

# %%

n_px = 20
dimension = 20
n_mode = -1
zp_factor = 4
stroke = 0.3  # [rad RMS]

pupil = get_circular_pupil(n_px)

labels = get_labels(dimension, remove_piston=True)

fourier_basis = compute_fourier_basis(
    n_px, labels=labels, remove_piston=True, pupil_mask=pupil, return_labels=False
)

fourier_basis_flat = fourier_basis[:, pupil].T

# %%

fourier_basis_flat_ortho = orthonormalize_basis(fourier_basis_flat)
print(f"fourier basis shape: {fourier_basis.shape}")
print(f"fourier basis flat shape: {fourier_basis_flat.shape}")
print(f"fourier basis flat ortho shape: {fourier_basis_flat_ortho.shape}")

fourier_basis_ortho = np.full(
    (fourier_basis_flat_ortho.shape[1],) + fourier_basis.shape[1:], 0.0
)
fourier_basis_ortho[:, pupil] = fourier_basis_flat_ortho.T

# %%

psf = (
    np.abs(
        np.fft.fftshift(
            np.fft.fft2(
                pad_array(
                    pupil * np.exp(1j * stroke * fourier_basis_ortho[n_mode]),
                    factor=zp_factor,
                )
            )
        )
    )
    ** 2
)

psf /= psf.max()  # normalize max to 1

# print(f"fourier basis shape: {fourier_basis.shape}")
# for label in labels:
#     print(label)

support = np.full(pupil.shape, np.nan)
support[pupil] = fourier_basis_flat_ortho[:, 200]
plt.figure()
plt.imshow(support)

# show pupil
plt.figure()
plt.imshow(pupil, cmap="gray")
plt.title("Pupil mask")

# visualize a given Fourier mode
plt.figure()
plt.imshow(fourier_basis[n_mode])
plt.title(f"Fourier mode: {labels[n_mode]}")

# check orthonormality
plt.figure()
plt.imshow(fourier_basis_flat_ortho.T @ fourier_basis_flat_ortho)
plt.title("Orthonormality check")

# Check unitary std
plt.figure()
plt.plot(fourier_basis_flat_ortho.std(axis=0))
plt.title("Standard deviation of each mode")
plt.xlabel("Mode index")
plt.ylabel("Standard deviation")

# Visualize in focal plane
plt.figure()
plt.imshow(
    psf,
    norm=LogNorm(
        vmin=1e-3,
        vmax=1,
    ),
    cmap="inferno",
)
plt.title(f"Fourier transform of mode: {labels[n_mode]}")
plt.show()

# %%
