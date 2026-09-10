import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from srffwfs.fourier_basis import get_labels, compute_fourier_basis
from srffwfs.miscellaneous import pad_array

n_px = 20
dimension = 10
n_mode = 10

labels = get_labels(dimension, remove_piston=True)

fourier_basis = compute_fourier_basis(
    n_px, labels=labels, remove_piston=True, return_labels=False
)

fourier_basis_flat = fourier_basis.reshape(fourier_basis.shape[0], -1)

stroke = 0.1  # [rad RMS]

psf = (
    np.abs(
        np.fft.fftshift(
            np.fft.fft2(pad_array(np.exp(1j * stroke * fourier_basis[n_mode]), 2))
        )
    )
    ** 2
)

print(f"fourier basis shape: {fourier_basis.shape}")
for label in labels:
    print(label)

# visualize a given Fourier mode
plt.figure()
plt.imshow(fourier_basis[n_mode])
plt.title(f"Fourier mode: {labels[n_mode]}")

# check orthonormality
plt.figure()
plt.imshow(fourier_basis_flat @ fourier_basis_flat.T)
plt.title("Orthonormality check")

# Check unitary std
plt.figure()
plt.plot(fourier_basis_flat.std(axis=1))
plt.title("Standard deviation of each mode")
plt.xlabel("Mode index")
plt.ylabel("Standard deviation")

# Visualize in focal plane
plt.figure()
plt.imshow(
    psf,
    norm=LogNorm(),
    cmap="inferno",
)
plt.title(f"Fourier transform of mode: {labels[n_mode]}")
plt.show()
