# %%

import numpy as np
import matplotlib.pyplot as plt
from srffwfs.fourier_basis import get_labels, compute_fourier_basis

# %%

n_sampling_points = 40
extra_sampling_factor = (
    4  # detector pixel is extra_sampling_factor by extra_sampling_factor large
)

n_px = (
    4 * n_sampling_points + extra_sampling_factor // 2
)  # extra_sampling_factor // 2 is for doing the half pixel shifts

labels = get_labels(n_sampling_points, remove_piston=True)
fourier_basis = compute_fourier_basis(
    n_px, labels=labels, remove_piston=True, pupil_mask=None, return_labels=False
)

fourier_basis_flat = fourier_basis.reshape(fourier_basis.shape[0], -1).T

print(f"fourier_basis shape: {fourier_basis.shape}")
print(f"fourier_basis std: {fourier_basis.std(axis=(1,2))}")
print(f"fourier_basis sum std: {fourier_basis.std(axis=(1,2)).sum()}")

print(f"fourier_basis_flat shape: {fourier_basis_flat.shape}")

gram_matrix = fourier_basis_flat.conj().T @ fourier_basis_flat
plt.figure()
plt.title("Gram Matrix")
plt.imshow(np.abs(gram_matrix))


# %%
