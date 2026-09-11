# %%

import numpy as np
import matplotlib.pyplot as plt

from srffwfs.fourier_basis import get_labels, compute_fourier_basis
from srffwfs.binning import bin_2d

# %%

n_sampling_points = 10
extra_sampling_factor = (
    32  # detector pixel shape is (extra_sampling_factor, extra_sampling_factor)
)

extra_px = (
    extra_sampling_factor // 2
)  # extra_sampling_factor // 2 is for doing the half pixel shifts

n_px = extra_sampling_factor * n_sampling_points + extra_px

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

mode_index = 44

fourier_basis_binned_1 = bin_2d(
    fourier_basis[mode_index, extra_px:, :-extra_px], extra_sampling_factor // 2
)

print(f"fourier_basis_binned_1 shape: {fourier_basis_binned_1.shape}")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(np.abs(fourier_basis[mode_index, extra_px:, :-extra_px]), cmap="gray")
axs[0].set_title("Original Fourier Basis Mode")
axs[1].imshow(np.abs(fourier_basis_binned_1), cmap="gray")
axs[1].set_title("Binned Fourier Basis Mode")

# %%
