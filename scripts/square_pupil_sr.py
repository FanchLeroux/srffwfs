# %%

import numpy as np
import matplotlib.pyplot as plt

from srffwfs.fourier_basis import get_labels, compute_fourier_basis
from srffwfs.binning import bin_2d

# %%

n_sampling_points = 10
extra_sampling_factor = (
    8  # detector pixel shape is (extra_sampling_factor, extra_sampling_factor)
)

extra_px = (
    extra_sampling_factor // 2
)  # extra_sampling_factor // 2 is for doing the half pixel shifts

n_px = extra_sampling_factor * n_sampling_points + extra_px

labels = get_labels(2 * n_sampling_points, remove_piston=True)
fourier_basis = compute_fourier_basis(
    n_px, labels=labels, remove_piston=True, pupil_mask=None, return_labels=False
)

fourier_basis_flat = fourier_basis.reshape(fourier_basis.shape[0], -1).T

print(f"fourier_basis shape: {fourier_basis.shape}")
print(f"fourier_basis std: {fourier_basis.std(axis=(1,2))}")
print(f"fourier_basis sum std: {fourier_basis.std(axis=(1,2)).sum()}")

print(f"fourier_basis_flat shape: {fourier_basis_flat.shape}")

fourier_basis_gram_matrix = fourier_basis_flat.conj().T @ fourier_basis_flat
plt.figure()
plt.title("Gram Matrix - Fourier Basis")
plt.imshow(np.abs(fourier_basis_gram_matrix))

# %%

mode_index = -1

fourier_basis_binned_1 = bin_2d(
    fourier_basis[:, :-extra_px, :-extra_px], extra_sampling_factor // 2
)
fourier_basis_binned_2 = bin_2d(
    fourier_basis[:, :-extra_px, extra_px:], extra_sampling_factor // 2
)
fourier_basis_binned_3 = bin_2d(
    fourier_basis[:, extra_px:, :-extra_px], extra_sampling_factor // 2
)
fourier_basis_binned_4 = bin_2d(
    fourier_basis[:, extra_px:, extra_px:], extra_sampling_factor // 2
)

print(f"fourier_basis_binned_1 shape: {fourier_basis_binned_1.shape}")
print(f"fourier_basis_binned_2 shape: {fourier_basis_binned_2.shape}")
print(f"fourier_basis_binned_3 shape: {fourier_basis_binned_3.shape}")
print(f"fourier_basis_binned_4 shape: {fourier_basis_binned_4.shape}")

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.imshow(fourier_basis[mode_index], cmap="gray")

fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0, 0].imshow(fourier_basis_binned_1[mode_index], cmap="gray")
axs[0, 0].set_title("Binned Fourier Basis Mode")
axs[0, 1].imshow(fourier_basis_binned_2[mode_index], cmap="gray")
axs[0, 1].set_title("Binned Fourier Basis Mode")
axs[1, 0].imshow(fourier_basis_binned_3[mode_index], cmap="gray")
axs[1, 0].set_title("Binned Fourier Basis Mode")
axs[1, 1].imshow(fourier_basis_binned_4[mode_index], cmap="gray")
axs[1, 1].set_title("Binned Fourier Basis Mode")

fourier_basis_binned_1_flat = fourier_basis_binned_1.reshape(
    fourier_basis_binned_1.shape[0], -1
).T
fourier_basis_binned_2_flat = fourier_basis_binned_2.reshape(
    fourier_basis_binned_2.shape[0], -1
).T
fourier_basis_binned_3_flat = fourier_basis_binned_3.reshape(
    fourier_basis_binned_3.shape[0], -1
).T
fourier_basis_binned_4_flat = fourier_basis_binned_4.reshape(
    fourier_basis_binned_4.shape[0], -1
).T

fourier_basis_binned_1_gram_matrix = (
    fourier_basis_binned_1_flat.T @ fourier_basis_binned_1_flat
)
fourier_basis_binned_2_gram_matrix = (
    fourier_basis_binned_2_flat.T @ fourier_basis_binned_2_flat
)
fourier_basis_binned_3_gram_matrix = (
    fourier_basis_binned_3_flat.T @ fourier_basis_binned_3_flat
)
fourier_basis_binned_4_gram_matrix = (
    fourier_basis_binned_4_flat.T @ fourier_basis_binned_4_flat
)

fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0, 0].imshow(np.abs(fourier_basis_binned_1_gram_matrix))
axs[0, 0].set_title("Binned Fourier Basis 1 Gram Matrix")
axs[0, 1].imshow(np.abs(fourier_basis_binned_2_gram_matrix))
axs[0, 1].set_title("Binned Fourier Basis 2 Gram Matrix")
axs[1, 0].imshow(np.abs(fourier_basis_binned_3_gram_matrix))
axs[1, 0].set_title("Binned Fourier Basis 3 Gram Matrix")
axs[1, 1].imshow(np.abs(fourier_basis_binned_4_gram_matrix))
axs[1, 1].set_title("Binned Fourier Basis 4 Gram Matrix")

u1, s1, vt1 = np.linalg.svd(fourier_basis_binned_4_gram_matrix, full_matrices=False)
u2, s2, vt2 = np.linalg.svd(fourier_basis_binned_3_gram_matrix, full_matrices=False)
u3, s3, vt3 = np.linalg.svd(fourier_basis_binned_2_gram_matrix, full_matrices=False)
u4, s4, vt4 = np.linalg.svd(fourier_basis_binned_1_gram_matrix, full_matrices=False)

print(f"cond1: {s1.max() / s1.min()}")
print(f"cond2: {s2.max() / s2.min()}")
print(f"cond3: {s3.max() / s3.min()}")
print(f"cond4: {s4.max() / s4.min()}")

fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0, 0].semilogy(s1, marker="+")
axs[0, 0].set_title("Singular Values - Binned Fourier Basis 1")
axs[0, 1].semilogy(s2, marker="+")
axs[0, 1].set_title("Singular Values - Binned Fourier Basis 2")
axs[1, 0].semilogy(s3, marker="+")
axs[1, 0].set_title("Singular Values - Binned Fourier Basis 3")
axs[1, 1].semilogy(s4, marker="+")
axs[1, 1].set_title("Singular Values - Binned Fourier Basis 4")

plt.show()

# %%
