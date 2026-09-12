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

labels = get_labels(int(2 * n_sampling_points), remove_piston=True)
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
    fourier_basis[:, :-extra_px, :-extra_px], extra_sampling_factor
)
fourier_basis_binned_2 = bin_2d(
    fourier_basis[:, :-extra_px, extra_px:], extra_sampling_factor
)
fourier_basis_binned_3 = bin_2d(
    fourier_basis[:, extra_px:, :-extra_px], extra_sampling_factor
)
fourier_basis_binned_4 = bin_2d(
    fourier_basis[:, extra_px:, extra_px:], extra_sampling_factor
)

print(f"fourier_basis_binned_1 shape: {fourier_basis_binned_1.shape}")
print(f"fourier_basis_binned_2 shape: {fourier_basis_binned_2.shape}")
print(f"fourier_basis_binned_3 shape: {fourier_basis_binned_3.shape}")
print(f"fourier_basis_binned_4 shape: {fourier_basis_binned_4.shape}")

fig, ax = plt.subplots(1, 1, constrained_layout=True)
ax.imshow(fourier_basis[mode_index], cmap="gray")
ax.set_title("Original Fourier Mode")

fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0, 0].imshow(fourier_basis_binned_1[mode_index], cmap="gray")
axs[0, 0].set_title("Binned Fourier Mode 1")
axs[0, 1].imshow(fourier_basis_binned_2[mode_index], cmap="gray")
axs[0, 1].set_title("Binned Fourier Mode 2")
axs[1, 0].imshow(fourier_basis_binned_3[mode_index], cmap="gray")
axs[1, 0].set_title("Binned Fourier Mode 3")
axs[1, 1].imshow(fourier_basis_binned_4[mode_index], cmap="gray")
axs[1, 1].set_title("Binned Fourier Mode 4")

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

fourier_basis_binned_1_flat_stacked = np.vstack(
    (
        fourier_basis_binned_1_flat,
        fourier_basis_binned_1_flat,
        fourier_basis_binned_1_flat,
        fourier_basis_binned_1_flat,
    )
)

fourier_basis_binned_2_flat_stacked = np.vstack(
    (
        fourier_basis_binned_2_flat,
        fourier_basis_binned_2_flat,
        fourier_basis_binned_2_flat,
        fourier_basis_binned_2_flat,
    )
)

fourier_basis_binned_3_flat_stacked = np.vstack(
    (
        fourier_basis_binned_3_flat,
        fourier_basis_binned_3_flat,
        fourier_basis_binned_3_flat,
        fourier_basis_binned_3_flat,
    )
)

fourier_basis_binned_4_flat_stacked = np.vstack(
    (
        fourier_basis_binned_4_flat,
        fourier_basis_binned_4_flat,
        fourier_basis_binned_4_flat,
        fourier_basis_binned_4_flat,
    )
)

fourier_basis_binned_all_flat_stacked = np.vstack(
    (
        fourier_basis_binned_1_flat,
        fourier_basis_binned_2_flat,
        fourier_basis_binned_3_flat,
        fourier_basis_binned_4_flat,
    )
)

fourier_basis_binned_2_grids_flat_stacked = np.vstack(
    (
        fourier_basis_binned_1_flat,
        fourier_basis_binned_4_flat,
        fourier_basis_binned_1_flat,
        fourier_basis_binned_4_flat,
    )
)

fourier_basis_binned_all_gram_matrix = (
    fourier_basis_binned_all_flat_stacked.T @ fourier_basis_binned_all_flat_stacked
)

plt.figure()
plt.title("Gram Matrix - all binned Fourier Basis")
plt.imshow(np.abs(fourier_basis_binned_all_gram_matrix))

u1, s1, vt1 = np.linalg.svd(fourier_basis_binned_1_flat_stacked, full_matrices=False)
u2, s2, vt2 = np.linalg.svd(fourier_basis_binned_2_flat_stacked, full_matrices=False)
u3, s3, vt3 = np.linalg.svd(fourier_basis_binned_3_flat_stacked, full_matrices=False)
u4, s4, vt4 = np.linalg.svd(fourier_basis_binned_4_flat_stacked, full_matrices=False)
uall, sall, vtall = np.linalg.svd(
    fourier_basis_binned_all_flat_stacked, full_matrices=False
)
u2g, s2g, vt2g = np.linalg.svd(
    fourier_basis_binned_2_grids_flat_stacked, full_matrices=False
)

print(f"cond1: {s1.max() / s1.min()}")
print(f"cond2: {s2.max() / s2.min()}")
print(f"cond3: {s3.max() / s3.min()}")
print(f"cond4: {s4.max() / s4.min()}")
print(f"cond2g: {s2g.max() / s2g.min()}")
print(f"condall: {sall.max() / sall.min()}")

fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0, 0].semilogy(s1, marker="+")
axs[0, 0].set_title("Singular Values - Binned Fourier Basis 1")
axs[0, 1].semilogy(s2, marker="+")
axs[0, 1].set_title("Singular Values - Binned Fourier Basis 2")
axs[1, 0].semilogy(s3, marker="+")
axs[1, 0].set_title("Singular Values - Binned Fourier Basis 3")
axs[1, 1].semilogy(s4, marker="+")
axs[1, 1].set_title("Singular Values - Binned Fourier Basis 4")

plt.figure()
plt.semilogy(sall, marker="+")
plt.title("Singular Values - Binned Fourier Basis All")

plt.figure()
plt.semilogy(s2g, marker="+")
plt.title("Singular Values - Binned Fourier Basis 2 Grids")

plt.show()

# %%
mode_index = 0

eigenmodes_1 = u1[:, :100].T.reshape(
    100,
    2 * fourier_basis_binned_1.shape[1],
    2 * fourier_basis_binned_1.shape[2],
)

plt.figure()
plt.imshow(
    eigenmodes_1[mode_index],
    cmap="gray",
)
plt.title(f"Left Singular mode {mode_index} - Binned Fourier Basis 1")

fp1 = np.sum(
    np.abs(
        np.fft.fftshift(
            np.fft.fft2(
                eigenmodes_1,
                axes=(1, 2),
            ),
            axes=(1, 2),
        )
    ),
    axis=0,
)

eigenmodes_2g = u2g[:, :399].T.reshape(
    399,
    2 * fourier_basis_binned_1.shape[1],
    2 * fourier_basis_binned_1.shape[2],
)

fp2g = np.sum(
    np.abs(
        np.fft.fftshift(
            np.fft.fft2(
                eigenmodes_2g,
                axes=(1, 2),
            ),
            axes=(1, 2),
        )
    ),
    axis=0,
)

eigenmodes_all = uall.T.reshape(
    399,
    2 * fourier_basis_binned_1.shape[1],
    2 * fourier_basis_binned_1.shape[2],
)

fp_all = np.sum(
    np.abs(np.fft.fftshift(np.fft.fft2(eigenmodes_all, axes=(1, 2)), axes=(1, 2))),
    axis=0,
)

plt.figure()
plt.imshow(np.abs(fp1), cmap="gray")
plt.title(f"FFT of Left Singular mode {mode_index} - Binned Fourier Basis 1")


plt.figure()
plt.imshow(np.abs(fp2g), cmap="gray")
plt.title(f"FFT of Left Singular mode {mode_index} - Binned Fourier Basis 2 Grids")

plt.figure()
plt.imshow(fp_all, cmap="gray")
