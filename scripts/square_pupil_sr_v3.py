# %%

import numpy as np
import matplotlib.pyplot as plt

from srffwfs.fourier_basis import (
    draw_labels,
    get_labels,
    compute_fourier_basis,
)
from srffwfs.binning import bin_2d

# %%

n_sampling_points = 12
extra_sampling_factor = (
    8  # detector pixel shape is (extra_sampling_factor, extra_sampling_factor)
)

extra_px = (
    extra_sampling_factor // 2
)  # extra_sampling_factor // 2 is for doing the half pixel shifts

n_px = extra_sampling_factor * n_sampling_points + extra_px

labels = get_labels(int(n_sampling_points), remove_piston=True)
labels_sr = get_labels(int(2 * n_sampling_points), remove_piston=True)

labels_2g_1 = [
    label for label in labels_sr if label[1] <= n_sampling_points / 2
]  # keep all labels for now

fig, ax = draw_labels(labels)

fig_sr, ax_sr = draw_labels(labels_sr)

fourier_basis = compute_fourier_basis(
    n_px, labels=labels_sr, remove_piston=True, pupil_mask=None, return_labels=False
)

fourier_basis_flat = fourier_basis.reshape(fourier_basis.shape[0], -1).T

# %%

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

# %%

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

fourier_basis_binned_2_grids_flat_stacked_1 = np.vstack(
    (
        fourier_basis_binned_1_flat,
        fourier_basis_binned_2_flat,
        fourier_basis_binned_1_flat,
        fourier_basis_binned_2_flat,
    )
)

fourier_basis_binned_2_grids_flat_stacked_2 = np.vstack(
    (
        fourier_basis_binned_1_flat,
        fourier_basis_binned_3_flat,
        fourier_basis_binned_1_flat,
        fourier_basis_binned_3_flat,
    )
)

fourier_basis_binned_2_grids_flat_stacked_3 = np.vstack(
    (
        fourier_basis_binned_1_flat,
        fourier_basis_binned_4_flat,
        fourier_basis_binned_1_flat,
        fourier_basis_binned_4_flat,
    )
)

u1, s1, vt1 = np.linalg.svd(fourier_basis_binned_1_flat_stacked, full_matrices=False)
u2, s2, vt2 = np.linalg.svd(fourier_basis_binned_2_flat_stacked, full_matrices=False)
u3, s3, vt3 = np.linalg.svd(fourier_basis_binned_3_flat_stacked, full_matrices=False)
u4, s4, vt4 = np.linalg.svd(fourier_basis_binned_4_flat_stacked, full_matrices=False)
uall, sall, vtall = np.linalg.svd(
    fourier_basis_binned_all_flat_stacked, full_matrices=False
)
u2g_1, s2g_1, vt2g_1 = np.linalg.svd(
    fourier_basis_binned_2_grids_flat_stacked_1, full_matrices=False
)
u2g_2, s2g_2, vt2g_2 = np.linalg.svd(
    fourier_basis_binned_2_grids_flat_stacked_2, full_matrices=False
)
u2g_3, s2g_3, vt2g_3 = np.linalg.svd(
    fourier_basis_binned_2_grids_flat_stacked_3, full_matrices=False
)
u2g_3, s2g_3, vt2g_3 = np.linalg.svd(
    fourier_basis_binned_2_grids_flat_stacked_3, full_matrices=False
)

print(f"cond1: {s1.max() / s1.min()}")
print(f"cond2: {s2.max() / s2.min()}")
print(f"cond3: {s3.max() / s3.min()}")
print(f"cond4: {s4.max() / s4.min()}")
print(f"cond2g_1: {s2g_1.max() / s2g_1.min()}")
print(f"cond2g_2: {s2g_2.max() / s2g_2.min()}")
print(f"cond2g_3: {s2g_3.max() / s2g_3.min()}")
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

# fig.savefig("singular_values_binned_fourier_basis_1_2_3_4.svg", dpi=300)

fig = plt.figure()
plt.semilogy(sall, marker="+")
plt.title("Singular Values - Binned Fourier Basis All")

fig = plt.figure()
plt.semilogy(s2g_1, marker="+")
plt.title("Singular Values - Binned Fourier Basis 2 Grids 1")

fig = plt.figure()
plt.semilogy(s2g_2, marker="+")
plt.title("Singular Values - Binned Fourier Basis 2 Grids 2")

fig = plt.figure()
plt.semilogy(s2g_3, marker="+")
plt.title("Singular Values - Binned Fourier Basis 2 Grids 3")

plt.show()

# %%

fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0, 0].imshow(np.abs(vt1))
axs[0, 0].set_title("V^T - Binned Fourier Basis 1")
axs[0, 1].imshow(np.abs(vt2g_1))
axs[0, 1].set_title("V^T - Binned Fourier Basis 2 Grids 1")
axs[1, 0].imshow(np.abs(vt2g_2))
axs[1, 0].set_title("V^T - Binned Fourier Basis 2 Grids 2")
axs[1, 1].imshow(np.abs(vt2g_3))
axs[1, 1].set_title("V^T - Binned Fourier Basis 2 Grids 3")

plt.figure()
plt.imshow(np.abs(vtall))
plt.title("V^T - Binned Fourier Basis All")

# %%

eigen_modes_2g_1_flat = fourier_basis_flat @ vt2g_1.T

eigen_modes_2g_1 = eigen_modes_2g_1_flat.T.reshape(
    (fourier_basis.shape[0],) + fourier_basis.shape[1:]
)

eigen_mode_index = 0

plt.figure()
plt.imshow(eigen_modes_2g_1[eigen_mode_index])
plt.title(f"Eigenmode {eigen_mode_index} - Binned Fourier Basis 2 Grids 1")
plt.show()

eigen_modes_2g_2_flat = fourier_basis_flat @ vt2g_2.T

eigen_modes_2g_2 = eigen_modes_2g_2_flat.T.reshape(
    (fourier_basis.shape[0],) + fourier_basis.shape[1:]
)

eigen_mode_index = 0

plt.figure()
plt.imshow(eigen_modes_2g_2[eigen_mode_index])
plt.title(f"Eigenmode {eigen_mode_index} - Binned Fourier Basis 2 Grids 2")
plt.show()

eigen_modes_2g_3_flat = fourier_basis_flat @ vt2g_3.T

eigen_modes_2g_3 = eigen_modes_2g_3_flat.T.reshape(
    (fourier_basis.shape[0],) + fourier_basis.shape[1:]
)

eigen_mode_index = 0

plt.figure()
plt.imshow(eigen_modes_2g_3[eigen_mode_index])
plt.title(f"Eigenmode {eigen_mode_index} - Binned Fourier Basis 2 Grids 3")
plt.show()

# %%

n_modes_2g = (n_sampling_points * 2) ** 2 // 2

modes_weight_2g_1 = (np.abs(vt2g_1[:n_modes_2g]) ** 2).sum(axis=0)
first_modes_2g_1 = np.argsort(modes_weight_2g_1)[-n_modes_2g:][::-1]

plt.figure()
plt.plot(modes_weight_2g_1)

print(f"First {n_modes_2g} modes: {first_modes_2g_1}")

first_modes_2g_1_labels = [labels_sr[i] for i in first_modes_2g_1]
draw_labels(labels_sr)
draw_labels(first_modes_2g_1_labels)

# %%

modes_weight_2g_2 = (np.abs(vt2g_2[:n_modes_2g]) ** 2).sum(axis=0)
first_modes_2g_2 = np.argsort(modes_weight_2g_2)[-n_modes_2g:][::-1]

plt.figure()
plt.plot(modes_weight_2g_2)

print(f"First {n_modes_2g} modes: {first_modes_2g_2}")

first_modes_2g_2_labels = [labels_sr[i] for i in first_modes_2g_2]
draw_labels(labels_sr)
draw_labels(first_modes_2g_2_labels)

# %%

modes_weight_2g_3 = (np.abs(vt2g_3[:n_modes_2g]) ** 2).sum(axis=0)
first_modes_2g_3 = np.argsort(modes_weight_2g_3)[-n_modes_2g:][::-1]

plt.figure()
plt.plot(modes_weight_2g_3)

print(f"First {n_modes_2g} modes: {first_modes_2g_3}")

first_modes_2g_3_labels = [labels_sr[i] for i in first_modes_2g_3]
draw_labels(labels_sr)
draw_labels(first_modes_2g_3_labels)

# %%
