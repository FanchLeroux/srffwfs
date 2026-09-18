# %%

import numpy as np
import matplotlib.pyplot as plt

from srffwfs.fourier_basis import (
    draw_labels,
    draw_multiple_labels,
    get_labels,
    compute_fourier_basis,
)
from srffwfs.binning import bin_2d
from srffwfs.config import Config

config = Config()

# %%

n_sampling_points = 10
extra_sampling_factor = (
    16  # detector pixel shape is (extra_sampling_factor, extra_sampling_factor)
)

extra_px = extra_sampling_factor // 2  # for half-pixel shifts
n_px = extra_sampling_factor * n_sampling_points + extra_px

labels = get_labels(int(n_sampling_points), remove_piston=True)
labels_sr = get_labels(int(2 * n_sampling_points), remove_piston=True)

fig, ax = draw_labels(labels)
fig_sr, ax_sr = draw_labels(labels_sr)

fourier_basis = compute_fourier_basis(
    n_px,
    labels=labels_sr,
    remove_piston=True,
    pupil_mask=None,
    return_labels=False,
)

fourier_basis_flat = fourier_basis.reshape(fourier_basis.shape[0], -1).T

fig, axs = plt.subplots(2, 2, constrained_layout=True)
axs[0, 0].imshow(fourier_basis[0], cmap="gray")
axs[0, 0].set_title("Fourier Basis Mode 0")
axs[0, 1].imshow(fourier_basis[1], cmap="gray")
axs[0, 1].set_title("Fourier Basis Mode 1")
axs[1, 0].imshow(fourier_basis[20], cmap="gray")
axs[1, 0].set_title("Fourier Basis Mode 20")
axs[1, 1].imshow(fourier_basis[260], cmap="gray")
axs[1, 1].set_title("Fourier Basis Mode 280")

# %%

fourier_basis_binned_g1 = bin_2d(
    fourier_basis[:, :-extra_px, :-extra_px],
    extra_sampling_factor,
)
fourier_basis_binned_g2 = bin_2d(
    fourier_basis[:, :-extra_px, extra_px:],
    extra_sampling_factor,
)
fourier_basis_binned_g3 = bin_2d(
    fourier_basis[:, extra_px:, :-extra_px],
    extra_sampling_factor,
)
fourier_basis_binned_g4 = bin_2d(
    fourier_basis[:, extra_px:, extra_px:],
    extra_sampling_factor,
)

fourier_basis_binned_g1_flat = fourier_basis_binned_g1.reshape(
    fourier_basis_binned_g1.shape[0], -1
).T
fourier_basis_binned_g2_flat = fourier_basis_binned_g2.reshape(
    fourier_basis_binned_g2.shape[0], -1
).T
fourier_basis_binned_g3_flat = fourier_basis_binned_g3.reshape(
    fourier_basis_binned_g3.shape[0], -1
).T
fourier_basis_binned_g4_flat = fourier_basis_binned_g4.reshape(
    fourier_basis_binned_g4.shape[0], -1
).T

# %%

fourier_basis_binned_g1_flat_stacked = np.vstack(
    (
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g1_flat,
    )
)

fourier_basis_binned_g2_flat_stacked = np.vstack(
    (
        fourier_basis_binned_g2_flat,
        fourier_basis_binned_g2_flat,
        fourier_basis_binned_g2_flat,
        fourier_basis_binned_g2_flat,
    )
)

fourier_basis_binned_g3_flat_stacked = np.vstack(
    (
        fourier_basis_binned_g3_flat,
        fourier_basis_binned_g3_flat,
        fourier_basis_binned_g3_flat,
        fourier_basis_binned_g3_flat,
    )
)

fourier_basis_binned_g4_flat_stacked = np.vstack(
    (
        fourier_basis_binned_g4_flat,
        fourier_basis_binned_g4_flat,
        fourier_basis_binned_g4_flat,
        fourier_basis_binned_g4_flat,
    )
)

fourier_basis_binned_all_g_flat_stacked = np.vstack(
    (
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g2_flat,
        fourier_basis_binned_g3_flat,
        fourier_basis_binned_g4_flat,
    )
)

fourier_basis_binned_g1_g2_flat_stacked = np.vstack(
    (
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g2_flat,
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g2_flat,
    )
)

fourier_basis_binned_g1_g3_flat_stacked = np.vstack(
    (
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g3_flat,
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g3_flat,
    )
)

fourier_basis_binned_g1_g4_flat_stacked = np.vstack(
    (
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g4_flat,
        fourier_basis_binned_g1_flat,
        fourier_basis_binned_g4_flat,
    )
)

u_g1, s_g1, vt_g1 = np.linalg.svd(
    fourier_basis_binned_g1_flat_stacked,
    full_matrices=False,
)
u_g2, s_g2, vt_g2 = np.linalg.svd(
    fourier_basis_binned_g2_flat_stacked,
    full_matrices=False,
)
u_g3, s_g3, vt_g3 = np.linalg.svd(
    fourier_basis_binned_g3_flat_stacked,
    full_matrices=False,
)
u_g4, s_g4, vt_g4 = np.linalg.svd(
    fourier_basis_binned_g4_flat_stacked,
    full_matrices=False,
)
u_all_g, s_all_g, vt_all_g = np.linalg.svd(
    fourier_basis_binned_all_g_flat_stacked,
    full_matrices=False,
)
u_g1_g2, s_g1_g2, vt_g1_g2 = np.linalg.svd(
    fourier_basis_binned_g1_g2_flat_stacked,
    full_matrices=False,
)
u_g1_g3, s_g1_g3, vt_g1_g3 = np.linalg.svd(
    fourier_basis_binned_g1_g3_flat_stacked,
    full_matrices=False,
)
u_g1_g4, s_g1_g4, vt_g1_g4 = np.linalg.svd(
    fourier_basis_binned_g1_g4_flat_stacked,
    full_matrices=False,
)

print(f"cond_g1: {s_g1.max() / s_g1.min()}")
print(f"cond_g2: {s_g2.max() / s_g2.min()}")
print(f"cond_g3: {s_g3.max() / s_g3.min()}")
print(f"cond_g4: {s_g4.max() / s_g4.min()}")
print(f"cond_g1_g2: {s_g1_g2.max() / s_g1_g2.min()}")
print(f"cond_g1_g3: {s_g1_g3.max() / s_g1_g3.min()}")
print(f"cond_g1_g4: {s_g1_g4.max() / s_g1_g4.min()}")
print(f"cond_all_g: {s_all_g.max() / s_all_g.min()}")

fig, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0, 0].semilogy(s_g1, marker="+")
axs[0, 0].set_title("Singular Values - Binned Fourier Basis Grid 1")

axs[0, 1].semilogy(s_g2, marker="+")
axs[0, 1].set_title("Singular Values - Binned Fourier Basis Grid 2")

axs[1, 0].semilogy(s_g3, marker="+")
axs[1, 0].set_title("Singular Values - Binned Fourier Basis Grid 3")

axs[1, 1].semilogy(s_g4, marker="+")
axs[1, 1].set_title("Singular Values - Binned Fourier Basis Grid 4")

# fig.savefig("singular_values_binned_fourier_basis_g1_g2_g3_g4.svg", dpi=300)

fig = plt.figure()
plt.semilogy(s_all_g, marker="+")
plt.title("Singular Values - Binned Fourier Basis All Grids")

fig = plt.figure()
plt.semilogy(s_g1_g2, marker="+")
plt.title("Singular Values - Binned Fourier Basis Grids 1 and 2")

fig = plt.figure()
plt.semilogy(s_g1_g3, marker="+")
plt.title("Singular Values - Binned Fourier Basis Grids 1 and 3")

fig = plt.figure()
plt.semilogy(s_g1_g4, marker="+")
plt.title("Singular Values - Binned Fourier Basis Grids 1 and 4")

plt.show()

# %%

fig, axs = plt.subplots(2, 2, constrained_layout=True)

axs[0, 0].imshow(np.abs(vt_g1))
axs[0, 0].set_title("V^T - Binned Fourier Basis Grid 1")

axs[0, 1].imshow(np.abs(vt_g1_g2))
axs[0, 1].set_title("V^T - Binned Fourier Basis Grids 1 and 2")

axs[1, 0].imshow(np.abs(vt_g1_g3))
axs[1, 0].set_title("V^T - Binned Fourier Basis Grids 1 and 3")

axs[1, 1].imshow(np.abs(vt_g1_g4))
axs[1, 1].set_title("V^T - Binned Fourier Basis Grids 1 and 4")

plt.figure()
plt.imshow(np.abs(vt_all_g))
plt.title("V^T - Binned Fourier Basis All Grids")

# %%

n_modes_2g = (n_sampling_points * 2) ** 2 // 2
eigen_mode_index = n_modes_2g - 6

eigen_modes_g1_g2_flat = fourier_basis_flat @ vt_g1_g2.T
eigen_modes_g1_g2 = eigen_modes_g1_g2_flat.T.reshape(
    (fourier_basis.shape[0],) + fourier_basis.shape[1:]
)

eigen_modes_g1_g3_flat = fourier_basis_flat @ vt_g1_g3.T
eigen_modes_g1_g3 = eigen_modes_g1_g3_flat.T.reshape(
    (fourier_basis.shape[0],) + fourier_basis.shape[1:]
)

eigen_modes_g1_g4_flat = fourier_basis_flat @ vt_g1_g4.T
eigen_modes_g1_g4 = eigen_modes_g1_g4_flat.T.reshape(
    (fourier_basis.shape[0],) + fourier_basis.shape[1:]
)

fig, axs = plt.subplots(1, 3, constrained_layout=True)

axs[0].imshow(eigen_modes_g1_g2[eigen_mode_index])
axs[0].set_title(
    f"Eigenmode {eigen_mode_index} - " "Binned Fourier Basis Grids 1 and 2"
)

axs[1].imshow(eigen_modes_g1_g3[eigen_mode_index])
axs[1].set_title(
    f"Eigenmode {eigen_mode_index} - " "Binned Fourier Basis Grids 1 and 3"
)

axs[2].imshow(eigen_modes_g1_g4[eigen_mode_index])
axs[2].set_title(
    f"Eigenmode {eigen_mode_index} - " "Binned Fourier Basis Grids 1 and 4"
)

# %%

modes_weight_g1_g2 = (np.abs(vt_g1_g2[:n_modes_2g]) ** 2).sum(axis=0)
first_modes_g1_g2 = np.argsort(modes_weight_g1_g2)[-n_modes_2g:][::-1]

plt.figure()
plt.plot(modes_weight_g1_g2)
plt.title("Mode Weight - Binned Fourier Basis Grids 1 and 2")

print(f"First {n_modes_2g} modes: {first_modes_g1_g2}")

first_modes_g1_g2_labels = [labels_sr[i] for i in first_modes_g1_g2]

draw_labels(labels_sr)
draw_labels(first_modes_g1_g2_labels)

# %%

modes_weight_g1_g3 = (np.abs(vt_g1_g3[:n_modes_2g]) ** 2).sum(axis=0)
first_modes_g1_g3 = np.argsort(modes_weight_g1_g3)[-n_modes_2g:][::-1]

plt.figure()
plt.plot(modes_weight_g1_g3)
plt.title("Mode Weight - Binned Fourier Basis Grids 1 and 3")

print(f"First {n_modes_2g} modes: {first_modes_g1_g3}")

first_modes_g1_g3_labels = [labels_sr[i] for i in first_modes_g1_g3]

draw_labels(labels_sr)
draw_labels(first_modes_g1_g3_labels)

# %%

modes_weight_g1_g4 = (np.abs(vt_g1_g4[:n_modes_2g]) ** 2).sum(axis=0)
first_modes_g1_g4 = np.argsort(modes_weight_g1_g4)[-n_modes_2g:][::-1]

plt.figure()
plt.plot(modes_weight_g1_g4)
plt.title("Mode Weight - Binned Fourier Basis Grids 1 and 4")

print(f"First {n_modes_2g} modes: {first_modes_g1_g4}")

first_modes_g1_g4_labels = [labels_sr[i] for i in first_modes_g1_g4]

draw_labels(labels_sr)
draw_labels(first_modes_g1_g4_labels)

# %%

fig, ax_all = draw_labels(labels_sr, plot_hermitian=True)
fig, ax_1 = draw_labels(first_modes_g1_g2_labels, plot_hermitian=True)
fig, ax_2 = draw_labels(first_modes_g1_g3_labels, plot_hermitian=True)
fig, ax_3 = draw_labels(first_modes_g1_g4_labels, plot_hermitian=True)

fig, axs = draw_multiple_labels(
    [
        labels_sr,
        first_modes_g1_g2_labels,
        first_modes_g1_g3_labels,
        first_modes_g1_g4_labels,
    ],
    titles=[
        "All modes",
        "First modes\nBinned Fourier Basis Grids 1 and 2",
        "First modes\nBinned Fourier Basis Grids 1 and 3",
        "First modes\nBinned Fourier Basis Grids 1 and 4",
    ],
    plot_hermitian=True,
    ncols=2,
)

fig.savefig(config.root_dir / "outputs" / "dominant_fourier_modes_2_grids.svg")
