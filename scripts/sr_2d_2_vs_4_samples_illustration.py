# %%

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from srffwfs.config import Config

config = Config()

# %% plotting parameters

fig_dir = config.root_dir / "outputs"
width_single_column = 3.45  # [Inches]

# %%

n_points = 3
indices = np.linspace(-1, 1, n_points)


def plot_case_2(ax):
    for shifts, color, marker, index in zip(
        [[0, 0], [0.5, 0.5]],
        ["black", "green"],
        ["o", "*"],
        [1, 2],
    ):
        for x in indices:
            for y in indices:
                ax.scatter(
                    x + shifts[0],
                    y + shifts[1],
                    color=color,
                    marker=marker,
                    s=20,
                    label=(f"Sampling grid {index}" if x == 1.0 and y == 1.0 else None),
                )

    for c in np.arange(-2, 2.1, 1):
        ax.axline(
            (c, 0),
            slope=1,
            color="grey",
            alpha=0.8,
            ls=":",
            linewidth=0.7,
        )
        ax.axline(
            (c, 0),
            slope=-1,
            color="grey",
            alpha=0.8,
            ls=":",
            linewidth=0.7,
        )

    ax.set_xlim([-1.1, 1.6])
    ax.set_ylim([-1.1, 1.6])
    ax.set_aspect("equal")
    ax.set_axisbelow(True)


def plot_case_4(ax):
    for shifts, color, marker, index in zip(
        [[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]],
        ["black", "red", "blue", "green"],
        ["o", "x", "+", "*"],
        [1, 2, 3, 4],
    ):
        for x in indices:
            for y in indices:
                ax.scatter(
                    x + shifts[0],
                    y + shifts[1],
                    color=color,
                    marker=marker,
                    s=20,
                    label=(f"Sampling grid {index}" if x == 1.0 and y == 1.0 else None),
                )

    ax.set_xticks(np.arange(-1, 1.6, 0.5))
    ax.set_yticks(np.arange(-1, 1.6, 0.5))
    ax.grid(color="grey", alpha=0.8, ls=":", linewidth=0.7)

    ax.set_xlim([-1.1, 1.6])
    ax.set_ylim([-1.1, 1.6])
    ax.set_aspect("equal")
    ax.set_axisbelow(True)


# %% Individual figures

fig_2_samples, ax_2_samples = plt.subplots(
    figsize=(width_single_column, width_single_column)
)
plot_case_2(ax_2_samples)
ax_2_samples.legend(
    loc="upper left",
    bbox_to_anchor=(0.2, -0.1),
    fontsize=8,
)

fig_4_samples, ax_4_samples = plt.subplots(
    figsize=(width_single_column, width_single_column)
)
plot_case_4(ax_4_samples)
ax_4_samples.legend(
    loc="upper left",
    bbox_to_anchor=(0.2, -0.1),
    fontsize=8,
)

# %% Combined figure

fig_both_cases, (ax_both_2, ax_both_4) = plt.subplots(
    1,
    2,
    figsize=(2 * width_single_column, width_single_column),
)

plot_case_2(ax_both_2)
plot_case_4(ax_both_4)

ax_both_2.legend(
    loc="upper left",
    bbox_to_anchor=(0.2, -0.1),
    fontsize=8,
)

ax_both_4.legend(
    loc="upper left",
    bbox_to_anchor=(0.2, -0.1),
    fontsize=8,
)

fig_both_cases.tight_layout()

fig_both_cases.savefig(
    fig_dir / "2d_2_vs_4_sampling_grids_illustration.svg",
    bbox_inches="tight",
    pad_inches=0.1,
    dpi=300,
)

fig_fourier, ax_fourier = plt.subplots(
    1, 1, figsize=(width_single_column, width_single_column)
)
rect_1_samples = Rectangle(
    (-0.5, -0.5),  # bottom-left corner
    1,
    1,
    facecolor=(0, 0, 0, 0.3),
    edgecolor="black",
    linewidth=2,
    linestyle=":",
    zorder=20,
    label="1 sample grid",
)
rect_2_samples_sx = Rectangle(
    (-1, -0.5),  # bottom-left corner
    2,
    1,
    facecolor=(0.5, 0.0, 0.0, 0.3),
    edgecolor="red",
    angle=0,
    linewidth=2,
    zorder=11,
    label="2 sample grids - (0.5, 0) shift",
    linestyle="-",
)
rect_2_samples_sy = Rectangle(
    (-0.5, -1),  # bottom-left corner
    1,
    2,
    facecolor=(0.0, 0.0, 0.5, 0.3),
    edgecolor="blue",
    angle=0,
    linewidth=2,
    zorder=11,
    label="2 sample grids - (0, 0.5) shift",
    linestyle="-",
)
rect_2_samples_sx_sy = Rectangle(
    (0, -1),  # bottom-left corner
    2**0.5,
    2**0.5,
    facecolor=(0.0, 0.5, 0.0, 0.3),
    edgecolor="green",
    angle=45,
    linewidth=2,
    label="2 sample grids - (0.5, 0.5) shift",
)
rect_4_samples = Rectangle(
    (-1, -1),  # bottom-left corner
    2,
    2,
    facecolor=(0.5, 0.5, 0.5, 0.3),
    edgecolor="grey",
    linewidth=2,
    label="4 sample grids",
    linestyle=":",
    zorder=11,
)

ax_fourier.add_patch(rect_1_samples)
ax_fourier.add_patch(rect_2_samples_sx)
ax_fourier.add_patch(rect_2_samples_sy)
ax_fourier.add_patch(rect_2_samples_sx_sy)
ax_fourier.add_patch(rect_4_samples)
ax_fourier.set_xlim([-1.1, 1.1])
ax_fourier.set_ylim([-1.1, 1.1])
ax_fourier.set_aspect("equal")
ax_fourier.set_xticks(np.arange(-1, 1.1, 0.5))
ax_fourier.set_yticks(np.arange(-1, 1.1, 0.5))

ax_fourier.set_xlabel(r"Spatial frequency [$d^{-1}$]")
ax_fourier.set_ylabel(r"Spatial frequency [$d^{-1}$]")

ax_fourier.legend(
    loc="upper left",
    bbox_to_anchor=(1.0, 0.7),
    fontsize=10,
)

fig_fourier.savefig(
    fig_dir / "sr_2d_2_vs_4_samples_illustration_accessible_frequencies.svg",
    bbox_inches="tight",
    dpi=300,
)

# %% Show

plt.show()

# %%
