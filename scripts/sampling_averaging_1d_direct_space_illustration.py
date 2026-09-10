import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from srffwfs.config import Config

config = Config()

# Continuous coordinate
x = np.linspace(-5, 5, 2000)

# Arbitrary continuous signal
f = (
    0.8 * np.exp(-((x + 1.5) ** 2) / 1.2)
    - 0.5 * np.exp(-((x - 1.5) ** 2) / 0.8)
    + 0.15 * np.sin(3 * x)
)

# Sampling positions
sampling_period = 1.0
x_samples = np.arange(-5, 5.01, sampling_period)

# Signal values at sampling positions
f_samples = np.interp(x_samples, x, f)

fig, ax = plt.subplots(
    figsize=(config.width_single_column, 0.8 * config.width_single_column)
)

# ------------------------------------------------------------
# Pixels
# ------------------------------------------------------------

pixel_width = sampling_period

for i, (x0, y0) in enumerate(zip(x_samples, f_samples)):

    alpha = 0.30 if i % 2 == 0 else 0.1

    ax.add_patch(
        Rectangle(
            (x0 - pixel_width / 2, 0),
            pixel_width,
            y0,
            facecolor="blue",
            alpha=alpha,
            edgecolor="black",
            linestyle=":",
            linewidth=1,
            zorder=1,
        )
    )

# ------------------------------------------------------------
# Continuous signal
# ------------------------------------------------------------

ax.plot(
    x,
    f,
    color="black",
    linewidth=1.5,
    zorder=3,
    label=r"$f(x)$",
)

# ------------------------------------------------------------
# Diracs
# ------------------------------------------------------------

ax.vlines(
    x_samples,
    0,
    f_samples,
    color="red",
    linewidth=1,
    zorder=4,
)

# Dirac tips
ax.plot(
    x_samples,
    f_samples,
    "o",
    color="red",
    markersize=3,
    zorder=6,
)

# ------------------------------------------------------------
# Horizontal bars representing pixel averaging
# ------------------------------------------------------------

ax.hlines(
    f_samples,
    x_samples - pixel_width / 2,
    x_samples + pixel_width / 2,
    color="blue",
    linewidth=1.5,
    zorder=5,
    label="pixel averaging",
)

# Zero level
ax.axhline(
    0,
    color="black",
    linewidth=0.8,
    zorder=2,
)

ax.set_xlabel(r"$x$ [pixel pitch]")
ax.set_ylabel(r"$\phi(x)$ [a.u.]")

ax.set_xticks(x_samples)
ax.set_xticklabels([f"{x:.0f}" for x in x_samples])

plt.tight_layout()

fig.savefig(
    config.root_dir / "outputs" / "sampling_averaging_1d_direct_space_illustration.svg",
    bbox_inches="tight",
)

plt.show()
