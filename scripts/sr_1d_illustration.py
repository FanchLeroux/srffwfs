# %%

import numpy as np
import matplotlib.pyplot as plt

from srffwfs.config import Config

config = Config()

# %% plotting parameters

fig_dir = config.root_dir / "outputs"
width_single_column = 3.45  # [Inche]

# %%

n_points = 100
support = 2.0
x = np.linspace(-support, support, n_points)

# Fried geometry
fried_sinc_0 = np.abs(np.sinc(x))
fried_sinc_1 = np.abs(np.sinc(x - 1))

# Fried geometry with super resolution
fried_sr_sinc_0 = np.abs(np.sinc(x))
fried_sr_sinc_1 = np.abs(np.sinc(x - 2))

# 2x more pixels than actuators DM
twice_fried_sinc_0 = np.abs(np.sinc(x / 2))
twice_fried_sinc_1 = np.abs(np.sinc((x - 2) / 2))

# 4x more pixels than actuators DM
four_times_fried_sinc_0 = np.abs(np.sinc(x / 4))
four_times_fried_sinc_1 = np.abs(np.sinc((x - 4) / 4))

fig = plt.figure(figsize=(2 * width_single_column, 0.8 * width_single_column))
plt.plot(
    x, fried_sinc_0, label="Fried geometry - pixel sinc", color="red", linestyle="-"
)
plt.plot(x, fried_sinc_1, label="Fried geometry - aliasing", color="red", linestyle=":")
plt.plot(
    x,
    fried_sr_sinc_0,
    label="Fried geometry with super resolution - pixel sinc",
    color="green",
    linestyle="--",
)
plt.plot(
    x,
    fried_sr_sinc_1,
    label="Fried geometry with super resolution - aliasing",
    color="green",
    linestyle=":",
)
plt.plot(
    x,
    twice_fried_sinc_0,
    label="2x more pixels than actuators - pixel sinc",
    color="blue",
    linestyle="-",
)
plt.plot(
    x,
    twice_fried_sinc_1,
    label="2x more pixels than actuators - aliasing",
    color="blue",
    linestyle=":",
)
plt.plot(
    x,
    four_times_fried_sinc_0,
    label="4x more pixels than actuators - pixel sinc",
    color="black",
    linestyle="-",
)
plt.plot(
    x,
    four_times_fried_sinc_1,
    label="4x more pixels than actuators - aliasing",
    color="black",
    linestyle=":",
)
plt.axvline(x=0.5, color="gray", linestyle=":", label="DM Nyquist frequency")
plt.xlim(0, 2)
plt.ylim(0, 1.01)
plt.legend(loc="upper left", bbox_to_anchor=(0.2, -0.25), fontsize=8)
plt.xlabel(r"Spatial frequency [$d^{-1}$]")
plt.ylabel("|MTF|")

fig.savefig(
    fig_dir / "mtf_fried_geometry_vs_2x_vs_4x_more_pixels_than_actuators_mod.svg",
    bbox_inches="tight",
    pad_inches=0.01,
    dpi=300,
)

fig2 = plt.figure(figsize=(2 * width_single_column, 0.8 * width_single_column))
plt.plot(
    x, fried_sinc_0**2, label="Fried geometry - pixel sinc", color="red", linestyle="-"
)
plt.plot(
    x, fried_sinc_1**2, label="Fried geometry - aliasing", color="red", linestyle=":"
)
plt.plot(
    x,
    fried_sr_sinc_0**2,
    label="Fried geometry with super resolution - pixel sinc",
    color="green",
    linestyle="--",
)
plt.plot(
    x,
    fried_sr_sinc_1**2,
    label="Fried geometry with super resolution - aliasing",
    color="green",
    linestyle=":",
)
plt.plot(
    x,
    twice_fried_sinc_0**2,
    label="2x more pixels than actuators - pixel sinc",
    color="blue",
    linestyle="-",
)
plt.plot(
    x,
    twice_fried_sinc_1**2,
    label="2x more pixels than actuators - aliasing",
    color="blue",
    linestyle=":",
)
plt.plot(
    x,
    four_times_fried_sinc_0**2,
    label="4x more pixels than actuators - pixel sinc",
    color="black",
    linestyle="-",
)
plt.plot(
    x,
    four_times_fried_sinc_1**2,
    label="4x more pixels than actuators - aliasing",
    color="black",
    linestyle=":",
)
plt.axvline(x=0.5, color="gray", linestyle=":", label="DM Nyquist frequency")
plt.xlim(0, 2)
plt.ylim(0, 1.01)
plt.legend(loc="upper left", bbox_to_anchor=(0.2, -0.25), fontsize=8)
plt.xlabel(r"Spatial frequency [$d^{-1}$]")
plt.ylabel("|MTF|$^2$")

fig2.savefig(
    fig_dir / "mtf_fried_geometry_vs_2x_vs_4x_more_pixels_than_actuators_mod2.svg",
    bbox_inches="tight",
    pad_inches=0.01,
    dpi=300,
)
