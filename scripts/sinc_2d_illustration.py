import numpy as np
import matplotlib.pyplot as plt

from srffwfs.config import Config

config = Config()

fig_dir = config.root_dir / "outputs"

# Spatial-frequency coordinates
x = np.linspace(-5, 5, 800)
y = np.linspace(-5, 5, 800)

X, Y = np.meshgrid(x, y)


def sinc2d(x, y):
    return np.sinc(x) * np.sinc(y)


centers = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]

centers_sr = [
    (-2, -2),
    (0, -2),
    (2, -2),
    (-2, 0),
    (0, 0),
    (2, 0),
    (-2, 2),
    (0, 2),
    (2, 2),
]


S = np.zeros_like(X)
S_sr = np.zeros_like(X)

for cx, cy in centers:
    S += np.abs(sinc2d(X - cx, Y - cy)) ** 2

for cx, cy in centers_sr:
    S_sr += np.abs(sinc2d(X - cx, Y - cy)) ** 2

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X,
    Y,
    S,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
)

ax.set_xlabel(r"$f_x/f_c$")
ax.set_ylabel(r"$f_y/f_c$")
ax.set_zlabel(r"$|MTF|^2$")

ax.set_box_aspect((1, 1, 0.2))

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

plt.tight_layout()

fig_sr = plt.figure(figsize=(9, 8))
ax_sr = fig_sr.add_subplot(111, projection="3d")

ax_sr.plot_surface(
    X,
    Y,
    S_sr,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
)

ax_sr.set_xlabel(r"$f_x/f_c$")
ax_sr.set_ylabel(r"$f_y/f_c$")
ax_sr.set_zlabel(r"$|MTF|^2$")

ax_sr.set_box_aspect((1, 1, 0.2))

ax_sr.set_xlim(-5, 5)
ax_sr.set_ylim(-5, 5)

plt.tight_layout()

fig.savefig(fig_dir / "sinc_2d.svg", bbox_inches="tight")
fig_sr.savefig(fig_dir / "sinc_2d_sr.svg", bbox_inches="tight")

plt.show()
