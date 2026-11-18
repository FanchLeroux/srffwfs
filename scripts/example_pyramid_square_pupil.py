# %% imports

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from srffwfs.config import Config
from srffwfs import (
    oopao_config,
)  # choose between GPU and CPU as well as float precision (float32/float64) for OOPAO

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.Detector import Detector
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

from srffwfs.modal_bases.fourier_basis import compute_fourier_basis

from srffwfs.pattern import get_circular_pupil
from srffwfs.modal_bases.KL_basis import compute_KL_basis
from srffwfs.sensitivity import compute_photon_noise_sensitivity
from srffwfs.closed_loop import close_the_loop, ClosedLoopResult
from srffwfs.miscellaneous import pad_array, crop_array

# %% outputs directory

config = Config()
fig_dir = config.root_dir / "outputs"

# %% simulation parameters

# ---------------------- NGS ---------------------- #

# phot.R = [0.640e-6, 0.150e-6, 4.01e12/cst]
wavelength = 640e-9  # [m] wavelength of the guide star
optical_band = "R"  # optical band of the guide star
magnitude = 0  # magnitude of the guide star

# ------------------ ATMOSPHERE ----------------- #

r0 = 0.1  # [m] value of r0 at 500 nm
external_scale = 30  # [m] value of L0 in the visibile
fractional_r0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
wind_speed = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of layers
wind_direction = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
altitude = [0, 1000, 5000, 10000, 12000]  # [m] altitude of layers

# ------------------- TELESCOPE ------------------ #

diameter = 2  # [m] telescope diameter
n_subaperture = 16  # number of WFS subaperture along the telescope diameter
n_pixel_per_subaperture = (
    16  # [pixel] sampling of the WFS subapertures in telescope pupil space
)
resolution = (
    n_subaperture * n_pixel_per_subaperture
)  # resolution of the telescope driven by the WFS
pupil_oversampling_factor: int = (
    10  # oversampling the pupil then bin it to avoid edge effects
)
# ------------------------ DM ---------------------- #

n_actuator = 2 * n_subaperture  # number of actuators

# ----------------------- WFS ---------------------- #

modulation = 2.0  # [lambda/D] modulation radius or half grey width
n_pix_separation = 2  # [pixel] separation ratio between the pupils
light_threshold = (
    0.3 if modulation > 0.0 else 0
)  # light threshold to select the valid pixels
detector_photon_noise = False
detector_read_out_noise = 0.0  # e- RMS

# super resolution
sr_amplitude = 0.25  # [pixel] super resolution shifts amplitude

# -------------------- CALIBRATION - MODAL BASIS ---------------- #

modal_basis = "KL"
stroke_rad = 0.01  # [rad]
stroke = stroke_rad * wavelength / (2 * np.pi)  # [nm]
single_pass = False  # push-pull or push only for the calibration

# -------------------- LOOP ----------------------- #

loop_gain = 0.7
n_iter = 200
delay = 1
loop_frequency = 1000  # [Hz] loop frequency

# %% Build objects

# % -----------------------    TELESCOPE   -----------------------------

pupil = np.ones((resolution, resolution))  # square pupil

# create the Telescope object
tel = Telescope(
    resolution=resolution,  # [pixel] resolution of the telescope
    diameter=diameter,  # [m] telescope diameter
    samplingTime=1 / loop_frequency,  # [s] sampling time of the telescope
    pupil=pupil,  # telescope pupil
)

# %% -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(
    optBand=optical_band,  # Source optical band
    # (see photometry.py)
    magnitude=magnitude,
)  # Source Magnitude

# % -----------------------    ATMOSPHERE   ----------------------------

# coupling telescope and source is mandatory to generate Atmosphere object
ngs * tel

# create the Atmosphere object
atm = Atmosphere(
    telescope=tel,  # Telescope
    r0=r0,  # Fried Parameter [m]
    L0=external_scale,  # Outer Scale [m]
    # Cn2 Profile (percentage)
    fractionalR0=fractional_r0,
    windSpeed=wind_speed,  # [m.s-1] wind speed of layers
    # [degrees] wind direction
    windDirection=wind_direction,
    # of layers
    altitude=altitude,
)  # [m] altitude of layers

# % -------------------------     DM   ----------------------------------

calibration_basis, labels = compute_fourier_basis(
    n_pixels=tel.resolution,
    n_subap=n_actuator,
    remove_piston=True,
    return_labels=True,
)
# -------------------------   Modal  DM   ----------------------------------

modal_dm = DeformableMirror(
    tel,
    nSubap=n_actuator,
    modes=calibration_basis.reshape(calibration_basis.shape[0], -1).T,
)

# % ----------------------- Science camera ---------------------------- #

science_camera = Detector(
    nRes=tel.resolution, psf_sampling=4, integrationTime=1 / loop_frequency
)

# %% super resolved pyramid - define the pupil shifts to be applied to the WFS

pupil_shifts_zeros = [
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
]

pupil_shifts_horizontal = [
    [
        0.25,
        0.25,
        -0.25,
        -0.25,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
]

pupil_shifts_vertical = [
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.25,
        0.25,
        -0.25,
        -0.25,
    ],
]

pupil_shifts_hv = [
    [
        0.25,
        0.0,
        -0.25,
        0.0,
    ],
    [
        0.0,
        0.25,
        0.0,
        -0.25,
    ],
]

pupil_shifts_quincux = [
    [
        0.25,
        -0.25,
        -0.25,
        0.25,
    ],
    [
        -0.25,
        -0.25,
        0.25,
        0.25,
    ],
]  # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)

pupil_shifts_alternative_quincux = [
    [
        -0.25,
        -0.25,
        0.25,
        0.25,
    ],
    [
        -0.25,
        -0.25,
        0.25,
        0.25,
    ],
]  # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)

# %% ----------------------- Pyramid ---------------------------- #

# pyramid

pyramid_no_sr = Pyramid(
    nSubap=n_subaperture,
    telescope=tel,
    modulation=modulation,
    lightRatio=light_threshold,
    n_pix_separation=n_pix_separation,
    postProcessing="fullFrame",
)

pyramid_sr = Pyramid(
    nSubap=n_subaperture,
    telescope=tel,
    modulation=modulation,
    lightRatio=light_threshold,
    n_pix_separation=n_pix_separation,
    postProcessing="fullFrame",
)

pupil_shifts = pupil_shifts_horizontal

pyramid_sr.apply_shift_wfs(
    pupil_shifts[0],
    pupil_shifts[1],
)
pyramid_sr.modulation = modulation  # update reference intensities etc.

reference_intensities_2d_no_sr = pyramid_no_sr.referenceSignal_2D

# %%

calib_no_sr = InteractionMatrix(
    ngs,
    tel,
    modal_dm,
    pyramid_no_sr,
    M2C=np.diag(np.ones(modal_dm.nValidAct)),
    stroke=stroke,
    single_pass=single_pass,
    noise="off",
    display=True,
    invert=False,
)

# %%

calib_sr = InteractionMatrix(
    ngs,
    tel,
    modal_dm,
    pyramid_sr,
    M2C=np.diag(np.ones(modal_dm.nValidAct)),
    stroke=stroke,
    single_pass=single_pass,
    noise="off",
    display=True,
    invert=False,
)

# %% SVD analysis of the interaction matrix

U, s, Vt = np.linalg.svd(calib_no_sr.D, full_matrices=False)

plt.figure()
plt.semilogy(s)
plt.title("Singular values of the interaction matrix\nNo super resolution")
plt.xlabel("Mode")
plt.ylabel("Singular value")

U, s, Vt = np.linalg.svd(calib_sr.D, full_matrices=False)

plt.figure()
plt.semilogy(s)
plt.title("Singular values of the interaction matrix\nWith super resolution")
plt.xlabel("Mode")
plt.ylabel("Singular value")

# %%

n_mode = 0

plt.figure()
plt.imshow(calibration_basis[n_mode])
plt.title(f"calibration basis mode {n_mode}")

mask_no_sr = pyramid_no_sr.valid_signal_2D.astype(bool)
mask_sr = pyramid_sr.valid_signal_2D.astype(bool)

support = np.full(mask_no_sr.shape, np.nan)
support[mask_no_sr] = calib_no_sr.D[:, n_mode]

plt.figure()
plt.imshow(support)
plt.colorbar()
plt.title(f"support mode {n_mode}")

# %%


def drop_weakest_pupils(values, mask, n_pupils=4, n_drop=2, tol=0.01):
    """
    Set the weakest pupils to zero, independently for each mode.

    Pupils are the connected components of `mask`. For each mode (column of
    `values`), the signal of a pupil is the standard deviation of `values`
    over its pixels. If the relative spread of the pupil signals,
    ``(max - min) / max``, is below `tol`, the pupils are considered
    equivalent and that mode is left unchanged.

    Parameters
    ----------
    values : array_like, shape (mask.sum(), n_modes) or (mask.sum(),)
        Interaction matrix restricted to the valid pixels. Each column is a
        different mode, and rows are ordered in row-major order like the
        pixels of `mask`. A 1D vector is treated as a single mode.
    mask : array_like of bool, shape (ny, nx)
        Boolean image of the valid pixels. Must contain exactly `n_pupils`
        connected components.
    n_pupils : int, optional
        Expected number of pupils in `mask`. Default is 4.
    n_drop : int, optional
        Number of weakest pupils to set to zero, for each mode. Default is 2.
    tol : float, optional
        Relative spread of the pupil signals under which all pupils are
        kept. Default is 0.01.

    Returns
    -------
    out : ndarray, same shape as `values`
        Copy of `values` in which, for each mode, the `n_drop` weakest pupils
        (lowest standard deviation) are set to 0, unless the pupil signals of
        that mode are within `tol`.

    Raises
    ------
    ValueError
        If the number of connected components in `mask` is not `n_pupils`, or
        if the number of rows of `values` does not match ``mask.sum()``.

    Notes
    -----
    The weakest pupils are selected separately for every mode, so different
    modes may have different pupils zeroed.

    Pupils are labelled with `scipy.ndimage.label`, which uses 4-connectivity
    by default. Pupils that touch each other will be merged into a single
    component and trigger the `ValueError`.

    Examples
    --------
    >>> mask = pyramid_sr.valid_signal_2D.astype(bool)
    >>> D_clean = drop_weakest_pupils(calib_no_sr.D, mask)
    >>> D_clean.shape == calib_no_sr.D.shape
    True
    """
    mask = np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(mask)
    if n != n_pupils:
        raise ValueError(f"Expected {n_pupils} pupils, found {n}")

    values = np.asarray(values, dtype=float)
    is_1d = values.ndim == 1
    if is_1d:
        values = values[:, None]
    if values.shape[0] != mask.sum():
        raise ValueError(f"`values` has {values.shape[0]} rows, expected {mask.sum()}")

    pixel_idx = labels[mask] - 1  # pupil index (0-based) of each valid pixel

    # Signal (std) per pupil and per mode, shape (n_pupils, n_modes)
    signal = np.stack([values[pixel_idx == k].std(axis=0) for k in range(n)])

    # Modes whose pupils are not all similar
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = (signal.max(axis=0) - signal.min(axis=0)) / signal.max(axis=0)
    active = spread >= tol  # shape (n_modes,)

    # Pupils to drop, per mode, shape (n_pupils, n_modes)
    drop = np.zeros(signal.shape, dtype=bool)
    weakest = np.argsort(signal, axis=0)[:n_drop]
    np.put_along_axis(drop, weakest, True, axis=0)
    drop &= active

    out = values.copy()
    out[drop[pixel_idx]] = 0  # (n_pix, n_modes) boolean selection

    return out[:, 0] if is_1d else out


imat_modified_no_sr = drop_weakest_pupils(calib_no_sr.D, mask_no_sr)
imat_modified_sr = drop_weakest_pupils(calib_sr.D, mask_sr)

support_modified = np.full(mask_no_sr.shape, np.nan)
support_modified[mask_no_sr] = imat_modified_no_sr[:, n_mode]
plt.figure()
plt.imshow(support_modified)
plt.colorbar()
plt.title(f"support modified mode {n_mode}")

# %% SVD analysis modified interaction matrix

U, s, Vt = np.linalg.svd(imat_modified_no_sr, full_matrices=False)

plt.figure()
plt.semilogy(s)
plt.title("Singular values of the modified interaction matrix")
plt.xlabel("Mode")
plt.ylabel("Singular value")

U, s, Vt = np.linalg.svd(imat_modified_sr, full_matrices=False)

plt.figure()
plt.semilogy(s)
plt.title("Singular values of the modified interaction matrix\nWith super resolution")
plt.xlabel("Mode")
plt.ylabel("Singular value")

# %%


def split_pupils(values, mask, n_pupils=4):
    """
    Split a signal vector into one column per pupil.

    Pupils are the connected components of `mask`. Column ``k`` of the output
    contains only the values of the ``k``-th pupil, in row-major order.

    Parameters
    ----------
    values : array_like, shape (mask.sum(),)
        Signal on the valid pixels, e.g. one column of the interaction
        matrix, ordered in row-major order like the pixels of `mask`.
    mask : array_like of bool, shape (ny, nx)
        Boolean image of the valid pixels. Must contain exactly `n_pupils`
        connected components, all with the same number of pixels.
    n_pupils : int, optional
        Expected number of pupils in `mask`. Default is 4.

    Returns
    -------
    out : ndarray, shape (mask.sum() // n_pupils, n_pupils)
        Column ``k`` holds the values of pupil ``k``. Row ``i`` of every
        column is the ``i``-th pixel of that pupil in row-major order.

    Raises
    ------
    ValueError
        If the number of connected components in `mask` is not `n_pupils`,
        if the pupils do not all have the same number of pixels, or if the
        length of `values` does not match ``mask.sum()``.

    Notes
    -----
    Pupils are labelled with `scipy.ndimage.label` (4-connectivity), so they
    are numbered in the order they are first met in row-major scan.

    If the pupils are translated copies of the same shape, row ``i`` of
    `out` corresponds to the same relative position in every pupil.

    Examples
    --------
    >>> mask = pyramid_sr.valid_signal_2D.astype(bool)
    >>> pupils = split_pupils(calib_no_sr.D[:, 140], mask)
    >>> pupils.shape
    (256, 4)
    """
    mask = np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(mask)
    if n != n_pupils:
        raise ValueError(f"Expected {n_pupils} pupils, found {n}")

    values = np.asarray(values)
    if values.shape != (mask.sum(),):
        raise ValueError(f"`values` has shape {values.shape}, expected ({mask.sum()},)")

    pixel_idx = labels[mask] - 1  # pupil index (0-based) of each valid pixel

    counts = np.bincount(pixel_idx, minlength=n)
    if not np.all(counts == counts[0]):
        raise ValueError(f"Pupils have different pixel counts: {counts}")

    # A stable sort groups pixels by pupil while keeping row-major order
    order = np.argsort(pixel_idx, kind="stable")
    return values[order].reshape(n, counts[0]).T


test = split_pupils(calib_no_sr.D[:, 140], mask_no_sr)
test_modified = split_pupils(imat_modified_no_sr[:, 140], mask_no_sr)

# %%


def free_family_check(pupils, tol=None):
    """
    Check whether the columns of `pupils` are linearly independent.

    Columns are normalized to unit norm first, so the result does not
    depend on the amplitude of each pupil.

    Parameters
    ----------
    pupils : ndarray, shape (n_pix, n_pupils)
    tol : float, optional
        Relative threshold on singular values. Default is the NumPy
        matrix-rank default, ``max(shape) * eps``.

    Returns
    -------
    rank : int
        Numerical rank (equals n_pupils if the family is free).
    s : ndarray, shape (n_pupils,)
        Singular values of the column-normalized matrix, in descending
        order. Their squares sum to n_pupils.
    cond : float
        s.max() / s.min(). Close to 1 = well separated, huge = nearly
        dependent.
    """
    norms = np.linalg.norm(pupils, axis=0)
    A = pupils / norms  # fails (NaN) if a column is zero
    s = np.linalg.svd(A, compute_uv=False)
    rank = np.linalg.matrix_rank(A, tol=None if tol is None else tol * s[0])
    return rank, s, s[0] / s[-1]


# %% no SR

pupils = split_pupils(calib_no_sr.D[:, 0], mask_no_sr)
rank, s, cond = free_family_check(pupils)
print(rank, s, cond)  # rank == 4 -> free family

# %% SR

pupils = split_pupils(calib_sr.D[:, 100], mask_sr)
rank, s, cond = free_family_check(pupils)
print(rank, s, cond)  # rank == 4 -> free family

# %% super resolved pyramid - define the pupil shifts to be applied to the WFS

pupil_shifts_zeros = [
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
]

pupil_shifts_horizontal = [
    [
        0.25,
        0.25,
        -0.25,
        -0.25,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
]

pupil_shifts_vertical = [
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.25,
        0.25,
        -0.25,
        -0.25,
    ],
]

pupil_shifts_hv = [
    [
        0.25,
        0.0,
        -0.25,
        0.0,
    ],
    [
        0.0,
        0.25,
        0.0,
        -0.25,
    ],
]

pupil_shifts_quincux = [
    [
        0.25,
        -0.25,
        -0.25,
        0.25,
    ],
    [
        -0.25,
        -0.25,
        0.25,
        0.25,
    ],
]  # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)

pupil_shifts_alternative_quincux = [
    [
        -0.25,
        -0.25,
        0.25,
        0.25,
    ],
    [
        -0.25,
        -0.25,
        0.25,
        0.25,
    ],
]  # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)

# %%

pupil_shifts_dict = {
    "zeros": pupil_shifts_zeros,
    "horizontal": pupil_shifts_horizontal,
    "vertical": pupil_shifts_vertical,
    "hv": pupil_shifts_hv,
    "quincux": pupil_shifts_quincux,
    "alternative quincux": pupil_shifts_alternative_quincux,
}

pupil_shifts_dict = {
    "alternative quincux": pupil_shifts_alternative_quincux,
}

for key, value in pupil_shifts_dict.items():

    pupil_shifts = (
        value  # choose between pupil_shifts_horizontal and pupil_shifts_quincux
    )
    pyramid_sr.apply_shift_wfs(
        pupil_shifts[0], pupil_shifts[1], units="pixels"
    )  # quadrant numbering: 3, 4, 2, 1 (top left, top right, bottom left, bottom right)
    pyramid_sr.modulation = modulation  # update reference intensities etc.
    reference_intensities_2d_sr = pyramid_sr.referenceSignal_2D

    plt.figure()
    plt.imshow(reference_intensities_2d_sr - reference_intensities_2d_no_sr)
    plt.title(
        f"reference intensities difference\nsuper resolved pyramid - pyramid\n{key} shifts"
    )

    pupil_1 = deepcopy(pyramid_sr.valid_signal_2D)
    pupil_1[pupil_1.shape[0] // 2 :, :] = 0
    pupil_1[:, pupil_1.shape[1] // 2 :] = 0

    pupil_2 = deepcopy(pyramid_sr.valid_signal_2D)
    pupil_2[pupil_2.shape[0] // 2 :, :] = 0
    pupil_2[:, : pupil_2.shape[1] // 2] = 0

    pupil_3 = deepcopy(pyramid_sr.valid_signal_2D)
    pupil_3[: pupil_3.shape[0] // 2, :] = 0
    pupil_3[:, pupil_3.shape[1] // 2 :] = 0

    pupil_4 = deepcopy(pyramid_sr.valid_signal_2D)
    pupil_4[: pupil_4.shape[0] // 2, :] = 0
    pupil_4[:, : pupil_4.shape[1] // 2] = 0

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(pupil_1)
    axs[0, 0].set_title("Pupil 1")
    axs[0, 1].imshow(pupil_2)
    axs[0, 1].set_title("Pupil 2")
    axs[1, 0].imshow(pupil_3)
    axs[1, 0].set_title("Pupil 3")
    axs[1, 1].imshow(pupil_4)
    axs[1, 1].set_title("Pupil 4")

    # calibration

    calib_sr = InteractionMatrix(
        ngs,
        tel,
        modal_dm,
        pyramid_sr,
        M2C=np.diag(np.ones(modal_dm.nValidAct)),
        stroke=stroke,
        single_pass=single_pass,
        noise="off",
        display=True,
    )

    interaction_matrix = calib_sr.D

    print(
        f"Interaction matrix shape: {interaction_matrix.shape}\n"
        f"Interaction matrix rank: {np.linalg.matrix_rank(interaction_matrix)}"
    )

    # choose number of controlled modes

    n_modes = int(
        np.pi * (pyramid_sr.nSubap / 2) ** 2 * 2
    )  # number of controlled modes

    # # compute classic lse reconstructor
    reconstructor_lse = np.linalg.pinv(
        interaction_matrix[:, :n_modes]
    )  # unweighted LSE reconstructor
    reconstructor_lse = np.concatenate(
        (
            reconstructor_lse,
            np.zeros(
                (interaction_matrix.shape[1] - n_modes, reconstructor_lse.shape[1])
            ),
        ),
        axis=0,
    )  # pad the reconstructor with zeros to match the number of WFS signals

    # compute reconstructor with truncated SVD weightened by the phase covariance matrix
    # L = np.linalg.cholesky(c_phi)
    # A = interaction_matrix @ L
    # U, s, Vh = np.linalg.svd(A, full_matrices=False)
    # U_k = U[:, :n_modes]
    # s_k = s[:n_modes]
    # Vh_k = Vh[:n_modes, :]
    # reconstructor_lse = L @ Vh_k.T / s_k @ U_k.T
    # T = L @ Vh_k.T  # (n_cal, n_modes): control modes in original-basis coefficients

    # control_basis = calibration_basis @ T  # control modes in phase space
    modal_projector_ctrl = np.linalg.pinv(
        calibration_basis
    )  # control_basis  # phase -> control-mode coefficients z

    # phase -> coefficients on the ORIGINAL basis, restricted to the controlled subspace
    # modal_projector = T @ modal_projector_ctrl  # (n_cal, n_pix)
    modal_projector = modal_projector_ctrl  # (n_cal, n_pix)
    print("modal projector computed!")

    # SEED

    seed = 12  # seed for atmosphere computation

    # Close the loop - LSE - SR

    closed_loop_results = close_the_loop(
        ngs,
        atm,
        tel,
        modal_dm,
        pyramid_sr,
        reconstructor_lse,
        loop_gain,
        n_iter,
        delay=delay,
        photon_noise=detector_photon_noise,
        read_out_noise=detector_read_out_noise,
        polc=False,
        interaction_matrix=interaction_matrix,
        seed=seed,
        save_telemetry=True,
        save_psf=True,
        science_camera=science_camera,
        modal_projector=modal_projector,
    )

    # post processing

    long_exposure_psf_lse_sr = np.sum(
        closed_loop_results.short_exposure_psf[:, :, 100:], axis=2
    )

    # plots

    # noise propagation
    plt.figure()
    plt.plot(np.diag(reconstructor_lse @ reconstructor_lse.T) / pyramid_sr.nSignal)
    plt.yscale("log")
    plt.title(f"modal uniform noise propagation\n{key} shifts")
    plt.xlabel("# modes")
    plt.savefig(fig_dir / "noise_propagation.png", bbox_inches="tight")

    fitting_error = np.std(
        closed_loop_results.fitting_limited_phase_screens[tel.pupil.flatten(), :],
        axis=0,
    )
    # residuals
    plt.figure()
    plt.plot(closed_loop_results.total, label="total")
    plt.plot(closed_loop_results.residual, label="residual")
    plt.plot(fitting_error, label="fitting error")
    plt.xlabel("loop iteration")
    plt.ylabel("residual phase RMS [nm]")
    plt.title(f"Closed Loop residuals\n{key} shifts")
    plt.legend()
    plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")

    # strehls
    plt.figure()
    plt.plot(closed_loop_results.strehl, label="strehl")
    plt.ylabel("strehl phase RMS [nm]")
    plt.title(f"Closed Loop strehls\n{key} shifts")
    plt.legend()
    plt.savefig(fig_dir / "strehls.png", bbox_inches="tight")

    # long exposure PSF
    plt.figure()
    plt.imshow(
        crop_array(np.log(long_exposure_psf_lse_sr), 100),
        norm="linear",
        cmap="inferno",
    )
    plt.title(
        f"long_exposure_psf_lse_sr\nPyramid - {n_modes} controlled modes\n{key} shifts"
    )
    plt.savefig(fig_dir / f"long_exposure_psf{key}_shifts.png", bbox_inches="tight")

    # ideal coronograph fitting limited

    pupil = tel.pupil.astype(float)
    n_pix, pad = tel.resolution, 2

    opd = closed_loop_results.fitting_limited_phase_screens.copy()  # (n_pix**2, n_iter)
    field = pupil.reshape(-1, 1) * np.exp(1j * 2 * np.pi / ngs.wavelength * opd)
    field = field.reshape(n_pix, n_pix, -1)  # zero outside the pupil

    # ideal coronagraph: remove the pupil mode, E - <E,P>/<P,P> * P
    field -= (
        np.sum(pupil[..., None] * field, axis=(0, 1)) / np.sum(pupil**2)
    ) * pupil[..., None]

    # long-exposure PSF
    psf_le = np.mean(
        np.abs(
            np.fft.fftshift(
                np.fft.fft2(field, s=(pad * n_pix,) * 2, axes=(0, 1)), axes=(0, 1)
            )
        )
        ** 2,
        axis=2,
    )
    psf_le /= (
        np.abs(np.fft.fft2(pupil, s=(pad * n_pix,) * 2)).max() ** 2
    )  # normalise by diffraction-limited peak

    plt.figure()
    plt.imshow(psf_le, norm="linear", cmap="inferno")
    plt.title(f"long exposure psf fitting limited ideal coronagraph\n{key} shifts")

# %%

plt.show()

# %%
