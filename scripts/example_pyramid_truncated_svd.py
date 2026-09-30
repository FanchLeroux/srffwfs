# %% imports

from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt

from srffwfs.config import Config
from srffwfs import (
    oopao_config,
)  # choose between GPU and CPU as well as float precision (float32/float64) for OOPAO

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

from srffwfs.modal_bases.KL_basis import compute_KL_basis
from srffwfs.sensitivity import compute_photon_noise_sensitivity
from srffwfs.compute_control_basis import compute_eigen_control_basis
from srffwfs.closed_loop import close_the_loop

# %% outputs directory

config = Config()
fig_dir = config.root_dir / "outputs"

# %% simulation parameters


# ---------------------- NGS ---------------------- #

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
wavelength = 670e-9  # [m] wavelength of the guide star
optical_band = "R4"  # optical band of the guide star
magnitude = 8  # magnitude of the guide star

# ------------------ ATMOSPHERE ----------------- #

r0 = 0.1  # [m] value of r0 at 500 nm
external_scale = 30  # [m] value of L0 in the visibile
fractional_r0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
wind_speed = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of layers
wind_direction = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
altitude = [0, 1000, 5000, 10000, 12000]  # [m] altitude of layers

# ------------------- TELESCOPE ------------------ #

diameter = 2  # [m] telescope diameter
n_subaperture = 20  # number of WFS subaperture along the telescope diameter
n_pixel_per_subaperture = (
    8  # [pixel] sampling of the WFS subapertures in telescope pupil space
)
resolution = (
    n_subaperture * n_pixel_per_subaperture
)  # resolution of the telescope driven by the WFS
# ------------------------ DM --------------------- #

n_actuator = 2 * n_subaperture  # number of actuators

# ----------------------- WFS ---------------------- #

modulation = 5.0  # [lambda/D] modulation radius or half grey width
n_pix_separation = 10  # [pixel] separation ratio between the pupils
light_threshold = 0.3  # light threshold to select the valid pixels
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

# %% Build objects

# % -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(
    resolution=resolution,  # [pixel] resolution of
    # the telescope
    diameter=diameter,
)  # [m] telescope diameter

# % -----------------------     NGS   ----------------------------------

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

dm = DeformableMirror(tel, nSubap=n_actuator)

# %% ----------------------- Pyramid ---------------------------- #

# pyramid
pyramid_sr = Pyramid(
    nSubap=n_subaperture,
    telescope=tel,
    modulation=modulation,
    lightRatio=light_threshold,
    n_pix_separation=n_pix_separation,
    postProcessing="fullFrame",
)

reference_intensities_2d_no_sr = pyramid_sr.referenceSignal_2D

# super resolved pyramid
pupil_shift_bioedge = [
    [
        sr_amplitude,
        -sr_amplitude,
        -sr_amplitude,
        sr_amplitude,
    ],
    [
        -sr_amplitude,
        -sr_amplitude,
        sr_amplitude,
        sr_amplitude,
    ],
]  # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)
pyramid_sr.apply_shift_wfs(
    pupil_shift_bioedge[0], pupil_shift_bioedge[1], units="pixels"
)  # quadrant numbering: 3, 4, 2, 1 (top left, top right, bottom left, bottom right)
pyramid_sr.modulation = modulation  # update reference intensities etc.
reference_intensities_2d_sr = pyramid_sr.referenceSignal_2D

plt.figure()
plt.imshow(reference_intensities_2d_sr - reference_intensities_2d_no_sr)
plt.title("reference intensities difference\nsuper resolved pyramid - pyramid")

# %% ------------------------- MODAL BASIS -------------------------------

if modal_basis == "KL":
    M2C = compute_KL_basis(tel, atm, dm)
    ngs**tel  # reset

elif modal_basis == "poke":
    M2C = np.identity(dm.nValidAct)

# %% extract calibration basis

influence_functions = dm.modes
calibration_basis = influence_functions @ M2C

# %% -------------------------   Modal  DM   ----------------------------------

first_calibration_modal_dm = DeformableMirror(
    tel, nSubap=n_actuator, modes=calibration_basis
)

# %% calibration

calib_sr = InteractionMatrix(
    ngs,
    tel,
    first_calibration_modal_dm,
    pyramid_sr,
    M2C=np.diag(np.ones(first_calibration_modal_dm.nValidAct)),
    stroke=stroke,
    single_pass=single_pass,
    noise="off",
    display=True,
)

interaction_matrix = calib_sr.D

# %% sensitivity analysis - allows low order mode cutoff identification

interaction_matrix_rad_normalized = interaction_matrix * wavelength / (2 * np.pi)
reference_intensities = pyramid_sr.referenceSignal

photon_noise_sensitivity = compute_photon_noise_sensitivity(
    interaction_matrix_rad_normalized, reference_intensities
)

fig_sensitivity, ax_sensitivity = plt.subplots()
ax_sensitivity.plot(photon_noise_sensitivity)
ax_sensitivity.axhline(y=2**0.5 / 2, color="k", linestyle="--", label=r"$\sqrt{2}/2$")
ax_sensitivity.axhline(y=1, color="k", linestyle=":", label=r"$1$")
ax_sensitivity.set_xlabel("# mode")
ax_sensitivity.set_ylabel(r"S_{ph}")
ax_sensitivity.set_xscale("log")
ax_sensitivity.set_yscale("log")
ax_sensitivity.legend(loc="lower left")
ax_sensitivity.set_title("first calibration basis sensitivity analysis")

# %% compute controll basis using SVD eigenmodes while keeping low order modes

n_lo_modes_to_keep = 100

full_eigen_control_basis, s_eigen_control_basis = compute_eigen_control_basis(
    calibration_basis, interaction_matrix, n_lo_modes_to_keep
)

plt.figure()
plt.plot(s_eigen_control_basis)
plt.title("Singular values of the eigen control basis")
plt.xlabel("# mode")
plt.ylabel("Singular value")
plt.yscale("log")

# %%

n_controlled_modes = 800
eigen_control_basis = full_eigen_control_basis[:, :n_controlled_modes]

# %% Modal dm eigen basis

eigen_modal_dm = DeformableMirror(tel, nSubap=n_actuator, modes=eigen_control_basis)

# %% calibration with eigen control basis

calib_sr_eigen_basis = InteractionMatrix(
    ngs,
    tel,
    eigen_modal_dm,
    pyramid_sr,
    M2C=np.diag(np.ones(eigen_modal_dm.nValidAct)),
    stroke=stroke,
    single_pass=single_pass,
    noise="off",
    display=True,
)

interaction_matrix_eigen_basis = calib_sr_eigen_basis.D


# %% sensitivity analysis - eigen control basis

interaction_matrix_eigen_basis_rad_normalized = (
    interaction_matrix_eigen_basis * wavelength / (2 * np.pi)
)
reference_intensities = pyramid_sr.referenceSignal

photon_noise_sensitivity_eigen_basis = compute_photon_noise_sensitivity(
    interaction_matrix_eigen_basis_rad_normalized, reference_intensities
)

fig_sensitivity, ax_sensitivity = plt.subplots()
ax_sensitivity.plot(photon_noise_sensitivity_eigen_basis)
ax_sensitivity.axhline(y=2**0.5 / 2, color="k", linestyle="--", label=r"$\sqrt{2}/2$")
ax_sensitivity.axhline(y=1, color="k", linestyle=":", label=r"$1$")
ax_sensitivity.set_xlabel("# mode")
ax_sensitivity.set_ylabel(r"S_{ph}")
ax_sensitivity.set_xscale("log")
ax_sensitivity.set_yscale("log")
ax_sensitivity.legend(loc="lower left")
ax_sensitivity.set_title("eigen control basis sensitivity analysis")

# %% LSE Reconstructor computation - eigen control basis

reconstructor_lse_sr = np.linalg.pinv(interaction_matrix_eigen_basis)

# %% SEED

seed = 12  # seed for atmosphere computation

# %% Close the loop - LSE - SR

(
    total_lse_sr,
    residual_lse_sr,
    strehl_lse_sr,
    dm_coefs_lse_sr,
    turbulence_phase_screens_lse_sr,
    residual_phase_screens_lse_sr,
    wfs_frames_lse_sr,
    wfs_signals_lse_sr,
    short_exposure_psf_lse_sr,
) = close_the_loop(
    tel,
    ngs,
    atm,
    eigen_modal_dm,
    pyramid_sr,
    reconstructor_lse_sr,
    loop_gain,
    n_iter,
    delay=delay,
    photon_noise=detector_photon_noise,
    read_out_noise=detector_read_out_noise,
    seed=seed,
    save_telemetry=True,
    save_psf=True,
)

# %% post processing

long_exposure_psf_lse_sr = np.sum(short_exposure_psf_lse_sr[:, :, 100:], axis=2)

# %% plots

# noise propagation
plt.figure()
plt.plot(np.diag(reconstructor_lse_sr @ reconstructor_lse_sr.T) / pyramid_sr.nSignal)
plt.yscale("log")
plt.title("modal uniform noise propagation")
plt.xlabel("# modes")
plt.savefig(fig_dir / "noise_propagation.png", bbox_inches="tight")

# %%

# residuals
plt.figure()
plt.plot(total_lse_sr, label="total_lse_sr")
plt.plot(residual_lse_sr, label="residual_lse_sr")
plt.xlabel("loop iteration")
plt.ylabel("residual phase RMS [nm]")
plt.title("Closed Loop residuals")
plt.legend()
plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")

# %%

# strehls
plt.figure()
plt.plot(strehl_lse_sr, label="strehl_lse_sr")
plt.ylabel("strehl phase RMS [nm]")
plt.title("Closed Loop strehls")
plt.legend()
plt.savefig(fig_dir / "strehls.png", bbox_inches="tight")

# %%

# long exposure PSF
plt.figure()
plt.imshow(np.log(long_exposure_psf_lse_sr), norm="linear", cmap="inferno")
plt.title(f"long_exposure_psf_lse_sr\nPyramid - {n_controlled_modes} controlled modes")
plt.savefig(fig_dir / "long_exposure_psf.png", bbox_inches="tight")

plt.show()

# %%
