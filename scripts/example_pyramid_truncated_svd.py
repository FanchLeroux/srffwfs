# %% imports

from copy import deepcopy

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

from srffwfs.pattern import get_circular_pupil
from srffwfs.modal_bases.KL_basis import compute_KL_basis
from srffwfs.sensitivity import compute_photon_noise_sensitivity
from srffwfs.compute_control_basis import compute_eigen_control_basis
from srffwfs.closed_loop import close_the_loop
from srffwfs.miscellaneous import pad_array, crop_array

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
n_subaperture = 8  # number of WFS subaperture along the telescope diameter
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

n_actuator = n_subaperture  # number of actuators

# ----------------------- WFS ---------------------- #

modulation = 3.0  # [lambda/D] modulation radius or half grey width
n_pix_separation = 10  # [pixel] separation ratio between the pupils
light_threshold = (
    0.3 if modulation > 0.0 else 0
)  # light threshold to select the valid pixels
detector_photon_noise = True
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
pupil_oversampled = get_circular_pupil(tel.resolution * pupil_oversampling_factor)
pupil_binned = pupil_oversampled.reshape(
    tel.resolution,
    pupil_oversampling_factor,
    tel.resolution,
    pupil_oversampling_factor,
).mean(axis=(1, 3))

# original_pupil = deepcopy(tel.pupil)
# tel.pupil = pupil_binned
# tel.pupilReflectivity = pupil_binned  # set the pupil reflectivity to the binned pupil

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(original_pupil)
# axs[0].set_title("original pupil")
# axs[1].imshow(tel.pupilReflectivity)
# axs[1].set_title("new oversampled then binned Pupil")

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

pupil_shifts = pupil_shifts_vertical  # choose between pupil_shifts_horizontal and pupil_shifts_quincux
pyramid_sr.apply_shift_wfs(
    pupil_shifts[0], pupil_shifts[1], units="pixels"
)  # quadrant numbering: 3, 4, 2, 1 (top left, top right, bottom left, bottom right)
pyramid_sr.modulation = modulation  # update reference intensities etc.
reference_intensities_2d_sr = pyramid_sr.referenceSignal_2D

plt.figure()
plt.imshow(reference_intensities_2d_sr - reference_intensities_2d_no_sr)
plt.title("reference intensities difference\nsuper resolved pyramid - pyramid")

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

# %% ------------------------- MODAL BASIS -------------------------------

if modal_basis == "KL":
    m2c, c_phi = compute_KL_basis(tel, atm, dm, return_covariance=True)
    ngs**tel  # reset

elif modal_basis == "poke":
    m2c = np.identity(dm.nValidAct)

# %% extract calibration basis

influence_functions = dm.modes
calibration_basis = influence_functions @ m2c
calibration_basis = tel.pupil.reshape(-1, 1) * calibration_basis  # apply pupil mask

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

print(
    f"Interaction matrix shape: {interaction_matrix.shape}\n"
    f"Interaction matrix rank: {np.linalg.matrix_rank(interaction_matrix)}"
)

# %% Visualize interaction matrix

mode_index = 2  # index of the mode to visualize
support_mode = np.full(tel.pupil.shape, np.nan)
support_mode[tel.pupil] = calibration_basis[tel.pupil.reshape(-1), mode_index]
support_imat = np.full(pyramid_sr.valid_signal_2D.shape, np.nan)
support_imat[pyramid_sr.valid_signal_2D] = interaction_matrix[:, mode_index]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(support_mode, cmap="viridis")
axs[0].set_title(f"Calibration basis - mode {mode_index}")
axs[0].axis("off")
axs[1].imshow(support_imat, cmap="viridis")
axs[1].set_title(f"Interaction matrix signal")
axs[1].axis("off")
plt.colorbar(axs[1].imshow(support_imat, cmap="viridis"), ax=axs[1])

# %% differentiate the diagonal pupils

imat_pupil_1 = np.full(pyramid_sr.valid_signal_2D.shape, 0.0)
imat_pupil_4 = np.full(pyramid_sr.valid_signal_2D.shape, 0.0)
imat_pupil_1[pyramid_sr.valid_signal_2D] = interaction_matrix[:, mode_index]
imat_pupil_4[pyramid_sr.valid_signal_2D] = interaction_matrix[:, mode_index]
imat_pupil_1 = imat_pupil_1[pupil_1]
imat_pupil_4 = imat_pupil_4[pupil_4]

diff_imat = imat_pupil_1 + imat_pupil_4

support = np.full(pyramid_sr.valid_signal_2D.shape, np.nan)
support[pupil_1] = diff_imat

plt.figure()
plt.imshow(support, cmap="viridis")
plt.colorbar()
plt.title("Difference in interaction matrix between pupils 1 and 4")

# %% corelate the pupils

from skimage.registration import phase_cross_correlation

imat_pupil_1_2d = np.full(pyramid_sr.valid_signal_2D.shape, 0.0)
imat_pupil_4_2d = np.full(pyramid_sr.valid_signal_2D.shape, 0.0)
imat_pupil_1_2d[pupil_1] = imat_pupil_1
imat_pupil_4_2d[pupil_1] = imat_pupil_4

shift, error, phasediff = phase_cross_correlation(
    imat_pupil_1_2d,
    -imat_pupil_4_2d,
    upsample_factor=100,
)

print(f"dy = {shift[0]:.3f} px")
print(f"dx = {shift[1]:.3f} px")

# %% sensitivity analysis - allows low/high order mode cutoff identification

n_lo_modes_to_keep = int(np.round(np.pi * modulation**2))
# n_lo_modes_to_keep = 0  # for standard svd

interaction_matrix_rad_normalized = interaction_matrix * wavelength / (2 * np.pi)
reference_intensities = pyramid_sr.referenceSignal

photon_noise_sensitivity = compute_photon_noise_sensitivity(
    interaction_matrix_rad_normalized, reference_intensities
)

fig_sensitivity, ax_sensitivity = plt.subplots()
ax_sensitivity.plot(photon_noise_sensitivity)
ax_sensitivity.axhline(y=2**0.5 / 2, color="k", linestyle=":", label=r"$\sqrt{2}/2$")
ax_sensitivity.axhline(y=1, color="k", linestyle="--", label=r"$1$")
ax_sensitivity.axhline(y=2**0.5, color="k", linestyle="-.", label=r"$\sqrt{2}$")
ax_sensitivity.axhline(y=2, color="k", linestyle="-", label=r"$2$")
ax_sensitivity.axvline(
    n_lo_modes_to_keep,
    color="r",
    linestyle="--",
    label=f"low order modes cutoff: {n_lo_modes_to_keep:.0f}",
)

ax_sensitivity.set_xlabel("# mode")
ax_sensitivity.set_ylabel(r"S_{ph}")
ax_sensitivity.set_xscale("log")
ax_sensitivity.set_yscale("log")
ax_sensitivity.legend(loc="lower left")
ax_sensitivity.set_title("first calibration basis sensitivity analysis")

# %% compute controll basis using SVD eigenmodes while keeping low order modes

full_eigen_control_basis, s_eigen_control_basis = compute_eigen_control_basis(
    calibration_basis, interaction_matrix, n_lo_modes_to_keep
)

# %% plot singular values of the eigen control basis to choose the controlmodal cutoff

n_controlled_modes = int(
    0.25 * pyramid_sr.nSignal
)  # number of controlled modes (modal cutoff)

plt.figure()
plt.plot(s_eigen_control_basis)
plt.axvspan(
    0,
    n_lo_modes_to_keep - 1,
    color="grey",
    alpha=0.3,
    label=f" {n_lo_modes_to_keep} low order modes kept",
)
plt.axvline(
    x=pyramid_sr.nSignal // 4,
    color="r",
    linestyle=":",
    label=f"n_valid_pixels/4",
)
plt.axvline(
    x=pyramid_sr.nSignal // 2,
    color="r",
    linestyle="--",
    label=f"n_valid_pixels/2",
)
plt.axvline(
    x=pyramid_sr.nSignal,
    color="r",
    linestyle="-",
    label=f"n_valid_pixels",
)
plt.axvline(
    x=n_controlled_modes,
    color="k",
    linestyle="--",
    label=f"modal cutoff: {n_controlled_modes} modes",
)
plt.title("Singular values of the eigen control basis")
plt.xlabel("# mode")
plt.ylabel("Singular value")
plt.yscale("log")
plt.legend()

# %%

eigen_control_basis = full_eigen_control_basis[:, :n_controlled_modes]

# %% accessible fourier plane illustration attempt

zero_padding_factor = 2

pupil_padded = pad_array(tel.pupil, zero_padding_factor)
pupil_plane_field = np.zeros(
    zero_padding_factor * np.array(tel.pupil.shape), dtype=complex
)
focal_plane_irradiance = np.zeros(zero_padding_factor * np.array(tel.pupil.shape))

pupil_fields = np.zeros(
    (*pupil_plane_field.shape, n_controlled_modes),
    dtype=full_eigen_control_basis.dtype,
)

pupil_fields[pupil_padded, :] = full_eigen_control_basis[
    tel.pupil.reshape(-1),
    :n_controlled_modes,
]

focal_plane_irradiance = np.sum(
    np.abs(
        np.fft.fftshift(
            np.fft.fft2(pupil_fields, axes=(0, 1)),
            axes=(0, 1),
        )
    )
    ** 2,
    axis=2,
)

npx = zero_padding_factor * 60
plt.figure()
plt.imshow(
    crop_array(focal_plane_irradiance, (npx, npx)),
    norm="linear",
    cmap="inferno",
)
plt.title("accessible fourier plane")

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

# %% map

reconstructor_map_sr = (
    c_phi
    @ interaction_matrix.T
    @ np.linalg.pinv(interaction_matrix @ c_phi @ interaction_matrix.T)
)

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
    polc=True,
    interaction_matrix=interaction_matrix_eigen_basis,
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

# %% Close the loop - map - SR

(
    total_map_sr,
    residual_map_sr,
    strehl_map_sr,
    dm_coefs_map_sr,
    turbulence_phase_screens_map_sr,
    residual_phase_screens_map_sr,
    wfs_frames_map_sr,
    wfs_signals_map_sr,
    short_exposure_psf_map_sr,
) = close_the_loop(
    tel,
    ngs,
    atm,
    first_calibration_modal_dm,
    pyramid_sr,
    reconstructor_map_sr,
    loop_gain,
    n_iter,
    delay=delay,
    photon_noise=detector_photon_noise,
    read_out_noise=detector_read_out_noise,
    polc=True,
    interaction_matrix=interaction_matrix,
    seed=seed,
    save_telemetry=True,
    save_psf=True,
)

# %% post processing

long_exposure_psf_map_sr = np.sum(short_exposure_psf_map_sr[:, :, 100:], axis=2)

# %% plots

# noise propagation
plt.figure()
plt.plot(np.diag(reconstructor_map_sr @ reconstructor_map_sr.T) / pyramid_sr.nSignal)
plt.yscale("log")
plt.title("modal uniform noise propagation")
plt.xlabel("# modes")
plt.savefig(fig_dir / "noise_propagation.png", bbox_inches="tight")

# %%

# residuals
plt.figure()
plt.plot(total_map_sr, label="total_map_sr")
plt.plot(residual_map_sr, label="residual_map_sr")
plt.xlabel("loop iteration")
plt.ylabel("residual phase RMS [nm]")
plt.title("Closed Loop residuals")
plt.legend()
plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")

# %%

# strehls
plt.figure()
plt.plot(strehl_map_sr, label="strehl_map_sr")
plt.ylabel("strehl phase RMS [nm]")
plt.title("Closed Loop strehls")
plt.legend()
plt.savefig(fig_dir / "strehls.png", bbox_inches="tight")

# %%

# long exposure PSF
plt.figure()
plt.imshow(np.log(long_exposure_psf_map_sr), norm="linear", cmap="inferno")
plt.title(f"long_exposure_psf_map_sr\nPyramid - {n_controlled_modes} controlled modes")
plt.savefig(fig_dir / "long_exposure_psf.png", bbox_inches="tight")

plt.show()

# %%
