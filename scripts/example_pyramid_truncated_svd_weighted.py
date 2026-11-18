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
from OOPAO.Detector import Detector
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

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

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
wavelength = 670e-9  # [m] wavelength of the guide star
optical_band = "R4"  # optical band of the guide star
magnitude = 12  # magnitude of the guide star

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

modulation = 3.0  # [lambda/D] modulation radius or half grey width
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

# create the Telescope object
tel = Telescope(
    resolution=resolution,  # [pixel] resolution of the telescope
    diameter=diameter,  # [m] telescope diameter
    samplingTime=1 / loop_frequency,  # [s] sampling time of the telescope
)
pupil_oversampled = get_circular_pupil(tel.resolution * pupil_oversampling_factor)
pupil_binned = pupil_oversampled.reshape(
    tel.resolution,
    pupil_oversampling_factor,
    tel.resolution,
    pupil_oversampling_factor,
).mean(axis=(1, 3))

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

# ------------------------- MODAL BASIS -------------------------------

if modal_basis == "KL":
    m2c, c_phi = compute_KL_basis(tel, atm, dm, return_covariance=True)
    ngs**tel  # reset

elif modal_basis == "poke":
    m2c = np.identity(dm.nValidAct)

# extract calibration basis

influence_functions = dm.modes
calibration_basis = influence_functions @ m2c
calibration_basis = tel.pupil.reshape(-1, 1) * calibration_basis  # apply pupil mask

# -------------------------   Modal  DM   ----------------------------------

modal_dm = DeformableMirror(tel, nSubap=n_actuator, modes=calibration_basis)

# % ----------------------- Science camera ---------------------------- #

science_camera = Detector(
    nRes=tel.resolution, psf_sampling=4, integrationTime=1 / loop_frequency
)


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
    "zeros": pupil_shifts_zeros,
    "hv": pupil_shifts_hv,
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

    n_modes = int(np.pi * (pyramid_sr.nSubap / 2) ** 2)  # number of controlled modes

    # # compute classic lse reconstructor
    # reconstructor_lse = np.linalg.pinv(
    #     interaction_matrix[:, :n_modes]
    # )  # unweighted LSE reconstructor
    # reconstructor_lse = np.concatenate(
    #     (
    #         reconstructor_lse,
    #         np.zeros(
    #             (interaction_matrix.shape[1] - n_modes, reconstructor_lse.shape[1])
    #         ),
    #     ),
    #     axis=0,
    # )  # pad the reconstructor with zeros to match the number of WFS signals

    # compute reconstructor with truncated SVD weightened by the phase covariance matrix
    L = np.linalg.cholesky(c_phi)
    A = interaction_matrix @ L
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :n_modes]
    s_k = s[:n_modes]
    Vh_k = Vh[:n_modes, :]
    reconstructor_lse = L @ Vh_k.T / s_k @ U_k.T
    T = L @ Vh_k.T  # (n_cal, n_modes): control modes in original-basis coefficients
    control_basis = calibration_basis @ T  # control modes in phase space
    modal_projector_ctrl = np.linalg.pinv(
        control_basis
    )  # phase -> control-mode coefficients z

    # phase -> coefficients on the ORIGINAL basis, restricted to the controlled subspace
    modal_projector = T @ modal_projector_ctrl  # (n_cal, n_pix)
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
