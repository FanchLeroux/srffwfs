# %% imports

from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm

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
from OOPAO.BioEdge import BioEdge
from OOPAO.Detector import Detector
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

from srffwfs.pattern import get_circular_pupil
from srffwfs.modal_bases.KL_basis import compute_KL_basis
from srffwfs.sensitivity import compute_photon_noise_sensitivity
from srffwfs.reconstructor import (
    compute_lse_reconstructor,
    compute_weighted_truncated_svd_reconstructor,
    compute_map_reconstructor,
)
from srffwfs.closed_loop import close_the_loop, ClosedLoopResult
from srffwfs.miscellaneous import pad_array, crop_array

# %% Class definitions


@dataclass
class CalibratedWfs:
    name: str
    wfs: Pyramid | BioEdge
    interaction_matrix: np.ndarray
    closed_loop_results: ClosedLoopResult | None = None
    long_exposure: np.ndarray | None = None


# %% outputs directory

config = Config()
fig_dir = config.root_dir / "outputs"

# %% simulation parameters

# ---------------------- NGS ---------------------- #

# phot.R = [0.640e-6, 0.150e-6, 4.01e12/cst]
wavelength = 640e-9  # [m] wavelength of the guide star
optical_band = "R"  # optical band of the guide star
n_photons_per_controlled_mode = 100

# ------------------ ATMOSPHERE ----------------- #

r0 = 0.10  # [m] value of r0 at 500 nm
external_scale = 30  # [m] value of L0 in the visibile
fractional_r0 = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
wind_speed = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of layers
wind_direction = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
wind_direction = [wd + 90 for wd in wind_direction]  # check influence on LO contrast
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

# ------------------------ DM ---------------------- #

n_actuator = 2 * n_subaperture  # number of actuators

# ----------------------- WFS ---------------------- #

modulation = 3.0  # [lambda/D] modulation radius or half grey width
n_pix_separation = 2  # [pixel] separation ratio between the pupils
light_threshold = (
    0.3 if modulation > 0.0 else 0
)  # light threshold to select the valid pixels
detector_photon_noise = True
detector_read_out_noise = 0.0  # e- RMS

# super resolution
sr_amplitude = 0.25  # [pixel] super resolution shifts amplitude

# ----------------------- SCIENCE CAMERA ---------------------- #

science_camera_psf_sampling = (
    4  # [pixel per lambda/D] sampling of the PSF in the science camera
)
science_camera_resolution = (
    3 * science_camera_psf_sampling * n_actuator
)  # [pixel] resolution of the science camera

# -------------------- CALIBRATION - MODAL BASIS ---------------- #

modal_basis = "KL"
stroke_rad = 0.01  # [rad]
stroke = stroke_rad * wavelength / (2 * np.pi)  # [nm]
single_pass = False  # push-pull or push only for the calibration

# -------------------- LOOP ----------------------- #

loop_gain = 0.7
n_iter = 1000
delay = 1
loop_frequency = 1000  # [Hz] loop frequency
n_controlled_modes = int(
    2.0 * np.pi * (n_subaperture / 2) ** 2
)  # number of controlled modes
use_weighted_svd = True  # use weighted truncated SVD reconstructor

# ------------------- SR shifts ------------------- #

# super resolved pyramid - define the pupil shifts to be applied to the WFS
pupil_shifts_zeros_pyramid = [
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
pupil_shifts_horizontal_pyramid = [
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
pupil_shifts_vertical_pyramid = [
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
pupil_shifts_hv_pyramid = [
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
pupil_shifts_quincux_pyramid = [
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

# define a dictionary to loop over the different pupil shifts configurations
pupil_shifts_dict_pyramid = {
    "zeros": pupil_shifts_zeros_pyramid,
    "horizontal": pupil_shifts_horizontal_pyramid,
    "vertical": pupil_shifts_vertical_pyramid,
    "hv": pupil_shifts_hv_pyramid,
    "quincux": pupil_shifts_quincux_pyramid,
}
pupil_shifts_dict_pyramid = {
    "horizontal": pupil_shifts_horizontal_pyramid,
}

# Bioedge
# super resolved bioedge. Each row of pupil is one bioedge filter
# quadrant numbering: 4, 3, 2, 1 (top left, top right, bottom left, bottom right)

pupil_shifts_zeros_bioedge = [
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
pupil_shifts_horizontal_bioedge = [
    [
        0.25,
        -0.25,
        0.25,
        -0.25,
    ],
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
]
pupil_shifts_vertical_bioedge = [
    [
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.25,
        -0.25,
        0.25,
        -0.25,
    ],
]
pupil_shifts_hv_bioedge = [
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
pupil_shifts_quincux_bioedge = [
    [
        0.25,
        -0.25,
        0.25,
        -0.25,
    ],
    [
        0.25,
        -0.25,
        -0.25,
        0.25,
    ],
]

pupil_shifts_dict_bioedge = {
    "zeros": pupil_shifts_zeros_bioedge,
    "horizontal": pupil_shifts_horizontal_bioedge,
    "vertical": pupil_shifts_vertical_bioedge,
    "hv": pupil_shifts_hv_bioedge,
    "quincux": pupil_shifts_quincux_bioedge,
}
pupil_shifts_dict_bioedge = {
    "horizontal": pupil_shifts_horizontal_bioedge,
}

# %% Build objects

# % -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(
    resolution=resolution,  # [pixel] resolution of the telescope
    diameter=diameter,  # [m] telescope diameter
    samplingTime=1 / loop_frequency,  # [s] sampling time of the telescope
)
telescope_surface = tel.pupil.sum() * tel.pixelSize * tel.pixelSize

# % -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(
    optBand=optical_band,  # Source optical band
    # (see photometry.py)
    magnitude=0.0,
)  # Source Magnitude. Will be updated to match the number of photons per controlled mode

# % -----------------------    ATMOSPHERE   ----------------------------

# coupling telescope and source is mandatory to generate Atmosphere object
ngs**tel

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
    m2c, c_phi = compute_KL_basis(tel, atm, dm, return_covariance=True, n_batch=2)
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
    nRes=science_camera_resolution,
    psf_sampling=science_camera_psf_sampling,
    integrationTime=1 / loop_frequency,
    readoutNoise=0.0,  # [e-] RMS
    photonNoise=False,  # no photon noise in the science camera
)

# %% ----------------------- Pyramid ---------------------------- #

# pyramid
pyramid = Pyramid(
    nSubap=n_subaperture,
    telescope=tel,
    modulation=modulation,
    lightRatio=light_threshold,
    n_pix_separation=n_pix_separation,
    postProcessing="fullFrame",
)

bioedge = BioEdge(
    nSubap=n_subaperture,
    telescope=tel,
    modulation=0,
    lightRatio=light_threshold,
    n_pix_separation=n_pix_separation,
    postProcessing="fullFrame",
    grey_width=modulation,
)

n_measurement_points_pyramid = int(np.sum(pyramid.validSignal) / 4)
n_measurement_points_bioedge = int(np.sum(bioedge.validSignal) / 4)
reference_intensities_2d_no_sr_pyramid = pyramid.referenceSignal_2D
reference_intensities_2d_no_sr_bioedge = bioedge.referenceSignal_2D

# %% Precompute shifted and calibrated WFS for every configuration.

calibrated_pyramids: dict[str, CalibratedWfs] = {}
calibrated_bioedges: dict[str, CalibratedWfs] = {}

for shift_name, pupil_shifts in tqdm(pupil_shifts_dict_pyramid.items()):

    pyramid = Pyramid(
        nSubap=n_subaperture,
        telescope=tel,
        modulation=modulation,
        lightRatio=light_threshold,
        n_pix_separation=n_pix_separation,
        postProcessing="fullFrame",
    )

    pyramid.apply_shift_wfs(pupil_shifts[0], pupil_shifts[1], units="pixels")
    pyramid.modulation = modulation  # update reference intensities

    calibration_pyramid = InteractionMatrix(
        ngs,
        tel,
        modal_dm,
        pyramid,
        M2C=np.diag(np.ones(modal_dm.nValidAct)),
        stroke=stroke,
        single_pass=single_pass,
        noise="off",
        display=True,
        invert=False,
    )
    interaction_matrix_pyramid = calibration_pyramid.D

    calibrated_pyramids[shift_name] = CalibratedWfs(
        name=shift_name,
        wfs=pyramid,
        interaction_matrix=interaction_matrix_pyramid,
    )

# %%

for shift_name, pupil_shifts in tqdm(pupil_shifts_dict_bioedge.items()):

    bioedge = BioEdge(
        nSubap=n_subaperture,
        telescope=tel,
        modulation=0,
        lightRatio=light_threshold,
        n_pix_separation=n_pix_separation,
        postProcessing="fullFrame",
        grey_width=modulation,
    )

    bioedge.apply_shift_wfs(
        pupil_shifts[0], pupil_shifts[1], units="pixels"
    )  # to be redifined
    bioedge.modulation = 0

    calibration_bioedge = InteractionMatrix(
        ngs,
        tel,
        modal_dm,
        bioedge,
        M2C=np.diag(np.ones(modal_dm.nValidAct)),
        stroke=stroke,
        single_pass=single_pass,
        noise="off",
        display=True,
        invert=False,
    )
    interaction_matrix_bioedge = calibration_bioedge.D

    calibrated_bioedges[shift_name] = CalibratedWfs(
        name=shift_name,
        wfs=bioedge,
        interaction_matrix=interaction_matrix_bioedge,
    )

# %% Closed-loop tests

modal_projector = None
seed = 0
use_weighted_svd = True
use_map = True

# set flux of the ngs
ngs.nPhoton = (
    n_photons_per_controlled_mode
    * loop_frequency
    / (telescope_surface / n_controlled_modes)
)  # nPhoton = # photons per s per m2

ngs**tel * dm * pyramid  # coupling telescope, DM and WFS is mandatory to compute the number of photons per measurement point

n_photons_per_frame = pyramid.cam.frame.sum()  # number of photons per frame

print(
    f"Photon noise: {detector_photon_noise}\n"
    f"Read-out noise: {detector_read_out_noise}\n"
    f"# photons per second per m^2: {ngs.nPhoton}\n"
    f"# photons per controlled mode per frame: {n_photons_per_frame / n_controlled_modes}\n"
    f"# photons per measurement points per frame: {n_photons_per_frame / n_measurement_points_pyramid}"
)

for shift_name, calibrated_pyramid in calibrated_pyramids.items():
    interaction_matrix = calibrated_pyramid.interaction_matrix

    if use_weighted_svd and not use_map:
        reconstructor, singular_values = compute_weighted_truncated_svd_reconstructor(
            interaction_matrix, c_phi, n_controlled_modes
        )

        plt.figure()
        plt.plot(singular_values)
        plt.yscale("log")
        plt.title(
            f"Singular values of the weighted interaction matrix\n{shift_name} shifts"
        )
        plt.xlabel("# modes")
        plt.ylabel("Singular value")
        plt.axvline(
            x=n_controlled_modes, color="r", linestyle="--", label="modal cutoff"
        )

    if use_map:
        c_b = (detector_read_out_noise / n_photons_per_frame) ** 2
        if detector_photon_noise:
            c_b += calibrated_pyramid.wfs.referenceSignal / n_photons_per_frame
        reconstructor = compute_map_reconstructor(
            interaction_matrix,
            c_phi,
            c_b,
        )

    else:
        reconstructor = compute_lse_reconstructor(
            interaction_matrix, n_controlled_modes
        )

    reference_intensities_2d_sr_pyramid = calibrated_pyramid.wfs.referenceSignal_2D
    valid_signal_2d_sr_pyramid = calibrated_pyramid.wfs.valid_signal_2D

    plt.figure()
    plt.imshow(reference_intensities_2d_sr_pyramid)
    plt.title(f"Reference intensities\nsuper resolved pyramid\n{shift_name} shifts")

    plt.figure()
    plt.imshow(
        reference_intensities_2d_sr_pyramid - reference_intensities_2d_no_sr_pyramid
    )
    plt.title(
        f"Reference intensities difference\nsuper resolved pyramid - pyramid\n{shift_name} shifts"
    )

    dm.coefs = 0
    calibrated_pyramid.wfs.cam.photonNoise = False
    calibrated_pyramid.wfs.cam.readoutNoise = 0.0
    ngs**tel

    closed_loop_results = close_the_loop(
        ngs,
        atm,
        tel,
        modal_dm,
        calibrated_pyramid.wfs,
        reconstructor,
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
        science_camera=science_camera,
        modal_projector=modal_projector,
    )

    calibrated_pyramid.closed_loop_results = closed_loop_results

    # post processing

    long_exposure_psf_lse_sr = np.sum(
        closed_loop_results.short_exposure_psf[:, :, 100:], axis=2
    )

    # plots

    # noise propagation
    plt.figure()
    plt.plot(np.diag(reconstructor @ reconstructor.T) / calibrated_pyramid.wfs.nSignal)
    plt.yscale("log")
    plt.title(f"modal uniform noise propagation\n{shift_name} shifts")
    plt.xlabel("# modes")
    plt.savefig(fig_dir / "noise_propagation.png", bbox_inches="tight")

    fitting_error = (
        np.std(
            closed_loop_results.fitting_limited_phase_screens[tel.pupil.flatten(), :],
            axis=0,
        )
        if modal_projector is not None
        else 0.0
    )
    # residuals
    plt.figure()
    plt.plot(closed_loop_results.total, label="total")
    plt.plot(closed_loop_results.residual, label="residual")
    plt.plot(fitting_error, label="fitting error")
    plt.xlabel("loop iteration")
    plt.ylabel("residual phase RMS [nm]")
    plt.title(f"Closed Loop residuals\n{shift_name} shifts")
    plt.legend()
    plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")

    # strehls
    plt.figure()
    plt.plot(closed_loop_results.strehl, label="strehl")
    plt.ylabel("strehl phase RMS [nm]")
    plt.title(f"Closed Loop strehls\n{shift_name} shifts")
    plt.legend()
    plt.savefig(fig_dir / "strehls.png", bbox_inches="tight")

    # long exposure PSF
    plt.figure()
    plt.imshow(
        np.log(long_exposure_psf_lse_sr),
        norm="linear",
        cmap="inferno",
    )
    n_controlled_modes_str = (
        "MAP" if use_map else f"{n_controlled_modes} controlled modes"
    )
    plt.title(
        f"long_exposure_psf_lse_sr\nPyramid - {n_controlled_modes_str}\n{shift_name} shifts"
    )
    plt.savefig(
        fig_dir / f"long_exposure_psf{shift_name}_shifts.png",
        bbox_inches="tight",
    )

    if modal_projector is not None:
        # ideal coronograph fitting limited
        pupil = tel.pupil.astype(float)
        n_pix, pad = tel.resolution, 2
        opd = (
            closed_loop_results.fitting_limited_phase_screens.copy()
        )  # (n_pix**2, n_iter)
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
        plt.title(
            f"long exposure psf fitting limited ideal coronagraph\n{shift_name} shifts"
        )

# %% bioedge

for shift_name, calibrated_bioedge in calibrated_bioedges.items():
    interaction_matrix = calibrated_bioedge.interaction_matrix

    if use_weighted_svd and not use_map:
        reconstructor, singular_values = compute_weighted_truncated_svd_reconstructor(
            interaction_matrix, c_phi, n_controlled_modes
        )

        plt.figure()
        plt.plot(singular_values)
        plt.yscale("log")
        plt.title(
            f"Singular values of the weighted interaction matrix\n{shift_name} shifts"
        )
        plt.xlabel("# modes")
        plt.ylabel("Singular value")
        plt.axvline(
            x=n_controlled_modes, color="r", linestyle="--", label="modal cutoff"
        )

    if use_map:
        c_b = (detector_read_out_noise / n_photons_per_frame) ** 2
        if detector_photon_noise:
            c_b += calibrated_bioedge.wfs.referenceSignal / n_photons_per_frame
        reconstructor = compute_map_reconstructor(
            interaction_matrix,
            c_phi,
            c_b,
        )

    else:
        reconstructor = compute_lse_reconstructor(
            interaction_matrix, n_controlled_modes
        )

    reference_intensities_2d_sr_bioedge = calibrated_bioedge.wfs.referenceSignal_2D
    valid_signal_2d_sr_bioedge = calibrated_bioedge.wfs.valid_signal_2D

    plt.figure()
    plt.imshow(reference_intensities_2d_sr_bioedge)
    plt.title(f"Reference intensities\nsuper resolved bioedge\n{shift_name} shifts")

    plt.figure()
    plt.imshow(
        reference_intensities_2d_sr_bioedge - reference_intensities_2d_no_sr_bioedge
    )
    plt.title(
        f"Reference intensities difference\nsuper resolved bioedge - bioedge\n{shift_name} shifts"
    )

    dm.coefs = 0
    calibrated_bioedge.wfs.cam.photonNoise = False
    calibrated_bioedge.wfs.cam.readoutNoise = 0.0
    ngs**tel

    closed_loop_results = close_the_loop(
        ngs,
        atm,
        tel,
        modal_dm,
        calibrated_bioedge.wfs,
        reconstructor,
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
        science_camera=science_camera,
        modal_projector=modal_projector,
    )

    calibrated_bioedge.closed_loop_results = closed_loop_results

    # post processing

    long_exposure_psf_lse_sr = np.sum(
        closed_loop_results.short_exposure_psf[:, :, 100:], axis=2
    )

    # plots

    # noise propagation
    plt.figure()
    plt.plot(np.diag(reconstructor @ reconstructor.T) / calibrated_bioedge.wfs.nSignal)
    plt.yscale("log")
    plt.title(f"modal uniform noise propagation\n{shift_name} shifts")
    plt.xlabel("# modes")
    plt.savefig(fig_dir / "noise_propagation.png", bbox_inches="tight")

    fitting_error = (
        np.std(
            closed_loop_results.fitting_limited_phase_screens[tel.pupil.flatten(), :],
            axis=0,
        )
        if modal_projector is not None
        else 0.0
    )
    # residuals
    plt.figure()
    plt.plot(closed_loop_results.total, label="total")
    plt.plot(closed_loop_results.residual, label="residual")
    plt.plot(fitting_error, label="fitting error")
    plt.xlabel("loop iteration")
    plt.ylabel("residual phase RMS [nm]")
    plt.title(f"Closed Loop residuals\n{shift_name} shifts")
    plt.legend()
    plt.savefig(fig_dir / "residuals.png", bbox_inches="tight")

    # strehls
    plt.figure()
    plt.plot(closed_loop_results.strehl, label="strehl")
    plt.ylabel("strehl phase RMS [nm]")
    plt.title(f"Closed Loop strehls\n{shift_name} shifts")
    plt.legend()
    plt.savefig(fig_dir / "strehls.png", bbox_inches="tight")

    # long exposure PSF
    plt.figure()
    plt.imshow(
        np.log(long_exposure_psf_lse_sr),
        norm="linear",
        cmap="inferno",
    )
    n_controlled_modes_str = (
        "MAP" if use_map else f"{n_controlled_modes} controlled modes"
    )
    plt.title(
        f"long_exposure_psf_lse_sr\nbioedge - {n_controlled_modes_str}\n{shift_name} shifts"
    )
    plt.savefig(
        fig_dir / f"long_exposure_psf{shift_name}_shifts.png",
        bbox_inches="tight",
    )

    if modal_projector is not None:
        # ideal coronograph fitting limited
        pupil = tel.pupil.astype(float)
        n_pix, pad = tel.resolution, 2
        opd = (
            closed_loop_results.fitting_limited_phase_screens.copy()
        )  # (n_pix**2, n_iter)
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
        plt.title(
            f"long exposure psf fitting limited ideal coronagraph\n{shift_name} shifts"
        )

# %% plot contrast


def compute_contrast(residual_phase_screens, pupil, pad=2, remove_tip_tilt=False):
    """
    Compute the long-exposure contrast (2D map) with an ideal coronagraph.

    The residual phase screens are converted to a complex field with constant
    amplitude over the pupil. The piston field mode (the component of the field
    proportional to the pupil) is removed to emulate an ideal coronagraph, and
    the resulting PSFs are averaged over all screens.

    Parameters
    ----------
    residual_phase_screens : numpy.ndarray
        Residual phase screens in radians, with shape ``(npx, npx, n_screens)``.
    pupil : numpy.ndarray
        Binary pupil mask of shape ``(npx, npx)``. The amplitude is assumed
        constant over the pupil.
    pad : int, optional
        Zero-padding factor. It sets the PSF sampling, in pixels per
        :math:`\\lambda/D`. Default is 2.
    remove_tip_tilt : bool, optional
        If True, subtract the least-squares piston/tip/tilt fitted over the
        pupil from each screen before computing the field (perfect tip-tilt
        correction applied independently to each screen). Default is False.

    Returns
    -------
    long_exposure : numpy.ndarray
        Long-exposure coronagraphic PSF of shape ``(npx * pad, npx * pad)``,
        normalised by the peak of the diffraction-limited (non-coronagraphic)
        PSF.

    Examples
    --------
    >>> contrast = compute_contrast(screens, tel.pupil, pad=2, remove_tip_tilt=True)
    >>> contrast.shape
    (512, 512)

    Author
    ------
    François Leroux (ONERA), Claude Sonnet 5.5 (Anthropic)

    Date
    ----
    2026-11-23
    """
    pupil = np.asarray(pupil, dtype=float)
    mask = pupil > 0
    npx, _, n_screens = residual_phase_screens.shape
    n_fft = pad * npx
    pupil_norm = np.sum(pupil**2)

    # normalisation: peak of the diffraction-limited PSF
    peak = np.abs(np.fft.fft2(pupil, s=(n_fft, n_fft))).max() ** 2

    # piston / tip / tilt basis restricted to the pupil pixels, and its pseudo-inverse
    if remove_tip_tilt:
        y, x = np.mgrid[:npx, :npx]
        x = (x - x[mask].mean()) / npx
        y = (y - y[mask].mean()) / npx
        ptt_basis = np.stack(
            [np.ones(mask.sum()), x[mask], y[mask]], axis=1
        )  # (n_in_pupil, 3)
        ptt_pinv = np.linalg.pinv(ptt_basis)  # (3, n_in_pupil)

    long_exposure = np.zeros((n_fft, n_fft))

    # accumulate screen by screen to avoid a (n_fft, n_fft, n_screens) complex cube
    for k in tqdm(range(n_screens), desc="Computing long-exposure contrast"):
        phase = np.where(mask, residual_phase_screens[:, :, k], 0.0)

        if remove_tip_tilt:
            phase_in_pupil = phase[mask]
            coefs = ptt_pinv @ phase_in_pupil
            phase[mask] = phase_in_pupil - ptt_basis @ coefs

        field = pupil * np.exp(1j * phase)

        # ideal coronagraph: remove the piston (pupil-shaped) component of the field
        field -= (np.sum(pupil * field) / pupil_norm) * pupil

        psf = np.abs(np.fft.fftshift(np.fft.fft2(field, s=(n_fft, n_fft)))) ** 2
        long_exposure += psf

    long_exposure /= n_screens
    long_exposure /= peak

    return long_exposure


# %%

n_bootstrap_frames = 25
remove_tip_tilt = False
padding_factor = 4
pupil = tel.pupil.astype(float)

for calibrated_pyramid, calibrated_bioedge in zip(
    calibrated_pyramids.values(), calibrated_bioedges.values()
):
    calibrated_pyramid.long_exposure = compute_contrast(
        calibrated_pyramid.closed_loop_results.residual_phase_screens[
            :, :, n_bootstrap_frames:
        ],
        pupil,
        pad=padding_factor,
        remove_tip_tilt=remove_tip_tilt,
    )
    calibrated_bioedge.long_exposure = compute_contrast(
        calibrated_bioedge.closed_loop_results.residual_phase_screens[
            :, :, n_bootstrap_frames:
        ],
        pupil,
        pad=padding_factor,
        remove_tip_tilt=remove_tip_tilt,
    )

# %%

field_of_view_factor = 1.2  # crop the field of view for display purposes. Percentage of the correction zone

for calibrated_pyramid, calibrated_bioedge in zip(
    calibrated_pyramids.values(), calibrated_bioedges.values()
):
    psfs = {
        f"pyramid {calibrated_pyramid.name}": np.log10(
            crop_array(
                calibrated_pyramid.long_exposure,
                int(n_actuator * padding_factor * field_of_view_factor),
            )
        ),
        f"Bi-O edge {calibrated_bioedge.name}": np.log10(
            crop_array(
                calibrated_bioedge.long_exposure,
                int(n_actuator * padding_factor * field_of_view_factor),
            )
        ),
    }

    # shared color scale so the two panels are directly comparable
    vmax = max(np.nanmax(p) for p in psfs.values())
    vmin = min(np.nanpercentile(p[np.isfinite(p)], 1) for p in psfs.values())

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5.5), sharex=True, sharey=True, layout="constrained"
    )

    for ax, (title, psf) in zip(axes, psfs.items()):
        im = ax.imshow(psf, cmap="inferno", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(
        im,
        ax=axes,
        location="bottom",
        orientation="horizontal",
        shrink=0.6,
        aspect=40,
        pad=0.04,
    )
    cbar.set_label(r"$\log_{10}$ (normalised intensity)", fontsize=11)
    cbar.outline.set_visible(False)


# %% plot done by claude

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid

# ------------------------- figure settings ------------------------- #

COLUMN_WIDTH_IN = 3.5  # single column; use ~7.2 to span both columns
FIGURE_HEIGHT_IN = 2.6

PAPER_STYLE = {
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
}

cmap = plt.get_cmap("inferno").copy()
cmap.set_bad("white")  # NaN pixels (zero contrast) are drawn in white

# crop geometry: padding_factor pixels per lambda/D
n_pix_crop = int(n_actuator * padding_factor * field_of_view_factor)
half_fov = n_actuator * field_of_view_factor / 2  # [lambda/D]
extent = (-half_fov, half_fov, -half_fov, half_fov)


# --------------------------- helpers --------------------------- #


def prepare_psf(long_exposure):
    """Crop the PSF and replace non-positive values by NaN."""
    psf = crop_array(long_exposure, n_pix_crop)
    return np.where(psf > 0, psf, np.nan)


def decade_limits(psfs):
    """Colour limits shared by all panels, rounded outward to whole decades."""
    values = np.concatenate([psf[np.isfinite(psf)] for psf in psfs])
    vmin = 10 ** np.floor(np.log10(np.percentile(values, 1)))
    vmax = 10 ** np.ceil(np.log10(values.max()))
    return vmin, vmax


# ----------------------------- plot ----------------------------- #

with plt.rc_context(PAPER_STYLE):
    for calibrated_pyramid, calibrated_bioedge in zip(
        calibrated_pyramids.values(), calibrated_bioedges.values()
    ):
        panels = {
            f"Pyramid {calibrated_pyramid.name}": prepare_psf(
                calibrated_pyramid.long_exposure
            ),
            f"Bi-O edge {calibrated_bioedge.name}": prepare_psf(
                calibrated_bioedge.long_exposure
            ),
        }
        norm = LogNorm(*decade_limits(panels.values()))

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(COLUMN_WIDTH_IN, 2.3),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        # tight padding [in] around the figure and between panels
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

        for ax, (title, psf) in zip(axes, panels.items()):
            im = ax.imshow(psf, cmap=cmap, norm=norm, origin="lower", extent=extent)
            ax.set_title(title, pad=3)
            ax.set_xlabel(r"$\lambda/D$", labelpad=1)
            ax.locator_params(nbins=4)
            ax.tick_params(pad=2)

        axes[0].set_ylabel(r"$\lambda/D$", labelpad=1)
        # hide the redundant y tick labels of the right panel
        axes[1].tick_params(labelleft=False)

        cbar = fig.colorbar(
            im,
            ax=axes,
            location="bottom",
            fraction=0.06,  # colorbar thickness relative to the panels
            aspect=40,
            pad=0.02,
        )
        cbar.set_label("Contrast", labelpad=1)

        stem = f"long_exposure_contrast_pyr_{calibrated_pyramid.name}_bio_{calibrated_bioedge.name}"
        fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(fig_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.01)

# %%

plt.show()
