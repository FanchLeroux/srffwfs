"""
Run in tmux session with
tmux new-session -d -s mysession "uv run python path/to/script.py"
"""

# %% imports

from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

from maoppy.utils import circavg

from srffwfs.pattern import get_circular_pupil
from srffwfs.modal_bases.KL_basis import compute_KL_basis
from srffwfs.sensitivity import compute_photon_noise_sensitivity
from srffwfs.reconstructor import (
    compute_lse_reconstructor,
    compute_weighted_truncated_svd_reconstructor,
    compute_map_reconstructor,
)
from srffwfs.closed_loop import close_the_loop, ClosedLoopResult
from srffwfs.performance_metric import compute_contrast
from srffwfs.miscellaneous import pad_array, crop_array

# %% Class definitions


@dataclass
class CalibratedWfs:
    name: str
    wfs: Pyramid | BioEdge
    interaction_matrix: np.ndarray
    closed_loop_results: ClosedLoopResult | None = None
    long_exposure: np.ndarray | None = None
    long_exposure_coronograph: np.ndarray | None = None
    radial_average: np.ndarray | None = None
    radial_average_coronograph: np.ndarray | None = None


# %% outputs directory

config = Config()
fig_dir = config.root_dir / "outputs"

# %% simulation parameters

# ---------------------- NGS ---------------------- #

# phot.R = [0.640e-6, 0.150e-6, 4.01e12/cst]
wavelength = 640e-9  # [m] wavelength of the guide star
optical_band = "R"  # optical band of the guide star
n_photons_per_frame = 100_000.0  # number of photons per frame

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
    "zeros": pupil_shifts_zeros_pyramid,
    "quincux": pupil_shifts_quincux_pyramid,
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
    "zeros": pupil_shifts_zeros_bioedge,
    "quincux": pupil_shifts_quincux_bioedge,
}

# --------------- PSF computation parameters --------------- #

padding_factor = 4  # pixel per lambda/D
field_of_view_factor = 1.2  # crop the field of view for display purposes. Percentage of the correction zone

# ----------------------- SCIENCE CAMERA ---------------------- #

science_camera_resolution = int(
    field_of_view_factor * padding_factor * n_actuator
)  # [pixel] resolution of the science camera
science_camera_resolution += science_camera_resolution % 2  # make it even

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
    psf_sampling=padding_factor,
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

# pyramid

modal_projector = None
seed = 0
use_weighted_svd = False
use_map = True

# set flux of the ngs
photon_flux_pyramid = (
    n_photons_per_frame * loop_frequency / telescope_surface
)  # nPhoton = # photons per s per m2
ngs.nPhoton = photon_flux_pyramid

ngs**tel * dm * pyramid  # coupling telescope, DM and WFS is mandatory to compute the number of photons per measurement point

n_photons_per_frame_pyramid = pyramid.cam.frame.sum()  # number of photons per frame

print(
    f"Photon noise: {detector_photon_noise}\n"
    f"Read-out noise: {detector_read_out_noise}\n"
    f"# photons per second per m^2: {ngs.nPhoton}\n"
    f"# photons per frame: {n_photons_per_frame_pyramid}\n"
    f"# photons per measurement points per frame: {n_photons_per_frame_pyramid / n_measurement_points_pyramid}"
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

    long_exposure_coronograph_psf_lse_sr = np.sum(
        closed_loop_results.short_exposure_psf[:, :, 100:], axis=2
    )

    # plots

    # noise propagation
    plt.figure()
    plt.plot(np.diag(reconstructor @ reconstructor.T) / calibrated_pyramid.wfs.nSignal)
    plt.yscale("log")
    plt.title(f"modal uniform noise propagation\n{shift_name} shifts")
    plt.xlabel("# modes")

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

    # strehls
    plt.figure()
    plt.plot(closed_loop_results.strehl, label="strehl")
    plt.ylabel("strehl phase RMS [nm]")
    plt.title(f"Closed Loop strehls\n{shift_name} shifts")
    plt.legend()

    # long exposure PSF
    plt.figure()
    plt.imshow(
        np.log(long_exposure_coronograph_psf_lse_sr),
        norm="linear",
        cmap="inferno",
    )
    n_controlled_modes_str = (
        "MAP" if use_map else f"{n_controlled_modes} controlled modes"
    )
    plt.title(
        f"long_exposure_coronograph_psf_lse_sr\nPyramid - {n_controlled_modes_str}\n{shift_name} shifts"
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

# bioedge

bioedge_handicap_factor = 1.0

# set flux of the ngs
photon_flux_bioedge = bioedge_handicap_factor * (
    n_photons_per_frame * loop_frequency / telescope_surface
)  # nPhoton = # photons per s per m2
ngs.nPhoton = photon_flux_bioedge

ngs**tel * dm * pyramid  # coupling telescope, DM and WFS is mandatory to compute the number of photons per measurement point

n_photons_per_frame_bioedge = pyramid.cam.frame.sum()  # number of photons per frame

print(
    f"Photon noise: {detector_photon_noise}\n"
    f"Read-out noise: {detector_read_out_noise}\n"
    f"# photons per second per m^2: {ngs.nPhoton}\n"
    f"# photons per frame: {n_photons_per_frame_bioedge}\n"
    f"# photons per measurement points per frame: {n_photons_per_frame_bioedge / n_measurement_points_pyramid}"
)

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
        c_b = (
            detector_read_out_noise / bioedge_handicap_factor * n_photons_per_frame
        ) ** 2
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

    long_exposure_coronograph_psf_lse_sr = np.sum(
        closed_loop_results.short_exposure_psf[:, :, 100:], axis=2
    )

    # plots

    # noise propagation
    plt.figure()
    plt.plot(np.diag(reconstructor @ reconstructor.T) / calibrated_bioedge.wfs.nSignal)
    plt.yscale("log")
    plt.title(f"modal uniform noise propagation\n{shift_name} shifts")
    plt.xlabel("# modes")

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

    # strehls
    plt.figure()
    plt.plot(closed_loop_results.strehl, label="strehl")
    plt.ylabel("strehl phase RMS [nm]")
    plt.title(f"Closed Loop strehls\n{shift_name} shifts")
    plt.legend()

    # long exposure PSF
    plt.figure()
    plt.imshow(
        np.log(long_exposure_coronograph_psf_lse_sr),
        norm="linear",
        cmap="inferno",
    )
    n_controlled_modes_str = (
        "MAP" if use_map else f"{n_controlled_modes} controlled modes"
    )
    plt.title(
        f"long_exposure_coronograph_psf_lse_sr\nbioedge - {n_controlled_modes_str}\n{shift_name} shifts"
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

# %% compute long exposure before and after the coronograph

n_bootstrap_frames = 25
remove_tip_tilt = False
pupil = tel.pupil.astype(float)

for cal in [*calibrated_pyramids.values(), *calibrated_bioedges.values()]:
    results = cal.closed_loop_results
    cal.long_exposure = results.short_exposure_psf[:, :, n_bootstrap_frames:].sum(
        axis=2
    )
    cal.long_exposure /= cal.long_exposure.max()
    cal.long_exposure_coronograph = compute_contrast(
        results.residual_phase_screens[:, :, n_bootstrap_frames:],
        pupil,
        pad=padding_factor,
        remove_tip_tilt=remove_tip_tilt,
    )
    cal.radial_average = circavg(
        cal.long_exposure,
        center=[
            cal.long_exposure.shape[0] // 2,
            cal.long_exposure.shape[1] // 2,
        ],
    )
    cal.radial_average_coronograph = circavg(
        cal.long_exposure_coronograph,
        center=[
            cal.long_exposure_coronograph.shape[0] // 2,
            cal.long_exposure_coronograph.shape[1] // 2,
        ],
    )

# %% plots (claude sonnet 5.5 assisted)

# ------------------------- figure settings ------------------------- #

FIGURE_WIDTH_IN = 3.5  # single column; use ~7.2 to span both columns
FIGURE_HEIGHT_IN = 2.8

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

PAPER_STYLE = {
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "savefig.dpi": 300,
}

# --------------------------- helpers --------------------------- #


def decade_limits(images):
    """Colour limits shared by the panels of a row, rounded outward to whole decades."""
    values = np.concatenate(
        [img[np.isfinite(img)] for img in images]
    )  # flatten and remove NaNs/Infs
    vmin = 10 ** np.floor(
        np.log10(np.percentile(values, 1))
    )  # ignore darkest 1% of pixels
    vmax = 10 ** np.ceil(np.log10(values.max()))
    return vmin, vmax


# %%

# crop geometry: padding_factor pixels per lambda/D
n_pix_crop = int(n_actuator * padding_factor * field_of_view_factor)
half_fov = n_actuator * field_of_view_factor / 2  # [lambda/D]
extent = (-half_fov, half_fov, -half_fov, half_fov)  # [lambda/D]

ROW_LABELS = ("Before coronagraph", "After coronagraph")
ROW_CBAR_LABELS = ("Normalized intensity", "Contrast")


# ----------------------------- plot ----------------------------- #

with plt.rc_context(PAPER_STYLE):
    for calibrated_pyramid, calibrated_bioedge in zip(
        calibrated_pyramids.values(), calibrated_bioedges.values()
    ):
        titles = (
            f"Pyramid {calibrated_pyramid.name}",
            f"Bi-O edge {calibrated_bioedge.name}",
        )
        # rows: before / after coronagraph; columns: pyramid / bi-o edge
        rows = (
            (
                crop_array(calibrated_pyramid.long_exposure, n_pix_crop),
                crop_array(calibrated_bioedge.long_exposure, n_pix_crop),
            ),
            (
                crop_array(calibrated_pyramid.long_exposure_coronograph, n_pix_crop),
                crop_array(calibrated_bioedge.long_exposure_coronograph, n_pix_crop),
            ),
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        # tight padding [in] around the figure and between panels
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.01, hspace=0.02)

        for i, (row_images, row_label, cbar_label) in enumerate(
            zip(rows, ROW_LABELS, ROW_CBAR_LABELS)
        ):
            norm = LogNorm(*decade_limits(row_images))  # one scale per row

            for j, image in enumerate(row_images):
                ax = axes[i, j]
                im = ax.imshow(
                    image, cmap="inferno", norm=norm, origin="lower", extent=extent
                )
                ax.locator_params(nbins=4)
                ax.tick_params(
                    pad=2, color="white", labelcolor="black"
                )  # white marks, black numbers
                if i == 0:
                    ax.set_title(titles[j], pad=3)
                if i == 1:
                    ax.set_xlabel(r"$\lambda/D$", labelpad=1)
                if j == 0:
                    ax.set_ylabel(f"{row_label}\n" + r"$\lambda/D$", labelpad=1)
                else:
                    ax.tick_params(labelleft=False)

            cbar = fig.colorbar(
                im,
                ax=axes[i, :],
                location="right",
                fraction=0.05,  # colorbar thickness relative to the panels
                aspect=25,
                pad=0.02,
            )
            cbar.set_label(cbar_label, labelpad=1)

        stem = f"long_exposure_before_after_coronograph_pyr_{calibrated_pyramid.name}_bio_{calibrated_bioedge.name}_{n_photons_per_frame:.0f}_photons_per_frame"
        fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(fig_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.01)

# %% radial average plots

# ------------------------- figure settings ------------------------- #

FIGURE_WIDTH_IN = 3.5  # single column; use ~7.2 to span both columns
FIGURE_WIDTH_IN = 3.6

# crop geometry: padding_factor pixels per lambda/D
n_pix_crop = int(n_actuator * padding_factor * field_of_view_factor)
n_half = n_pix_crop // 2
half_fov = n_actuator * field_of_view_factor / 2  # [lambda/D]
r = np.arange(n_pix_crop // 2) / padding_factor  # [lambda/D]

PANELS = (  # (attribute, y label, first index to plot)
    ("radial_average", "Normalized intensity", 0),
    ("radial_average_coronograph", "Contrast", 1),  # skip r = 0
)
CURVES = (  # (label, color, linestyle)
    ("Pyramid", "C0", "-"),
    ("Bi-O edge", "C3", "--"),
)

# ----------------------------- plot ----------------------------- #

with plt.rc_context(PAPER_STYLE):
    for calibrated_pyramid, calibrated_bioedge in zip(
        calibrated_pyramids.values(), calibrated_bioedges.values()
    ):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN),
            sharex=True,
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.02)

        for ax, (attr, ylabel, start), tag in zip(axes, PANELS, ("(a)", "(b)")):
            for (label, color, ls), cal in zip(
                CURVES, (calibrated_pyramid, calibrated_bioedge)
            ):
                ax.plot(
                    r[start:],
                    getattr(cal, attr)[start:n_half],
                    color=color,
                    ls=ls,
                    label=label,
                )
            ax.set_yscale("log")
            ax.set_xlim(0, half_fov)
            ax.set_ylabel(ylabel, labelpad=1)
            ax.tick_params(pad=2)
            ax.grid(True, which="major", alpha=0.3, lw=0.4)
            ax.text(0.03, 0.94, tag, transform=ax.transAxes, va="top", weight="bold")

        axes[0].legend(frameon=False, loc="upper right")
        axes[0].set_title(
            f"{calibrated_pyramid.name} / {calibrated_bioedge.name}", pad=3
        )
        axes[1].set_xlabel(r"Separation [$\lambda/D$]", labelpad=1)

        stem = f"radial_average_pyr_{calibrated_pyramid.name}_bio_{calibrated_bioedge.name}_{n_photons_per_frame:.0f}_photons_per_frame"
        fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(fig_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.01)

# %% radial average plots: zero shifts vs other configs

# ------------------------- figure settings ------------------------- #

FIGURE_WIDTH_IN = 3.5  # single column; use ~7.2 to span both columns
FIGURE_WIDTH_IN = 3.6


REFERENCE_CONFIG = "zeros"

# crop geometry: padding_factor pixels per lambda/D
n_pix_crop = int(n_actuator * padding_factor * field_of_view_factor)
n_half = n_pix_crop // 2
half_fov = n_actuator * field_of_view_factor / 2  # [lambda/D]
r = np.arange(n_half) / padding_factor  # [lambda/D]

PANELS = (  # (attribute, y label, first index to plot)
    ("radial_average", "Normalized intensity", 0),
    ("radial_average_coronograph", "Contrast", 1),  # skip r = 0
)
CURVE_STYLES = (("k", "-"), ("C3", "--"))  # (reference, other): (color, linestyle)
WFS_GROUPS = {  # display name -> (file slug, dict of CalibratedWfs)
    "Pyramid": ("pyr", calibrated_pyramids),
    "Bi-O edge": ("bio", calibrated_bioedges),
}

# ----------------------------- plot ----------------------------- #

with plt.rc_context(PAPER_STYLE):
    for wfs_name, (slug, calibrated) in WFS_GROUPS.items():
        reference = calibrated[REFERENCE_CONFIG]

        for config, other in calibrated.items():
            if config == REFERENCE_CONFIG:
                continue

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN),
                sharex=True,
                layout="constrained",
            )
            fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.02)

            for ax, (attr, ylabel, start), tag in zip(axes, PANELS, ("(a)", "(b)")):
                for cal, (color, ls) in zip((reference, other), CURVE_STYLES):
                    ax.plot(
                        r[start:],
                        getattr(cal, attr)[start:n_half],
                        color=color,
                        ls=ls,
                        label=cal.name,
                    )
                ax.set_yscale("log")
                ax.set_xlim(0, half_fov)
                ax.set_ylabel(ylabel, labelpad=1)
                ax.tick_params(pad=2)
                ax.grid(True, which="major", alpha=0.3, lw=0.4)
                ax.text(
                    0.03, 0.94, tag, transform=ax.transAxes, va="top", weight="bold"
                )

            axes[0].legend(frameon=False, loc="upper right")
            axes[0].set_title(wfs_name, pad=3)
            axes[1].set_xlabel(r"Separation [$\lambda/D$]", labelpad=1)

            stem = f"radial_average_{slug}_{REFERENCE_CONFIG}_vs_{config}_{n_photons_per_frame:.0f}_photons_per_frame"
            fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
            fig.savefig(fig_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.01)

# %% radial average plots (one 2x2 figure per section)

# ------------------------- figure settings ------------------------- #

FIGURE_WIDTH_IN = 7.2  # spans both columns; use ~3.5 for a single column
FIGURE_HEIGHT_IN = 3.6

# crop geometry: padding_factor pixels per lambda/D
n_pix_crop = int(n_actuator * padding_factor * field_of_view_factor)
n_half = n_pix_crop // 2
half_fov = n_actuator * field_of_view_factor / 2  # [lambda/D]
r = np.arange(n_half) / padding_factor  # [lambda/D]

PANELS = (  # (attribute, y label, first index to plot)
    ("radial_average", "Normalized intensity", 0),
    ("radial_average_coronograph", "Contrast", 1),  # skip r = 0
)


def plot_radial_grid(columns, title_fn, stem):
    """
    columns: list of (column_title, [(cal, label, color, linestyle), ...]).
    Rows are the PANELS, columns are the entries of `columns`.
    """
    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        sharex=True,
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    for j, (col_title, curves) in enumerate(columns):
        for i, (attr, ylabel, start) in enumerate(PANELS):
            ax = axes[i, j]
            for cal, label, color, ls in curves:
                ax.plot(
                    r[start:],
                    getattr(cal, attr)[start:n_half],
                    color=color,
                    ls=ls,
                    label=label,
                )
            ax.set_yscale("log")
            ax.set_xlim(0, half_fov)
            ax.tick_params(pad=2)
            ax.grid(True, which="major", alpha=0.3, lw=0.4)
            ax.text(
                0.03,
                0.94,
                f"({chr(ord('a') + i * len(columns) + j)})",
                transform=ax.transAxes,
                va="top",
                weight="bold",
            )
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=1)
            if i == 0:
                ax.set_title(title_fn(col_title), pad=3)
            if i == len(PANELS) - 1:
                ax.set_xlabel(r"Separation [$\lambda/D$]", labelpad=1)

    axes[0, 0].legend(frameon=False, loc="upper right")

    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(fig_dir / f"{stem}.png", bbox_inches="tight", pad_inches=0.01)
    return fig


# ---------- section 1: Pyramid vs Bi-O edge, one column per config ---------- #

columns = [
    (
        pyr.name,
        [
            (pyr, "Pyramid", "C0", "-"),
            (bio, "Bi-O edge", "C3", "--"),
        ],
    )
    for pyr, bio in zip(calibrated_pyramids.values(), calibrated_bioedges.values())
]
names = "_".join(name for name, _ in columns)

with plt.rc_context(PAPER_STYLE):
    plot_radial_grid(
        columns,
        title_fn=lambda name: f"{name} shifts",
        stem=f"radial_average_pyr_vs_bio_{names}_{n_photons_per_frame:.0f}_photons_per_frame",
    )

# ---------- section 2: zero shifts vs other config, one column per WFS ---------- #

REFERENCE_CONFIG = "zeros"
CURVE_STYLES = (("k", "-"), ("C3", "--"))  # (reference, other): (color, linestyle)
WFS_GROUPS = {  # display name -> dict of CalibratedWfs
    "Pyramid": calibrated_pyramids,
    "Bi-O edge": calibrated_bioedges,
}

columns = []
for wfs_name, calibrated in WFS_GROUPS.items():
    reference = calibrated[REFERENCE_CONFIG]
    for config, other in calibrated.items():
        if config == REFERENCE_CONFIG:
            continue
        columns.append(
            (
                wfs_name,
                [
                    (cal, f"{cal.name} shifts", color, ls)
                    for cal, (color, ls) in zip((reference, other), CURVE_STYLES)
                ],
            )
        )

with plt.rc_context(PAPER_STYLE):
    plot_radial_grid(
        columns,
        title_fn=lambda name: name,
        stem=f"radial_average_{REFERENCE_CONFIG}_vs_others_{n_photons_per_frame:.0f}_photons_per_frame",
    )

# %%

plt.show()
