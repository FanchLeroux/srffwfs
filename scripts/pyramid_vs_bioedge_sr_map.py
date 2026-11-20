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
from srffwfs.closed_loop import close_the_loop, ClosedLoopResult
from srffwfs.miscellaneous import pad_array, crop_array

# %% Class definitions


@dataclass
class CalibratedWfs:
    name: str
    wfs: Pyramid | BioEdge
    interaction_matrix: np.ndarray


# %% Fucntions definitions


def compute_weighted_truncated_svd_reconstructor(
    interaction_matrix, c_phi, n_controlled_modes
):
    L = np.linalg.cholesky(c_phi)
    A = interaction_matrix @ L
    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :n_controlled_modes]
    s_k = singular_values[:n_controlled_modes]
    Vh_k = Vh[:n_controlled_modes, :]
    reconstructor = L @ Vh_k.T / s_k @ U_k.T

    return reconstructor, singular_values


def compute_lse_reconstructor(interaction_matrix, n_controlled_modes):
    reconstructor = np.linalg.pinv(interaction_matrix[:, :n_controlled_modes])
    reconstructor = np.concatenate(
        (
            reconstructor,
            np.zeros(
                (
                    interaction_matrix.shape[1] - n_controlled_modes,
                    reconstructor.shape[1],
                )
            ),
        ),
        axis=0,
    )
    return reconstructor


def compute_map_reconstructor(interaction_matrix, c_phi, c_b, rcond=None):
    """
    MAP modal reconstructor  C_phi Dᵀ (D C_phi Dᵀ + C_b)⁺

    c_b can be a scalar, a (n_signals,) vector or a (n_signals, n_signals)
    matrix, and may be zero (fully or partially). In that case S is
    rank-deficient and the pseudo-inverse is used.

    rcond : eigenvalues of S below rcond * max(eigenvalue) are discarded.
            Default: n_signals * machine epsilon of D's dtype.
    """
    D = interaction_matrix
    n_signals = D.shape[0]

    # noise covariance as a full matrix, whatever the input form
    c_b = np.asarray(c_b, dtype=D.dtype)
    if c_b.ndim == 0:
        c_b = c_b * np.eye(n_signals, dtype=D.dtype)
    elif c_b.ndim == 1:
        c_b = np.diag(c_b)

    D_c_phi = D @ c_phi
    S = D_c_phi @ D.T + c_b
    S = 0.5 * (S + S.T)  # remove round-off asymmetry

    # pseudo-inverse of the symmetric PSD matrix S
    if rcond is None:
        rcond = n_signals * np.finfo(D.dtype).eps
    eigvals, eigvecs = np.linalg.eigh(S)
    keep = eigvals > rcond * eigvals.max()
    S_pinv = (eigvecs[:, keep] / eigvals[keep]) @ eigvecs[:, keep].T

    # C_phi Dᵀ S⁺ = (S⁺ D C_phi)ᵀ since S and C_phi are symmetric
    return (S_pinv @ D_c_phi).T


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

r0 = 0.15  # [m] value of r0 at 500 nm
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

# ------------------------ DM ---------------------- #

n_actuator = 2 * n_subaperture  # number of actuators

# ----------------------- WFS ---------------------- #

modulation = 3.0  # [lambda/D] modulation radius or half grey width
n_pix_separation = 2  # [pixel] separation ratio between the pupils
light_threshold = (
    0.3 if modulation > 0.0 else 0
)  # light threshold to select the valid pixels
detector_photon_noise = False
detector_read_out_noise = 3.0  # e- RMS

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
n_iter = 200
delay = 1
loop_frequency = 1000  # [Hz] loop frequency
n_controlled_modes = int(
    2.0 * np.pi * (n_subaperture / 2) ** 2
)  # number of controlled modes
use_weighted_svd = True  # use weighted truncated SVD reconstructor

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

# define a dictionary to loop over the different pupil shifts configurations
pupil_shifts_dict = {
    "zeros": pupil_shifts_zeros,
    "horizontal": pupil_shifts_horizontal,
    "vertical": pupil_shifts_vertical,
    "hv": pupil_shifts_hv,
    "quincux": pupil_shifts_quincux,
}


# %% Precompute shifted and calibrated WFS for every configuration.

calibrated_pyramids: dict[str, CalibratedWfs] = {}
calibrated_bioedges: dict[str, CalibratedWfs] = {}

for shift_name, pupil_shifts in tqdm(pupil_shifts_dict.items()):

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

    pyramid.apply_shift_wfs(pupil_shifts[0], pupil_shifts[1], units="pixels")
    pyramid.modulation = modulation  # update reference intensities

    bioedge.apply_shift_wfs(
        pupil_shifts[0], pupil_shifts[1], units="pixels"
    )  # to be redifined
    bioedge.modulation = 0

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

    calibrated_pyramids[shift_name] = CalibratedWfs(
        name=shift_name,
        wfs=pyramid,
        interaction_matrix=interaction_matrix_pyramid,
    )

    calibrated_bioedges[shift_name] = CalibratedWfs(
        name=shift_name,
        wfs=bioedge,
        interaction_matrix=interaction_matrix_bioedge,
    )

# %% Closed-loop tests

modal_projector = None
seed = 12
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
        reconstructor = compute_map_reconstructor(
            interaction_matrix,
            c_phi,
            (detector_read_out_noise / n_photons_per_frame) ** 2,
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
        reconstructor = compute_map_reconstructor(
            interaction_matrix,
            c_phi,
            (detector_read_out_noise / n_photons_per_frame) ** 2,
        )

    else:
        reconstructor = compute_lse_reconstructor(
            interaction_matrix, n_controlled_modes
        )

    reference_intensities_2d_sr_bioedge = calibrated_bioedge.wfs.referenceSignal_2D
    valid_signal_2d_sr_bioedge = calibrated_bioedge.wfs.valid_signal_2D

    plt.figure()
    plt.imshow(reference_intensities_2d_sr_bioedge)
    plt.title(f"Reference intensities\nsuper resolved pyramid\n{shift_name} shifts")

    plt.figure()
    plt.imshow(
        reference_intensities_2d_sr_bioedge - reference_intensities_2d_no_sr_bioedge
    )
    plt.title(
        f"Reference intensities difference\nsuper resolved pyramid - pyramid\n{shift_name} shifts"
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

# %%

plt.show()
