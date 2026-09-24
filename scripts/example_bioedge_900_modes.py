# %% -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:37:17 2025

@author: fleroux
"""

import pathlib
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from srffwfs.config import Config
from srffwfs import oopao_config

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

# %%

config = Config()
fig_dir = config.root_dir / "outputs"

# %% define functions


def close_the_loop(
    tel,
    ngs,
    atm,
    dm,
    wfs,
    reconstructor,
    loop_gain,
    n_iter=100,
    delay=1,
    photon_noise=False,
    read_out_noise=0.0,
    seed=0,
    save_telemetry=False,
    save_psf=False,
    display=False,
):

    wfs.cam.photonNoise = photon_noise
    wfs.cam.readoutNoise = read_out_noise

    ngs * tel
    tel.computePSF()  # just to get the shape

    # Memory allocation

    total = np.zeros(n_iter)  # turbulence phase std [nm]
    residual = np.zeros(n_iter)  # residual phase std [nm]
    strehl = np.zeros(n_iter)  # Strehl Ratio

    buffer_wfs_measure = np.zeros([wfs.signal.shape[0]] + [delay])

    if save_telemetry:
        dm_coefs = np.zeros([dm.nValidAct, n_iter])
        turbulence_phase_screens = np.zeros(
            [tel.OPD.shape[0], tel.OPD.shape[1]] + [n_iter]
        )
        residual_phase_screens = np.zeros(
            [tel.OPD.shape[0], tel.OPD.shape[1]] + [n_iter]
        )
        wfs_frames = np.zeros(
            [wfs.cam.frame.shape[0], wfs.cam.frame.shape[1]] + [n_iter]
        )
        wfs_signals = np.zeros([wfs.signal.shape[0]] + [n_iter])

    if save_psf:
        short_exposure_psf = np.zeros([tel.PSF.shape[0], tel.PSF.shape[1]] + [n_iter])

    # initialization

    atm.initializeAtmosphere(tel)
    atm.generateNewPhaseScreen(seed=seed)
    tel + atm

    dm.coefs = 0

    ngs * tel * dm * wfs

    # close the loop

    for k in tqdm(range(n_iter)):

        atm.update()
        total[k] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9  # [nm]

        if save_telemetry:
            turbulence_phase_screens[:, :, k] = tel.OPD

        ngs * tel * dm * wfs

        buffer_wfs_measure = np.roll(buffer_wfs_measure, -1, axis=1)
        buffer_wfs_measure[:, -1] = wfs.signal

        if save_telemetry:

            residual_phase_screens[:, :, k] = tel.OPD
            dm_coefs[:, k] = dm.coefs
            wfs_frames[:, :, k] = wfs.cam.frame
            wfs_signals[:, k] = buffer_wfs_measure[:, 0]

        residual[k] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9  # [nm]
        strehl[k] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil > 0)]))

        dm.coefs = dm.coefs - loop_gain * reconstructor @ buffer_wfs_measure[:, 0]

        if save_psf:

            tel.computePSF()
            short_exposure_psf[:, :, k] = tel.PSF

    # return

    if save_telemetry and save_psf:
        return (
            total,
            residual,
            strehl,
            dm_coefs,
            turbulence_phase_screens,
            residual_phase_screens,
            wfs_frames,
            wfs_signals,
            short_exposure_psf,
        )
    elif save_telemetry:
        return (
            total,
            residual,
            strehl,
            dm_coefs,
            turbulence_phase_screens,
            residual_phase_screens,
            wfs_frames,
        )
    elif save_psf:
        total, residual, strehl, short_exposure_psf
    else:
        return total, residual, strehl


# %% Define parameters


# initialize the dictionary
param = {}

# fill the dictionary
# ------------------ ATMOSPHERE ----------------- #

param["r0"] = 0.15  # [m] value of r0 in the visibile
param["L0"] = 30  # [m] value of L0 in the visibile
param["fractionnal_r0"] = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
param["wind_speed"] = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of  layers
param["wind_direction"] = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
param["altitude"] = [0, 1000, 5000, 10000, 12000]  # [m] altitude of layers
param["seeds"] = range(1)

# ------------------- TELESCOPE ------------------ #

param["diameter"] = 8  # [m] telescope diameter
param["n_subaperture"] = 20  # number of WFS subaperture along the
# telescope diameter
# [pixel] sampling of the WFS subapertures
param["n_pixel_per_subaperture"] = 8
# in telescope pupil space
param["resolution"] = (
    param["n_subaperture"] * param["n_pixel_per_subaperture"]
)  # resolution of the telescope driven by
# the WFS
param["size_subaperture"] = (
    param["diameter"] / param["n_subaperture"]
)  # [m] size of a subaperture projected in M1 space
param["sampling_time"] = 1 / 1000  # [s] loop sampling time
param["centralObstruction"] = 0  # central obstruction in percentage
# of the diameter

# ---------------------- NGS ---------------------- #

param["magnitude"] = 8  # magnitude of the guide star

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
param["optical_band"] = "R4"  # optical band of the guide star

# ------------------------ DM --------------------- #

param["n_actuator"] = 2 * param["n_subaperture"]  # number of actuators

# ----------------------- WFS ---------------------- #

param["modulation"] = 5.0  # [lambda/D] modulation radius or half grey width
param["grey_length"] = param["modulation"]  # [lambda/D] grey length in case of
# small grey bioedge WFS
param["n_pix_separation"] = 10  # [pixel] separation ratio between the pupils
param["psf_centering"] = False  # centering of the FFT and of the mask on
# the 4 central pixels
param["light_threshold"] = 0.3  # light threshold to select the valid pixels
param["post_processing"] = "fullFrame"  # post-processing of the WFS signals
# ('slopesMaps' or 'fullFrame')
param["detector_photon_noise"] = False
param["detector_read_out_noise"] = 0.0  # e- RMS

# super resolution
param["sr_amplitude"] = 0.25  # [pixel] super resolution shifts amplitude

# [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)
param["pupil_shift_bioedge"] = [
    [
        param["sr_amplitude"],
        -param["sr_amplitude"],
        param["sr_amplitude"],
        -param["sr_amplitude"],
    ],
    [
        param["sr_amplitude"],
        -param["sr_amplitude"],
        -param["sr_amplitude"],
        param["sr_amplitude"],
    ],
]

# -------------------- CALIBRATION - MODAL BASIS ---------------- #

param["modal_basis"] = "KL"
# [m] actuator stroke for interaction matrix computation
param["stroke"] = 1e-9
param["single_pass"] = False  # push-pull or push only for the calibration
param["compute_M2C_Folder"] = str(pathlib.Path(__file__).parent)

# ----------------------- RECONSTRUCTION ------------------------ #

param["mmse_noise_level_guess_slopes_maps"] = (
    10e-9  # noise level assumption for MMSE reconstruction
)
param["mmse_noise_level_guess_full_frame"] = (
    5e-11  # noise level assumption for MMSE full frame reconstruction
)
param["mmse_alpha"] = (
    1.0  # Weight for the turbulence statistics for MMSE reconstruction
)

# -------------------- LOOP ----------------------- #

param["n_modes_to_show_lse_sr"] = 900

param["loop_gain"] = 0.5

param["n_iter"] = 200

param["delay"] = 2

# --------------------- FILENAME -------------------- #

# name of the system
param["filename"] = (
    "_"
    + param["optical_band"]
    + "_band_"
    + str(param["n_subaperture"])
    + "x"
    + str(param["n_subaperture"])
    + "_"
    + param["modal_basis"]
    + "_basis"
)

# %% Build objects

# % -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(
    resolution=param["resolution"],  # [pixel] resolution of
    # the telescope
    diameter=param["diameter"],
)  # [m] telescope diameter

# % -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(
    optBand=param["optical_band"],  # Source optical band
    # (see photometry.py)
    magnitude=param["magnitude"],
)  # Source Magnitude

# % -----------------------    ATMOSPHERE   ----------------------------

# coupling telescope and source is mandatory to generate Atmosphere object
ngs * tel

# create the Atmosphere object
atm = Atmosphere(
    telescope=tel,  # Telescope
    r0=param["r0"],  # Fried Parameter [m]
    L0=param["L0"],  # Outer Scale [m]
    # Cn2 Profile (percentage)
    fractionalR0=param["fractionnal_r0"],
    windSpeed=param["wind_speed"],  # [m.s-1] wind speed of layers
    # [degrees] wind direction
    windDirection=param["wind_direction"],
    # of layers
    altitude=param["altitude"],
)  # [m] altitude of layers

# %% -------------------------     DM   ----------------------------------

dm = DeformableMirror(tel, nSubap=param["n_actuator"])

# %% ------------------------- MODAL BASIS -------------------------------

if param["modal_basis"] == "KL":

    M2C_KL_full, HHt, PSD_atm, df = compute_M2C(
        telescope=tel,
        atmosphere=atm,
        deformableMirror=dm,
        param=param,
        nameFolder=param["compute_M2C_Folder"],
        remove_piston=False,
        HHtName="KL_covariance_matrix",
        baseName="KL_basis",
        mem_available=6.1e9,
        minimF=False,
        nmo=None,
        ortho_spm=True,
        SZ=np.int64(2 * tel.OPD.shape[0]),
        nZer=3,
        NDIVL=1,
        lim_inversion=1e-16,
        returnHHt_PSD_df=True,
        save_output=False,
    )

    M2C = M2C_KL_full[:, 1:]  # remove piston

    dm.coefs = np.zeros(dm.nValidAct)  # reset dm.OPD

elif param["modal_basis"] == "poke":
    M2C = np.identity(dm.nValidAct)

# %% ----------------------- Grey Bi-O-Edge ---------------------------- #

# super resolved grey bioedge
gbioedge_sr = BioEdge(
    nSubap=param["n_subaperture"],
    telescope=tel,
    modulation=0.0,
    grey_width=param["modulation"],
    lightRatio=param["light_threshold"],
    n_pix_separation=param["n_pix_separation"],
    postProcessing="fullFrame",
)

gbioedge_sr.apply_shift_wfs(
    param["pupil_shift_bioedge"][0], param["pupil_shift_bioedge"][1], units="pixels"
)
gbioedge_sr.modulation = 0.0  # update reference intensities etc.

# %% Calibration

calib_sr = InteractionMatrix(
    ngs,
    tel,
    dm,
    gbioedge_sr,
    M2C=M2C,
    stroke=param["stroke"],
    single_pass=param["single_pass"],
    noise="off",
    display=True,
)

# %% LSE Reconstructor computation

R = np.linalg.pinv(calib_sr.D[:, : param["n_modes_to_show_lse_sr"]])

reconstructor_lse_sr = M2C[:, : param["n_modes_to_show_lse_sr"]] @ R

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
    dm,
    gbioedge_sr,
    reconstructor_lse_sr,
    param["loop_gain"],
    param["n_iter"],
    delay=param["delay"],
    photon_noise=param["detector_photon_noise"],
    read_out_noise=param["detector_read_out_noise"],
    seed=seed,
    save_telemetry=True,
    save_psf=True,
    display=False,
)

# %% post processing

long_exposure_psf_lse_sr = np.sum(short_exposure_psf_lse_sr[:, :, 100:], axis=2)

# %% plots

# noise propagation
plt.figure()
plt.plot(np.diag(R @ R.T) / gbioedge_sr.nSignal)
plt.yscale("log")
plt.title("modal uniform noise propagation")
plt.xlabel("# modes")
plt.savefig(fig_dir / pathlib.Path("noise_propagation" + ".png"), bbox_inches="tight")

# %%

# residuals
plt.figure()
plt.plot(total_lse_sr, label="total_lse_sr")
plt.plot(residual_lse_sr, label="residual_lse_sr")
plt.xlabel("loop iteration")
plt.ylabel("residual phase RMS [nm]")
plt.title("Closed Loop residuals")
plt.legend()
plt.savefig(fig_dir / pathlib.Path("residuals" + ".png"), bbox_inches="tight")

# %%

# strehls
plt.figure()
plt.plot(strehl_lse_sr, label="strehl_lse_sr")
plt.ylabel("strehl phase RMS [nm]")
plt.title("Closed Loop strehls")
plt.legend()
plt.savefig(fig_dir / pathlib.Path("strehls" + ".png"), bbox_inches="tight")

# %%

# long exposure PSF
plt.figure()
plt.imshow(np.log(long_exposure_psf_lse_sr))
plt.title("long_exposure_psf_lse_sr")
plt.savefig(fig_dir / pathlib.Path("long_exposure_psf" + ".png"), bbox_inches="tight")

plt.show()
