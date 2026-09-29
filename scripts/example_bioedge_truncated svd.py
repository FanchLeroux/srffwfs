# %% imports

import pathlib
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
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

from srffwfs.sensitivity import compute_photon_noise_sensitivity
from srffwfs.closed_loop import close_the_loop

# %% functions definitions


@lru_cache(maxsize=None)
def compute_KL_basis(tel, atm, dm):

    M2C_KL_full, HHt, PSD_atm, df = compute_M2C(
        telescope=tel,
        atmosphere=atm,
        deformableMirror=dm,
        param=None,
        nameFolder=None,
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

    return M2C


# %%

config = Config()
fig_dir = config.root_dir / "outputs"

# %% Define parameters


# initialize the dictionary
param = {}

# fill the dictionary

# ---------------------- NGS ---------------------- #

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
wavelength = 670e-9  # [m] wavelength of the guide star
param["optical_band"] = "R4"  # optical band of the guide star
param["magnitude"] = 8  # magnitude of the guide star

# ------------------ ATMOSPHERE ----------------- #

param["r0"] = 0.1  # [m] value of r0 at 500 nm
param["L0"] = 30  # [m] value of L0 in the visibile
param["fractionnal_r0"] = [0.45, 0.1, 0.1, 0.25, 0.1]  # Cn2 profile (percentage)
param["wind_speed"] = [5, 4, 8, 10, 2]  # [m.s-1] wind speed of  layers
param["wind_direction"] = [0, 72, 144, 216, 288]  # [degrees] wind direction of layers
param["altitude"] = [0, 1000, 5000, 10000, 12000]  # [m] altitude of layers
param["seeds"] = range(1)

# ------------------- TELESCOPE ------------------ #

param["diameter"] = 2  # [m] telescope diameter
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

# ------------------------ DM --------------------- #

param["n_actuator"] = 2 * param["n_subaperture"]  # number of actuators

# ----------------------- WFS ---------------------- #

param["modulation"] = 2.0  # [lambda/D] modulation radius or half grey width
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
stroke_rad = 0.01  # [rad]
param["stroke"] = stroke_rad * wavelength / (2 * np.pi)  # [nm]
param["single_pass"] = False  # push-pull or push only for the calibration
param["compute_M2C_Folder"] = str(pathlib.Path(__file__).parent)

# ----------------------- RECONSTRUCTION ------------------------ #

param["n_modes_to_show_lse_sr"] = 900

# -------------------- LOOP ----------------------- #

param["loop_gain"] = 0.7

param["n_iter"] = 200

param["delay"] = 1

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
    M2C = compute_KL_basis(tel, atm, dm)

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

# %% Sensitivity analysis

interaction_matrix_rad_normalized = calib_sr.D * wavelength / (2 * np.pi)
reference_intensities = gbioedge_sr.referenceSignal

photon_noise_sensitivity = compute_photon_noise_sensitivity(
    interaction_matrix_rad_normalized, reference_intensities
)

fig_sensitivity, ax_sensitivity = plt.subplots()
ax_sensitivity.plot(photon_noise_sensitivity)
ax_sensitivity.axhline(y=2**0.5, color="k", linestyle="--", label=r"$\sqrt{2}$")
ax_sensitivity.set_xlabel("# mode")
ax_sensitivity.set_ylabel(r"S_{ph}")
ax_sensitivity.set_xscale("log")
ax_sensitivity.set_yscale("log")
ax_sensitivity.legend(loc="lower left")


# %% SVD

u, s, vt = np.linalg.svd(calib_sr.D, full_matrices=False)

plt.figure()
plt.plot(s, label=f"cond: {s[0]/s[-1]:.2e}")
plt.yscale("log")
plt.title("Singular values of the interaction matrix")
plt.xlabel("# eigen mode")
plt.ylabel("Singular values [a.u.]")
plt.legend(loc="upper right")

# %% extract eigen modes

influence_functions_2d = dm.modes.T.reshape(
    dm.nValidAct, tel.OPD.shape[0], tel.OPD.shape[1]
)
influence_functions_flat = influence_functions_2d[:, tel.pupil].T

calibration_basis_flat = influence_functions_flat @ M2C
calibration_basis_2d = np.zeros((M2C.shape[1], tel.OPD.shape[0], tel.OPD.shape[1]))
calibration_basis_2d[:, tel.pupil] = calibration_basis_flat.T

eigen_modes_flat = calibration_basis_flat @ vt.T
eigen_modes_2d = np.zeros((vt.shape[0], tel.OPD.shape[0], tel.OPD.shape[1]))
eigen_modes_2d[:, tel.pupil] = eigen_modes_flat.T

# %% show eigen modes

eigenmodes_indices = [0, 100, 1000, -1]

fig_eigen_modes, axes_eigen_modes = plt.subplots(2, 2, constrained_layout=True)
axes_eigen_modes[0, 0].imshow(eigen_modes_2d[eigenmodes_indices[0]])
axes_eigen_modes[0, 0].set_title(f"Eigen mode {eigenmodes_indices[0]}")
axes_eigen_modes[0, 1].imshow(eigen_modes_2d[eigenmodes_indices[1]])
axes_eigen_modes[0, 1].set_title(f"Eigen mode {eigenmodes_indices[1]}")
axes_eigen_modes[1, 0].imshow(eigen_modes_2d[eigenmodes_indices[2]])
axes_eigen_modes[1, 0].set_title(f"Eigen mode {eigenmodes_indices[2]}")
axes_eigen_modes[1, 1].imshow(eigen_modes_2d[eigenmodes_indices[3]])
axes_eigen_modes[1, 1].set_title(f"Eigen mode {eigenmodes_indices[3]}")
fig_eigen_modes.suptitle(
    "Eigen modes of the interaction matrix\n"
    rf"grey width: {param['modulation']} $\lambda/D$"
)

# %% compute controll basis using SVD eigenmodes while keeping low order modes

n_modes_to_keep = 100  # number of low order modes to keep in the final control basis

calibration_basis_flat_ho = calibration_basis_flat[:, n_modes_to_keep:]
interaction_matrix_flat_ho = calib_sr.D[:, n_modes_to_keep:]

u_ho, s_ho, vt_ho = np.linalg.svd(interaction_matrix_flat_ho, full_matrices=False)

eigen_modes_flat_ho = calibration_basis_flat_ho @ vt_ho.T
eigen_modes_2d_ho = np.zeros((vt_ho.shape[0], tel.OPD.shape[0], tel.OPD.shape[1]))
eigen_modes_2d_ho[:, tel.pupil] = eigen_modes_flat_ho.T

# %% show eigen modes ho

eigenmodes_indices_ho = [0, 100, 1000, -1]

fig_eigen_modes_ho, axes_eigen_modes_ho = plt.subplots(2, 2, constrained_layout=True)
axes_eigen_modes_ho[0, 0].imshow(eigen_modes_2d_ho[eigenmodes_indices_ho[0]])
axes_eigen_modes_ho[0, 0].set_title(f"Eigen mode {eigenmodes_indices_ho[0]}")
axes_eigen_modes_ho[0, 1].imshow(eigen_modes_2d_ho[eigenmodes_indices_ho[1]])
axes_eigen_modes_ho[0, 1].set_title(f"Eigen mode {eigenmodes_indices_ho[1]}")
axes_eigen_modes_ho[1, 0].imshow(eigen_modes_2d_ho[eigenmodes_indices_ho[2]])
axes_eigen_modes_ho[1, 0].set_title(f"Eigen mode {eigenmodes_indices_ho[2]}")
axes_eigen_modes_ho[1, 1].imshow(eigen_modes_2d_ho[eigenmodes_indices_ho[3]])
axes_eigen_modes_ho[1, 1].set_title(f"Eigen mode {eigenmodes_indices_ho[3]}")
fig_eigen_modes_ho.suptitle(
    "Eigen modes ho of the interaction matrix\n"
    rf"grey width: {param['modulation']} $\lambda/D$"
    rf"\nkeeping {n_modes_to_keep} low order modes in the final control basis"
)

# %% Final controll basis

full_final_control_basis_2d = np.concatenate(
    [
        calibration_basis_2d[:n_modes_to_keep, :, :],
        eigen_modes_2d_ho,
    ],
    axis=0,
)

final_control_basis_2d = full_final_control_basis_2d[
    : param["n_modes_to_show_lse_sr"], :, :
]

final_control_basis_flat = final_control_basis_2d[:, tel.pupil].T

# %% LSE Reconstructor computation - KL truncation

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

# %%
