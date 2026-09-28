from tqdm import tqdm
import numpy as np


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
        return total, residual, strehl, short_exposure_psf
    else:
        return total, residual, strehl
