from tqdm import tqdm
import numpy as np


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
