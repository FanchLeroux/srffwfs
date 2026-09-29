import numpy as np


def compute_eigen_control_basis(
    calibration_basis: np.ndarray,
    interaction_matrix: np.ndarray,
    n_lo_modes_to_keep: int | None = None,
) -> np.ndarray:
    """
    Compute the controll basis of the system from the interaction matrix using SVD decomposition.
    Optionally, a given number of low order modes can be kept in the final control basis.

    Parameters
    ----------
    calibration_basis : np.ndarray, shape: (n_pixels, n_modes)
        The modal basis used to aquire the interaction matrix.
    interaction_matrix : np.ndarray, shape: (n_pixels, n_modes)
        The interaction matrix of the system.
    n_lo_modes_to_keep : int | None, optional
        The number of low order modes to keep in the final control basis. If None, no modes are kept. The default is None.

    Returns
    -------
    control_basis: np.ndarray
        The controll basis of the system.
    """

    calibration_basis_lo = calibration_basis[:, :n_lo_modes_to_keep]
    calibration_basis_ho = calibration_basis[:, n_lo_modes_to_keep:]
    interaction_matrix_flat_ho = interaction_matrix[:, n_lo_modes_to_keep:]

    u_ho, s_ho, vt_ho = np.linalg.svd(interaction_matrix_flat_ho, full_matrices=False)

    eigen_modes_ho = calibration_basis_ho @ vt_ho.T

    control_basis = np.concatenate(
        [calibration_basis_lo, eigen_modes_ho],
        axis=1,
    )

    return control_basis
