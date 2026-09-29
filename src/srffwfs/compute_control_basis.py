import numpy as np


def compute_eigen_control_basis(interaction_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the controll basis of the system from the interaction matrix using SVD decomposition.
    Optionally, a given number of low order modes can be kept in the final control basis.

    Parameters
    ----------
    interaction_matrix : np.ndarray
        The interaction matrix of the system.

    Returns
    -------
    np.ndarray
        The controll basis of the system.
    """
    u, s, vt = np.linalg.svd(interaction_matrix, full_matrices=False)
    return vt.T
