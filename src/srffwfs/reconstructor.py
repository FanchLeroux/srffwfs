import numpy as np


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
