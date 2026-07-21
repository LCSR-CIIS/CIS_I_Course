# Author: X.M. Christine Zhu

"""
Shared metric helpers for comparing predicted vs. empirical SE(3) covariances.

Used by analyze_extended.py (exp1) and intended to be reusable by exp2/exp3.
All blocks are taken from 6x6 tangent-space covariances in [alpha; epsilon]
ordering (rotation first, then translation), matching se3_stats.py.
"""

import numpy as np

ROT_BLOCK = slice(0, 3)
TRANS_BLOCK = slice(3, 6)


def tip_point_covariance(mean_T: np.ndarray, C: np.ndarray,
                         p_tip_local: np.ndarray) -> np.ndarray:
    """
    Propagate SE(3) covariance C of a transform T to 3×3 tip position covariance.

    With right-mult perturbation T_true = T_nom @ Exp([α; ε]):
        Δq ≈ [-R·skew(p_tip), R] · [α; ε]
        C_tip = J @ C @ J^T          (3×3)

    The result is the covariance of the tip point expressed in the frame that
    T maps *into* (e.g. if T is drill→Anatomy, the tip covariance is in the
    Anatomy frame).

    Parameters
    ----------
    mean_T     : (4, 4) — nominal transform
    C          : (6, 6) — SE(3) covariance, [α; ε] (rotation-first) ordering
    p_tip_local : (3,)  — tip offset in the source frame of T (metres)
    """
    R = mean_T[:3, :3]
    v = p_tip_local
    skew_p = np.array([[0, -v[2], v[1]],
                       [v[2],  0, -v[0]],
                       [-v[1], v[0],  0]])
    J = np.hstack([-R @ skew_p, R])   # 3×6
    return J @ C @ J.T


def block_compare(C_pred: np.ndarray, C_emp: np.ndarray, idx: slice) -> dict:
    """
    Compare a 3x3 diagonal block of predicted vs. empirical 6x6 covariance.

    Parameters
    ----------
    C_pred, C_emp : ndarray, shape (6, 6)
    idx : slice — ROT_BLOCK or TRANS_BLOCK

    Returns
    -------
    dict with sigma_pred, sigma_emp (rad or m, raw units), frob_error,
    rel_error_pct, restricted to the given block.
    """
    block_pred = C_pred[idx, idx]
    block_emp = C_emp[idx, idx]
    sigma_pred = float(np.sqrt(np.trace(block_pred) / 3.0))
    sigma_emp = float(np.sqrt(np.trace(block_emp) / 3.0))
    frob_error = float(np.linalg.norm(block_pred - block_emp, "fro"))
    rel_error_pct = frob_error / (np.linalg.norm(block_emp, "fro") + 1e-30) * 100.0
    return {
        "sigma_pred": sigma_pred,
        "sigma_emp": sigma_emp,
        "frob_error": frob_error,
        "rel_error_pct": rel_error_pct,
    }


def cross_block_compare(C_pred: np.ndarray, C_emp: np.ndarray) -> dict:
    """
    Compare the rotation-translation cross-coupling block C[:3, 3:].

    Returns dict with frob_error, rel_error_pct.
    """
    cross_pred = C_pred[ROT_BLOCK, TRANS_BLOCK]
    cross_emp = C_emp[ROT_BLOCK, TRANS_BLOCK]
    frob_error = float(np.linalg.norm(cross_pred - cross_emp, "fro"))
    rel_error_pct = frob_error / (np.linalg.norm(cross_emp, "fro") + 1e-30) * 100.0
    return {"frob_error": frob_error, "rel_error_pct": rel_error_pct}


def ellipsoid_shape_compare(C_pred: np.ndarray, C_emp: np.ndarray, idx: slice) -> dict:
    """
    Compare the shape (principal axes + radii) of the uncertainty ellipsoids
    described by a 3x3 diagonal block of predicted vs. empirical covariance.

    Eigenvalues are sorted ascending by np.linalg.eigh; axes are compared
    pairwise in that order.

    Returns
    -------
    dict with:
        sigma_ratio : ndarray, shape (3,) — sqrt(eigval_pred)/sqrt(eigval_emp)
            per principal axis, ideally close to 1.0
        axis_angle_deg : ndarray, shape (3,) — angle between corresponding
            eigenvectors in degrees, ideally close to 0
    """
    block_pred = C_pred[idx, idx]
    block_emp = C_emp[idx, idx]

    eigval_pred, eigvec_pred = np.linalg.eigh(block_pred)
    eigval_emp, eigvec_emp = np.linalg.eigh(block_emp)

    eigval_pred = np.maximum(eigval_pred, 1e-30)
    eigval_emp = np.maximum(eigval_emp, 1e-30)
    sigma_ratio = np.sqrt(eigval_pred) / np.sqrt(eigval_emp)

    axis_angle_deg = np.zeros(3)
    for k in range(3):
        cos_angle = np.clip(np.abs(eigvec_pred[:, k] @ eigvec_emp[:, k]), 0.0, 1.0)
        axis_angle_deg[k] = np.degrees(np.arccos(cos_angle))

    return {"sigma_ratio": sigma_ratio, "axis_angle_deg": axis_angle_deg}
