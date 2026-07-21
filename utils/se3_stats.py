# Author: X.M. Christine Zhu

"""
SE(3) empirical statistics utilities.

Used by all three experiments to turn a batch of raw tracker measurements
(N × 4×4 homogeneous matrices) into a Fréchet mean and 6×6 covariance.

Convention: twist ordering is [alpha; epsilon] (rotation first, then translation),
matching the CIS I right-perturbation convention used throughout this project.
"""

import sys
import pathlib
import numpy as np
from typing import Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from uncertainty_networks.se3 import log_se3, inv_se3, exp_se3


def se3_mean(samples: np.ndarray, max_iter: int = 100, tol: float = 1e-12) -> np.ndarray:
    """
    Fréchet mean of N SE(3) samples via iterative right-geodesic update.

    Parameters
    ----------
    samples : ndarray, shape (N, 4, 4)
    max_iter : int
    tol : float  convergence threshold on the update norm

    Returns
    -------
    mu : ndarray, shape (4, 4)  — Fréchet mean transform
    """
    mu = samples[0].copy()
    for _ in range(max_iter):
        residuals = np.array([log_se3(inv_se3(mu) @ T) for T in samples])
        delta = residuals.mean(axis=0)
        if np.linalg.norm(delta) < tol:
            break
        mu = mu @ exp_se3(delta)
    return mu


def se3_empirical_stats(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical Fréchet mean and 6×6 tangent-space covariance.

    Parameters
    ----------
    samples : ndarray, shape (N, 4, 4)
        Batch of SE(3) measurements.

    Returns
    -------
    mean_F : ndarray, shape (4, 4)
        Fréchet mean.
    C : ndarray, shape (6, 6)
        Sample covariance in [alpha; epsilon] tangent-space coordinates.
    """
    if samples.ndim != 3 or samples.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N,4,4), got {samples.shape}")

    mean_F = se3_mean(samples)
    residuals = np.array([log_se3(inv_se3(mean_F) @ T) for T in samples])
    N = len(residuals)
    C = (residuals.T @ residuals) / (N - 1)
    return mean_F, 0.5 * (C + C.T)


def load_poses_csv(path: str) -> np.ndarray:
    """
    Load a CSV of flattened 4×4 row-major matrices.

    Each row of the CSV is 16 floats: the 4×4 matrix read row by row.
    Returns ndarray of shape (N, 4, 4).
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] != 16:
        raise ValueError(f"Expected 16 columns (flattened 4x4), got {data.shape[1]}")
    return data.reshape(-1, 4, 4)


def save_poses_csv(path: str, poses: np.ndarray) -> None:
    """
    Save (N, 4, 4) array as a CSV of 16-column rows (row-major flattened).
    """
    np.savetxt(path, poses.reshape(-1, 16), delimiter=",",
               header="T00,T01,T02,T03,T10,T11,T12,T13,T20,T21,T22,T23,T30,T31,T32,T33",
               comments="")


def summary_stats(mean_F: np.ndarray, C: np.ndarray) -> dict:
    """
    Return a dict of human-readable summary statistics.

    Keys
    ----
    sigma_rot_deg     : float   — rotational std (deg) = sqrt(trace(C[:3,:3])/3)
    sigma_trans_mm    : float   — translational std (mm) = sqrt(trace(C[3:,3:])/3)
    distance_mm       : float   — distance from origin to mean translation (mm)
    """
    sigma_rot_rad = np.sqrt(np.trace(C[:3, :3]) / 3.0)
    sigma_trans_m = np.sqrt(np.trace(C[3:, 3:]) / 3.0)
    dist = np.linalg.norm(mean_F[:3, 3])
    return {
        "sigma_rot_deg":   float(np.degrees(sigma_rot_rad)),
        "sigma_trans_mm":  float(sigma_trans_m * 1000.0),
        "distance_mm":     float(dist * 1000.0),
    }
