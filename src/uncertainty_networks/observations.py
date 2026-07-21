# Author: X.M. Christine Zhu
# Date: 03/28/2026

"""
Observation and Factor Abstraction for SE(3) uncertainty networks.

Motivation
----------
The existing closed_loop module handles loop constraints directly, with the
residual and Jacobian hard-coded to the loop geometry.  This module introduces
a general *Observation* interface so that any measurement — a loop closure, an
observed point position, an observed distance, or any future sensor modality —
can be expressed in the same linear-Gaussian form and conditioned upon jointly.

Measurement model (shared by all observation types)
----------------------------------------------------
Each Observation constrains a set of named state perturbations

    eta_k ~ N(0, C_k),  k in state_keys

through a linearized residual:

    r(eta) ≈ r0 + sum_k  J_k @ eta_k  ≈  0 + nu,   nu ~ N(0, C_nu)

where:
    r0    : residual at zero perturbation (nominal)         shape (m,)
    J_k   : Jacobian of residual w.r.t. eta_k              shape (m, 6)
    C_nu  : observation noise covariance                   shape (m, m)

Joint conditioning  (condition_on_observations)
-----------------------------------------------
Given a prior  {state_key: C_k}  and a list of Observations, the posterior
covariance is obtained via the information filter:

    C0     = block_diag({C_k})                             shape (6K, 6K)
    H      = [row-blocks of each observation Jacobian]     shape (M, 6K)
    C_nu   = block_diag({C_nu_i})                          shape (M, M)

    C_post = ( C0^{-1} + H^T C_nu^{-1} H )^{-1}           shape (6K, 6K)

The posterior covariance for each state key is extracted from C_post.

Concrete observation types
--------------------------
LoopObservation     : loop-closure residual Log(T_res^{-1} T_k) ≈ 0   (6-dim)
PointObservation    : observed 3D point position in a frame            (3-dim)
DistanceObservation : observed scalar distance between two points      (1-dim)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .se3 import skew
from .closed_loop import (
    linearize_loop_residual,
    LoopLinearization,
    _sym,
    _assert_positive_definite,
)

Array = np.ndarray


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Observation(ABC):
    """
    Abstract base for a linearized measurement that constrains named state variables.

    Subclasses must implement:
        residual()   -> (m,) ndarray        nominal residual at zero perturbation
        jacobians()  -> {key: (m,6) ndarray}  partial of residual w.r.t. each eta
        noise_cov()  -> (m,m) ndarray       observation noise covariance

    The conditioning equation is:
        r(eta) ≈ r0 + sum_k J_k eta_k  ≈  0 + nu,  nu ~ N(0, C_nu)
    """

    @abstractmethod
    def residual(self) -> Array:
        """Residual at zero perturbation. Shape (m,)."""
        ...

    @abstractmethod
    def jacobians(self) -> Dict[str, Array]:
        """
        Jacobians of the residual w.r.t. each state perturbation.

        Returns
        -------
        dict mapping  state_key (str)  ->  J (ndarray, shape (m, 6))
        """
        ...

    @abstractmethod
    def noise_cov(self) -> Array:
        """Observation noise covariance C_nu. Shape (m, m)."""
        ...

    @property
    def dim(self) -> int:
        """Residual dimension m."""
        return self.residual().shape[0]

    @property
    def state_keys(self) -> List[str]:
        """Names of state variables this observation constrains."""
        return list(self.jacobians().keys())


# ---------------------------------------------------------------------------
# LoopObservation
# ---------------------------------------------------------------------------

class LoopObservation(Observation):
    """
    Loop-closure constraint: two paths with the same endpoints should compose
    to the identity transform.

    Residual (6-dimensional):
        r = Log( T_res^{-1}  T_k )  ≈  0 + nu,   nu ~ N(0, C_nu)

    Jacobians are computed via finite differences (same as closed_loop.py).

    Parameters
    ----------
    F_res : (4,4) ndarray
        Nominal transform of the *residual* path.
    F_k : (4,4) ndarray
        Nominal transform of the *alternative* path.
    key_res : str
        State key for the residual-path perturbation eta_res.
    key_k : str
        State key for the alternative-path perturbation eta_k.
    C_nu : (6,6) ndarray, optional
        Observation noise. Default: 1e-9 * I  (hard constraint).
    """

    def __init__(
        self,
        F_res: Array,
        F_k: Array,
        key_res: str,
        key_k: str,
        C_nu: Optional[Array] = None,
    ) -> None:
        self._F_res = np.asarray(F_res, dtype=np.float64)
        self._F_k   = np.asarray(F_k,   dtype=np.float64)
        self._key_res = key_res
        self._key_k   = key_k
        self._C_nu = (
            1e-9 * np.eye(6, dtype=np.float64)
            if C_nu is None
            else np.asarray(C_nu, dtype=np.float64).reshape(6, 6)
        )
        # Precompute linearization once
        self._lin: LoopLinearization = linearize_loop_residual(self._F_res, self._F_k)

    def residual(self) -> Array:
        return self._lin.r0.copy()

    def jacobians(self) -> Dict[str, Array]:
        return {
            self._key_res: self._lin.J_res.copy(),
            self._key_k:   self._lin.J_k.copy(),
        }

    def noise_cov(self) -> Array:
        return self._C_nu.copy()

    def __repr__(self) -> str:
        return f"LoopObservation(key_res={self._key_res!r}, key_k={self._key_k!r})"


# ---------------------------------------------------------------------------
# PointObservation
# ---------------------------------------------------------------------------

class PointObservation(Observation):
    """
    Observed 3D point position in a coordinate frame.

    Measurement model:
        z = p(eta) + nu,   nu ~ N(0, C_nu)
        p(eta) ≈ p_nom + J_eta @ eta       (first-order)

    Residual (3-dimensional):
        r = p_nom - z

    Jacobian (CIS I right-perturbation):
        J_eta = [-R [p_in]x  |  R]         shape (3, 6)

    where R is the rotation of the nominal transform and p_in is the input
    point (before transformation).  When F_nom is not provided, falls back to
    the identity approximation  J_eta = [-[p_nom]x | I_3].

    Parameters
    ----------
    p_nom : (3,) ndarray
        Nominal point position in the query frame.
    J_eta : (3,6) ndarray
        CIS I Jacobian of the point w.r.t. path perturbation.
    key : str
        State key for the path perturbation eta.
    z : (3,) ndarray
        Measured point position in the same frame.
    C_nu : (3,3) ndarray
        Measurement noise covariance.

    Class method
    ------------
    PointObservation.build(p_nom, key, z, C_nu, F_nom=None)
        Constructs the Jacobian automatically using the CIS I right-perturbation
        formula when F_nom is provided.
    """

    def __init__(
        self,
        p_nom: Array,
        J_eta: Array,
        key: str,
        z: Array,
        C_nu: Array,
    ) -> None:
        self._p_nom = np.asarray(p_nom, dtype=np.float64).reshape(3)
        self._J_eta = np.asarray(J_eta, dtype=np.float64).reshape(3, 6)
        self._key   = key
        self._z     = np.asarray(z, dtype=np.float64).reshape(3)
        self._C_nu  = np.asarray(C_nu, dtype=np.float64).reshape(3, 3)

    @classmethod
    def build(
        cls,
        p_nom: Array,
        key: str,
        z: Array,
        C_nu: Array,
        F_nom: Optional[Array] = None,
    ) -> "PointObservation":
        """
        Construct a PointObservation, computing J_eta via CIS I right-perturbation.

        If F_nom is provided:
            p_in  = R^T (p_nom - t)              (recover input point)
            J_eta = [-R [p_in]x | R]             shape (3, 6)

        If F_nom is None (identity approximation):
            J_eta = [-[p_nom]x | I_3]            shape (3, 6)

        Parameters
        ----------
        p_nom : (3,) ndarray
            Nominal point position in the query frame.
        key : str
            State key for the path perturbation.
        z : (3,) ndarray
            Measured position.
        C_nu : (3,3) ndarray
            Measurement noise.
        F_nom : (4,4) ndarray, optional
            Nominal SE(3) transform used to compute the right-convention Jacobian.
        """
        p = np.asarray(p_nom, dtype=np.float64).reshape(3)
        if F_nom is not None:
            F = np.asarray(F_nom, dtype=np.float64).reshape(4, 4)
            R = F[:3, :3]
            t = F[:3, 3]
            p_in = R.T @ (p - t)
            J_eta = np.hstack([-R @ skew(p_in), R])
        else:
            J_eta = np.hstack([-skew(p), np.eye(3)])
        return cls(p_nom=p, J_eta=J_eta, key=key, z=z, C_nu=C_nu)

    def residual(self) -> Array:
        return self._p_nom - self._z

    def jacobians(self) -> Dict[str, Array]:
        return {self._key: self._J_eta.copy()}

    def noise_cov(self) -> Array:
        return self._C_nu.copy()

    def __repr__(self) -> str:
        return f"PointObservation(key={self._key!r}, z={self._z})"


# ---------------------------------------------------------------------------
# DistanceObservation
# ---------------------------------------------------------------------------

class DistanceObservation(Observation):
    """
    Observed scalar distance between two 3D points in a common frame.

    Measurement model:
        z = d(eta_1, eta_2) + nu,   nu ~ N(0, sigma^2)
        d(eta) ≈ d_nom + J_d @ [eta_1; eta_2]

    Residual (1-dimensional):
        r = d_nom - z

    Jacobians via chain rule (unit vector u = (p1 - p2) / d_nom):
        d_d / d_eta_1 =  u^T  J_eta_1     shape (1, 6)
        d_d / d_eta_2 = -u^T  J_eta_2     shape (1, 6)

    Special case: if key_1 == key_2 (both points live in the same path),
    the Jacobians are summed into a single entry.

    Parameters
    ----------
    p1_nom, p2_nom : (3,) ndarray
        Nominal positions of the two points in the common query frame.
    J_eta_1, J_eta_2 : (3,6) ndarray
        CIS I Jacobians for each point.
    key_1, key_2 : str
        State keys for each point's path perturbation.
    z : float
        Measured distance.
    sigma : float
        Standard deviation of the distance measurement noise.

    Class method
    ------------
    DistanceObservation.build(p1_nom, p2_nom, key_1, key_2, z, sigma,
                               F_nom_1=None, F_nom_2=None)
        Constructs both Jacobians using CIS I right-perturbation when
        F_nom_i is provided; falls back to identity approximation otherwise.
    """

    def __init__(
        self,
        p1_nom: Array,
        p2_nom: Array,
        J_eta_1: Array,
        J_eta_2: Array,
        key_1: str,
        key_2: str,
        z: float,
        sigma: float,
    ) -> None:
        self._p1 = np.asarray(p1_nom, dtype=np.float64).reshape(3)
        self._p2 = np.asarray(p2_nom, dtype=np.float64).reshape(3)
        self._J1 = np.asarray(J_eta_1, dtype=np.float64).reshape(3, 6)
        self._J2 = np.asarray(J_eta_2, dtype=np.float64).reshape(3, 6)
        self._key_1 = key_1
        self._key_2 = key_2
        self._z     = float(z)
        self._C_nu  = np.array([[sigma ** 2]], dtype=np.float64)

    @classmethod
    def build(
        cls,
        p1_nom: Array,
        p2_nom: Array,
        key_1: str,
        key_2: str,
        z: float,
        sigma: float,
        F_nom_1: Optional[Array] = None,
        F_nom_2: Optional[Array] = None,
    ) -> "DistanceObservation":
        """
        Construct a DistanceObservation using CIS I right-perturbation Jacobians.

        If F_nom_i is provided:
            p_in_i = R_i^T (p_i - t_i)
            J_eta_i = [-R_i [p_in_i]x | R_i]     shape (3, 6)

        If F_nom_i is None (identity approximation):
            J_eta_i = [-[p_i]x | I_3]              shape (3, 6)
        """
        p1 = np.asarray(p1_nom, dtype=np.float64).reshape(3)
        p2 = np.asarray(p2_nom, dtype=np.float64).reshape(3)

        if F_nom_1 is not None:
            F1 = np.asarray(F_nom_1, dtype=np.float64).reshape(4, 4)
            R1, t1 = F1[:3, :3], F1[:3, 3]
            p1_in = R1.T @ (p1 - t1)
            J1 = np.hstack([-R1 @ skew(p1_in), R1])
        else:
            J1 = np.hstack([-skew(p1), np.eye(3)])

        if F_nom_2 is not None:
            F2 = np.asarray(F_nom_2, dtype=np.float64).reshape(4, 4)
            R2, t2 = F2[:3, :3], F2[:3, 3]
            p2_in = R2.T @ (p2 - t2)
            J2 = np.hstack([-R2 @ skew(p2_in), R2])
        else:
            J2 = np.hstack([-skew(p2), np.eye(3)])

        return cls(
            p1_nom=p1, p2_nom=p2,
            J_eta_1=J1, J_eta_2=J2,
            key_1=key_1, key_2=key_2,
            z=z, sigma=sigma,
        )

    def _unit_and_dist(self) -> Tuple[Array, float]:
        delta = self._p1 - self._p2
        d = float(np.linalg.norm(delta))
        if d < 1e-12:
            unit = np.array([1.0, 0.0, 0.0])
        else:
            unit = delta / d
        return unit, d

    def residual(self) -> Array:
        _, d = self._unit_and_dist()
        return np.array([d - self._z])

    def jacobians(self) -> Dict[str, Array]:
        unit, _ = self._unit_and_dist()
        J1 = (unit @ self._J1).reshape(1, 6)    #  u^T J_eta_1  (1, 6)
        J2 = (-unit @ self._J2).reshape(1, 6)   # -u^T J_eta_2  (1, 6)

        if self._key_1 == self._key_2:
            # Both points depend on the same path perturbation: sum Jacobians.
            return {self._key_1: J1 + J2}
        return {self._key_1: J1, self._key_2: J2}

    def noise_cov(self) -> Array:
        return self._C_nu.copy()

    def __repr__(self) -> str:
        _, d = self._unit_and_dist()
        return (
            f"DistanceObservation(key_1={self._key_1!r}, key_2={self._key_2!r}, "
            f"d_nom={d:.4f}, z={self._z:.4f})"
        )


# ---------------------------------------------------------------------------
# Conditioning result
# ---------------------------------------------------------------------------

@dataclass
class ConditioningResult:
    """
    Result of jointly conditioning on a set of observations.

    Attributes
    ----------
    posteriors : dict  state_key -> (6,6) posterior covariance
        Per-variable posterior covariance extracted from C_full.
    C_full : (6K, 6K) ndarray
        Full joint posterior covariance (state ordered by sorted keys).
    keys : list of str
        Sorted list of state keys (index order for C_full).
    r0 : (M,) ndarray
        Stacked nominal residuals from all observations.
    """
    posteriors: Dict[str, Array]
    C_full: Array
    keys: List[str]
    r0: Array

    def cross_cov(self, key_a: str, key_b: str) -> Array:
        """
        Return the (6,6) cross-covariance between two state variables.

        Parameters
        ----------
        key_a, key_b : str
            State keys (must be in self.keys).
        """
        ia = self.keys.index(key_a)
        ib = self.keys.index(key_b)
        return self.C_full[6 * ia : 6 * ia + 6, 6 * ib : 6 * ib + 6].copy()


# ---------------------------------------------------------------------------
# Joint conditioning
# ---------------------------------------------------------------------------

def condition_on_observations(
    priors: Dict[str, Array],
    observations: List[Observation],
) -> ConditioningResult:
    """
    Jointly condition on a list of observations via the information filter.

    State
    -----
    x = [eta_0; eta_1; ...; eta_{K-1}]   (6K-dimensional, keys sorted)

    Prior
    -----
    C0 = block_diag(C_0, C_1, ..., C_{K-1})

    Each observation i contributes:
        H_i x ≈ 0,   H_i[:,6k:6k+6] = J_{i,k}   for each state key k it touches
        noise: C_nu_i  (m_i × m_i)

    Information update:
        H      = [H_0; H_1; ...; H_{N-1}]          shape (M, 6K)
        C_nu   = block_diag(C_nu_0, ..., C_nu_{N-1}) shape (M, M)
        C_post = (C0^{-1} + H^T C_nu^{-1} H)^{-1}

    Parameters
    ----------
    priors : dict  str -> (6,6) ndarray
        Prior covariance for each named state perturbation.
    observations : list of Observation
        Any mix of LoopObservation, PointObservation, DistanceObservation, etc.

    Returns
    -------
    ConditioningResult
        Posterior covariances per key and full joint posterior matrix.

    Raises
    ------
    KeyError
        If an observation references a state key not present in priors.
    ValueError
        If any prior or noise covariance is not positive definite.
    """
    if not priors:
        raise ValueError("priors must be non-empty.")
    if not observations:
        raise ValueError("observations must be non-empty.")

    # Assign a fixed order to state variables (sorted for reproducibility).
    keys: List[str] = sorted(priors.keys())
    K = len(keys)
    key_to_col: Dict[str, int] = {k: 6 * i for i, k in enumerate(keys)}
    dim = 6 * K

    # --- Build block-diagonal prior C0 ---
    C0 = np.zeros((dim, dim), dtype=np.float64)
    for i, key in enumerate(keys):
        C = np.asarray(priors[key], dtype=np.float64).reshape(6, 6)
        _assert_positive_definite(C, f"prior['{key}']")
        s = 6 * i
        C0[s : s + 6, s : s + 6] = C

    # --- Stack observation Jacobians, residuals, and noise ---
    m_total = sum(obs.dim for obs in observations)
    H          = np.zeros((m_total, dim),       dtype=np.float64)
    C_nu_full  = np.zeros((m_total, m_total),   dtype=np.float64)
    r0_stacked = np.zeros(m_total,              dtype=np.float64)

    row = 0
    for obs in observations:
        m = obs.dim
        # Noise block
        C_nu_full[row : row + m, row : row + m] = obs.noise_cov()
        # Residual
        r0_stacked[row : row + m] = obs.residual()
        # Jacobian blocks (one per state key this observation touches)
        for key, J in obs.jacobians().items():
            if key not in key_to_col:
                raise KeyError(
                    f"Observation {obs!r} references unknown state key '{key}'. "
                    f"Available keys: {keys}"
                )
            col = key_to_col[key]
            H[row : row + m, col : col + J.shape[1]] += np.asarray(J, dtype=np.float64)
        row += m

    # --- Information filter update ---
    _assert_positive_definite(C_nu_full, "stacked C_nu")

    C0_inv   = np.linalg.inv(_sym(C0))
    Cnu_inv  = np.linalg.inv(_sym(C_nu_full))
    I_post   = C0_inv + H.T @ Cnu_inv @ H
    C_post   = _sym(np.linalg.inv(_sym(I_post)))

    # --- Extract per-variable posteriors ---
    posteriors: Dict[str, Array] = {}
    for i, key in enumerate(keys):
        s = 6 * i
        posteriors[key] = _sym(C_post[s : s + 6, s : s + 6])

    return ConditioningResult(
        posteriors=posteriors,
        C_full=C_post,
        keys=keys,
        r0=r0_stacked,
    )
