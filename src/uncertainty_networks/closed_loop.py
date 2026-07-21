# Author: X.M. Christine Zhu
# Date: 02/08/2026

"""
Closed-loop constraint updates for SE(3) uncertainty networks.

This module implements the "closed loop" section of the math note
as an inference layer ON TOP of forward uncertainty propagation.

Core idea
---------
Forward propagation gives a prior on uncertain transforms along a path.
Closed-loop constraints add an "observation equation" that couples two uncertain
transforms, producing a posterior covariance by Gaussian conditioning.

Residual model (SE(3), CIS I right-perturbation)
------------------------------------------------
Let F_res and F_k be uncertain transforms (nominal + perturbation):
    T_res = F_res ∘ Exp(eta_res)
    T_k   = F_k   ∘ Exp(eta_k)

Define the loop residual transform:
    T_loop = T_res^{-1} ∘ T_k

Residual in tangent space:
    r(eta_res, eta_k) = Log( T_loop )

Linearization about eta_res=0, eta_k=0:
    r ≈ r0 + J_res eta_res + J_k eta_k

Closed-loop constraint (typical):
    r ≈ 0   (or r ≈ z) with additive Gaussian noise nu ~ N(0, C_nu).

Then the linear-Gaussian observation is:
    y = H x + b + nu
with
    x = [eta_res; eta_k]
    y = 0   (or z)
    H = [J_res  J_k]
    b = r0  (or r0 - z)

Posterior covariance (information form)
---------------------------------------
Given a prior x ~ N(0, C0), and observation y=0 with noise C_nu:
    C_post = ( C0^{-1} + H^T C_nu^{-1} H )^{-1}

We return blocks:
    cov(eta_res), cov(eta_k), and cross-cov.

Gaussian fusion of multiple estimates
--------------------------------------
Given N independent Gaussian estimates (mu_k, C_k) of the same quantity x:
    C_fused = ( sum_k C_k^{-1} )^{-1}
    mu_fused = C_fused * sum_k C_k^{-1} mu_k

This is the information-form optimal combination.

Multiple simultaneous loop constraints
---------------------------------------
For N loops sharing the same "residual" path but different "alternative" paths,
the joint state is:
    x = [eta_res; eta_k_1; ...; eta_k_N]   (6(1+N) dimensional)

Prior (independent paths):
    C0 = block_diag(C_res, C_k_1, ..., C_k_N)

Each constraint i contributes an observation block:
    H_i x ≈ 0,   H_i = [J_res_i  0 ... J_k_i ... 0]   (6 × 6(1+N))

Stacking all N constraints:
    H x ≈ 0,   H = [H_1; ...; H_N]   (6N × 6(1+N))
    C_nu = block_diag(C_nu_1, ..., C_nu_N)

Then:
    C_post = ( C0^{-1} + H^T C_nu^{-1} H )^{-1}

Notes
-----
- This module uses finite-difference Jacobians for robustness and correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .se3 import exp_se3, log_se3, inv_se3


Array = np.ndarray


@dataclass
class LoopLinearization:
    """Linearization result for a loop residual r = Log(T_res^{-1} T_k)."""
    r0: Array          # (6,)
    J_res: Array       # (6,6) derivative wrt eta_res
    J_k: Array         # (6,6) derivative wrt eta_k


@dataclass
class LoopPosterior:
    """Posterior covariance blocks after conditioning on a loop constraint."""
    C_res: Array       # (6,6)
    C_k: Array         # (6,6)
    C_cross: Array     # (6,6) = cov(eta_res, eta_k)
    C_full: Array      # (12,12)


def _sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def _assert_positive_definite(C: Array, name: str = "C") -> None:
    """
    Raise ValueError if C is not positive definite.

    Uses Cholesky decomposition as the check — this is the standard
    criterion for PD matrices and is needed before any covariance inversion.
    A zero or near-singular covariance will fail here with a clear message
    rather than a cryptic LinAlgError downstream.
    """
    try:
        np.linalg.cholesky(_sym(C))
    except np.linalg.LinAlgError:
        raise ValueError(
            f"'{name}' must be positive definite before closed-loop conditioning. "
            f"Got a singular or indefinite matrix (shape {C.shape}). "
            f"Ensure all covariances are strictly positive definite — "
            f"a zero covariance cannot be inverted."
        )


def loop_residual(F_res: Array, F_k: Array, eta_res: Array, eta_k: Array) -> Array:
    """
    Compute residual r(eta_res, eta_k) = Log( (F_res Exp(eta_res))^{-1} (F_k Exp(eta_k)) ).

    Parameters
    ----------
    F_res, F_k : ndarray, shape (4,4)
        Nominal transforms.
    eta_res, eta_k : ndarray, shape (6,)
        Right perturbations in CIS I ordering [alpha; epsilon].

    Returns
    -------
    r : ndarray, shape (6,)
        Residual tangent vector.
    """
    T_res = F_res @ exp_se3(eta_res)
    T_k = F_k @ exp_se3(eta_k)
    T_loop = inv_se3(T_res) @ T_k
    return log_se3(T_loop)


def linearize_loop_residual(
    F_res: Array,
    F_k: Array,
    eps: float = 1e-8,
) -> LoopLinearization:
    """
    Finite-difference linearization of r(eta_res, eta_k) about (0,0).

    r ≈ r0 + J_res eta_res + J_k eta_k

    Returns
    -------
    LoopLinearization with r0, J_res, J_k.
    """
    eta0 = np.zeros(6, dtype=np.float64)
    r0 = loop_residual(F_res, F_k, eta0, eta0)

    J_res = np.zeros((6, 6), dtype=np.float64)
    J_k = np.zeros((6, 6), dtype=np.float64)

    # FD w.r.t eta_res
    for i in range(6):
        d = np.zeros(6, dtype=np.float64)
        d[i] = eps
        rp = loop_residual(F_res, F_k, d, eta0)
        rm = loop_residual(F_res, F_k, -d, eta0)
        J_res[:, i] = (rp - rm) / (2.0 * eps)

    # FD w.r.t eta_k
    for i in range(6):
        d = np.zeros(6, dtype=np.float64)
        d[i] = eps
        rp = loop_residual(F_res, F_k, eta0, d)
        rm = loop_residual(F_res, F_k, eta0, -d)
        J_k[:, i] = (rp - rm) / (2.0 * eps)

    return LoopLinearization(r0=r0, J_res=J_res, J_k=J_k)


def condition_on_loop(
    C_res: Array,
    C_k: Array,
    lin: LoopLinearization,
    C_nu: Optional[Array] = None,
    z: Optional[Array] = None,
) -> LoopPosterior:
    """
    Condition on a loop constraint / observation.

    Observation:
        r(eta_res, eta_k) ≈ z + nu,   nu ~ N(0, C_nu)

    With linearization:
        r ≈ r0 + J_res eta_res + J_k eta_k

    Put into y = H x + b + nu form with y=z:
        z = H x + r0 + nu
    or equivalently with y=0:
        0 = H x + (r0 - z) + nu

    Covariance update does NOT depend on the offset (r0 - z) if you only
    care about covariance, so we compute:
        C_post = (C0^{-1} + H^T C_nu^{-1} H)^{-1}

    Parameters
    ----------
    C_res, C_k : ndarray, shape (6,6)
        Prior covariances for eta_res and eta_k.
    lin : LoopLinearization
        Linearization (r0, J_res, J_k).
    C_nu : ndarray, shape (6,6), optional
        Residual/observation noise covariance. Default: 1e-9 * I (tight constraint).
    z : ndarray, shape (6,), optional
        Measurement target for residual. (Not needed for covariance-only.)

    Returns
    -------
    LoopPosterior with posterior covariance blocks and full covariance.
    """
    C_res = np.asarray(C_res, dtype=np.float64).reshape(6, 6)
    C_k = np.asarray(C_k, dtype=np.float64).reshape(6, 6)

    if C_nu is None:
        C_nu = 1e-9 * np.eye(6, dtype=np.float64)
    else:
        C_nu = np.asarray(C_nu, dtype=np.float64).reshape(6, 6)

    _assert_positive_definite(C_res, "C_res")
    _assert_positive_definite(C_k, "C_k")
    _assert_positive_definite(C_nu, "C_nu")

    # Prior covariance (independent blocks)
    C0 = np.zeros((12, 12), dtype=np.float64)
    C0[:6, :6] = C_res
    C0[6:, 6:] = C_k

    # Observation Jacobian H = [J_res J_k]
    H = np.zeros((6, 12), dtype=np.float64)
    H[:, :6] = lin.J_res
    H[:, 6:] = lin.J_k

    # Information update
    C0_inv = np.linalg.inv(_sym(C0))
    Cnu_inv = np.linalg.inv(_sym(C_nu))

    I_post = C0_inv + H.T @ Cnu_inv @ H
    C_post = np.linalg.inv(_sym(I_post))
    C_post = _sym(C_post)

    C_res_post = _sym(C_post[:6, :6])
    C_k_post = _sym(C_post[6:, 6:])
    C_cross = C_post[:6, 6:]

    return LoopPosterior(C_res=C_res_post, C_k=C_k_post, C_cross=C_cross, C_full=C_post)


def select_subspace(indices: list[int]) -> Array:
    """
    Build a selection matrix S picking components of a 6-vector.

    Example:
        rotation-only alpha: indices=[0,1,2]
        translation-only epsilon: indices=[3,4,5]
    """
    S = np.zeros((len(indices), 6), dtype=np.float64)
    for r, c in enumerate(indices):
        S[r, c] = 1.0
    return S


def condition_on_loop_subspace(
    C_res: Array,
    C_k: Array,
    lin: LoopLinearization,
    indices: list[int],
    C_nu_sub: Optional[Array] = None,
) -> LoopPosterior:
    """
    Condition on only a subspace of the residual (e.g., alpha-only or epsilon-only).

    We form:
        r_sub = S r  where S selects the desired components.
        H_sub = S [J_res J_k]
        C_nu_sub is noise in the subspace.

    This matches the math note's "similar things involving just alpha or epsilon".
    """
    S = select_subspace(indices)

    if C_nu_sub is None:
        C_nu_sub = 1e-9 * np.eye(S.shape[0], dtype=np.float64)
    else:
        C_nu_sub = np.asarray(C_nu_sub, dtype=np.float64).reshape(S.shape[0], S.shape[0])

    # Build reduced H and perform information update in the same way
    C_res = np.asarray(C_res, dtype=np.float64).reshape(6, 6)
    C_k = np.asarray(C_k, dtype=np.float64).reshape(6, 6)

    _assert_positive_definite(C_res, "C_res")
    _assert_positive_definite(C_k, "C_k")
    _assert_positive_definite(C_nu_sub, "C_nu_sub")

    C0 = np.zeros((12, 12), dtype=np.float64)
    C0[:6, :6] = C_res
    C0[6:, 6:] = C_k

    H_full = np.zeros((6, 12), dtype=np.float64)
    H_full[:, :6] = lin.J_res
    H_full[:, 6:] = lin.J_k

    H = S @ H_full  # (m,12)

    C0_inv = np.linalg.inv(_sym(C0))
    Cnu_inv = np.linalg.inv(_sym(C_nu_sub))

    I_post = C0_inv + H.T @ Cnu_inv @ H
    C_post = np.linalg.inv(_sym(I_post))
    C_post = _sym(C_post)

    return LoopPosterior(
        C_res=_sym(C_post[:6, :6]),
        C_k=_sym(C_post[6:, 6:]),
        C_cross=C_post[:6, 6:],
        C_full=C_post,
    )

def fuse_gaussians(
    means: List[Array],
    covs: List[Array],
) -> Tuple[Array, Array]:
    """
    Fuse N independent Gaussian estimates in information form.

    Given estimates x_k ~ N(mu_k, C_k), the optimal fused estimate is:

        C_fused  = ( sum_k C_k^{-1} )^{-1}
        mu_fused = C_fused * sum_k C_k^{-1} mu_k

    Parameters
    ----------
    means : list of ndarray, each shape (d,)
        Mean vectors of each estimate.
    covs : list of ndarray, each shape (d,d)
        Covariance matrices. All must be positive definite.

    Returns
    -------
    mu_fused : ndarray, shape (d,)
    C_fused  : ndarray, shape (d,d)
    """
    if len(means) != len(covs) or len(means) == 0:
        raise ValueError("means and covs must be non-empty lists of equal length.")

    d = len(means[0])
    I_total = np.zeros((d, d), dtype=np.float64)
    h_total = np.zeros(d, dtype=np.float64)

    for k, (mu, C) in enumerate(zip(means, covs)):
        C = np.asarray(C, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        _assert_positive_definite(C, f"covs[{k}]")
        C_inv = np.linalg.inv(_sym(C))
        I_total += C_inv
        h_total += C_inv @ mu

    C_fused = np.linalg.inv(_sym(I_total))
    C_fused = _sym(C_fused)
    mu_fused = C_fused @ h_total
    return mu_fused, C_fused


def fuse_gaussian_covs(covs: List[Array]) -> Array:
    """
    Fuse N independent Gaussian covariances in information form (means discarded).

        C_fused = ( sum_k C_k^{-1} )^{-1}

    Parameters
    ----------
    covs : list of ndarray, each shape (d,d)
        Covariance matrices. All must be positive definite.

    Returns
    -------
    C_fused : ndarray, shape (d,d)
    """
    if len(covs) == 0:
        raise ValueError("covs must be a non-empty list.")

    d = covs[0].shape[0]
    zeros = [np.zeros(d, dtype=np.float64)] * len(covs)
    _, C_fused = fuse_gaussians(zeros, covs)
    return C_fused


@dataclass
class MultiLoopPosterior:
    """
    Posterior covariance blocks after conditioning on N simultaneous loop constraints.

    State: x = [eta_res; eta_k_1; ...; eta_k_N]  (6(1+N) dimensional)

    Attributes
    ----------
    C_res : (6,6) posterior covariance of eta_res
    C_k_list : list of N (6,6) posterior covariances for eta_k_1 ... eta_k_N
    C_cross_list : list of N (6,6) cross-covariances cov(eta_res, eta_k_i)
    C_full : (6(1+N), 6(1+N)) full posterior covariance
    """
    C_res: Array
    C_k_list: List[Array] = field(default_factory=list)
    C_cross_list: List[Array] = field(default_factory=list)
    C_full: Array = field(default_factory=lambda: np.zeros((0, 0)))


def condition_on_multiple_loops(
    C_res: Array,
    C_k_list: List[Array],
    lin_list: List[LoopLinearization],
    C_nu_list: Optional[List[Array]] = None,
) -> MultiLoopPosterior:
    """
    Condition on N simultaneous loop constraints sharing the same residual path.

    Each constraint i says:
        r_i(eta_res, eta_k_i) ≈ 0 + nu_i,   nu_i ~ N(0, C_nu_i)

    The joint state is:
        x = [eta_res; eta_k_1; ...; eta_k_N]   shape (6(1+N),)

    Prior (independent paths):
        C0 = block_diag(C_res, C_k_1, ..., C_k_N)

    For constraint i, the observation Jacobian row-block is:
        H_i = [J_res_i | 0 ... J_k_i ... 0]   shape (6, 6(1+N))

    All constraints stacked:
        H = [H_1; H_2; ...; H_N]              shape (6N, 6(1+N))
        C_nu = block_diag(C_nu_1,...,C_nu_N)  shape (6N, 6N)

    Information update:
        C_post = ( C0^{-1} + H^T C_nu^{-1} H )^{-1}

    Parameters
    ----------
    C_res : ndarray, shape (6,6)
        Prior covariance for the shared residual path perturbation eta_res.
    C_k_list : list of ndarray, each shape (6,6)
        Prior covariances for each alternative path perturbation eta_k_i.
    lin_list : list of LoopLinearization
        Linearization for each loop (must match length of C_k_list).
    C_nu_list : list of ndarray (6,6), optional
        Observation noise for each constraint. Default: 1e-9 * I for each.

    Returns
    -------
    MultiLoopPosterior
    """
    N = len(C_k_list)
    if N == 0:
        raise ValueError("C_k_list must be non-empty.")
    if len(lin_list) != N:
        raise ValueError("lin_list and C_k_list must have the same length.")

    if C_nu_list is None:
        C_nu_list = [1e-9 * np.eye(6, dtype=np.float64)] * N
    elif len(C_nu_list) != N:
        raise ValueError("C_nu_list must have the same length as C_k_list.")

    C_res = np.asarray(C_res, dtype=np.float64).reshape(6, 6)
    _assert_positive_definite(C_res, "C_res")

    dim = 6 * (1 + N)  # full state dimension

    # Build block-diagonal prior C0
    C0 = np.zeros((dim, dim), dtype=np.float64)
    C0[:6, :6] = C_res
    for i, C_k in enumerate(C_k_list):
        C_k = np.asarray(C_k, dtype=np.float64).reshape(6, 6)
        _assert_positive_definite(C_k, f"C_k_list[{i}]")
        s = 6 + 6 * i
        C0[s:s+6, s:s+6] = C_k

    # Build stacked observation Jacobian H  (6N × dim)
    H = np.zeros((6 * N, dim), dtype=np.float64)
    for i, lin in enumerate(lin_list):
        row = 6 * i
        H[row:row+6, :6] = lin.J_res          # eta_res block
        s = 6 + 6 * i
        H[row:row+6, s:s+6] = lin.J_k         # eta_k_i block

    # Build block-diagonal observation noise C_nu  (6N × 6N)
    C_nu_full = np.zeros((6 * N, 6 * N), dtype=np.float64)
    for i, C_nu in enumerate(C_nu_list):
        C_nu = np.asarray(C_nu, dtype=np.float64).reshape(6, 6)
        _assert_positive_definite(C_nu, f"C_nu_list[{i}]")
        row = 6 * i
        C_nu_full[row:row+6, row:row+6] = C_nu

    # Information update
    C0_inv = np.linalg.inv(_sym(C0))
    Cnu_inv = np.linalg.inv(_sym(C_nu_full))

    I_post = C0_inv + H.T @ Cnu_inv @ H
    C_post = _sym(np.linalg.inv(_sym(I_post)))

    C_res_post = _sym(C_post[:6, :6])
    C_k_post_list = []
    C_cross_list = []
    for i in range(N):
        s = 6 + 6 * i
        C_k_post_list.append(_sym(C_post[s:s+6, s:s+6]))
        C_cross_list.append(C_post[:6, s:s+6])

    return MultiLoopPosterior(
        C_res=C_res_post,
        C_k_list=C_k_post_list,
        C_cross_list=C_cross_list,
        C_full=C_post,
    )


def estimate_residual_covariance(
    F_k_nom,
    C_k,
    C_res_prior,
    mode="se3",
    C_nu=None
):
    """
    Estimate covariance of loop residual perturbation η_res.

    Implements Gaussian conditioning derived in the math note.

    Parameters
    ----------
    F_k_nom : SE3
        Nominal composed transform around the loop.

    C_k : (6x6)
        Covariance of composed transform.

    C_res_prior : (6x6)
        Prior covariance of residual transform.

    mode : {"se3","rot","trans"}
        Whether constraint uses full pose, rotation only, or translation only.

    C_nu : optional measurement noise
    """

    import numpy as np

    if C_nu is None:
        if mode == "se3":
            C_nu = 1e-12*np.eye(6)
        else:
            C_nu = 1e-12*np.eye(3)

    # selection matrix
    if mode == "se3":
        S = np.eye(6)
    elif mode == "rot":
        S = np.zeros((3,6))
        S[:,0:3] = np.eye(3)
    elif mode == "trans":
        S = np.zeros((3,6))
        S[:,3:6] = np.eye(3)

    A = np.eye(6)
    b = np.zeros(6)

    H = np.hstack((-S, S @ A))

    z = -S @ b

    C0 = np.block([
        [C_res_prior, np.zeros((6,6))],
        [np.zeros((6,6)), C_k]
    ])

    mu0 = np.zeros(12)

    _assert_positive_definite(C_res_prior, "C_res_prior")
    _assert_positive_definite(C_k, "C_k")
    _assert_positive_definite(C_nu, "C_nu")

    C0_inv = np.linalg.inv(C0)
    Cnu_inv = np.linalg.inv(C_nu)

    Info = C0_inv + H.T @ Cnu_inv @ H

    C_post = np.linalg.inv(Info)

    mu_post = C_post @ (C0_inv @ mu0 + H.T @ Cnu_inv @ z)

    return mu_post[:6], C_post[:6,:6]
