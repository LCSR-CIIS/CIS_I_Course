# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
 This module implements minimal SE(3) utilities required by the mathematical
 framework defined in:

   "Mathematical Framework for Uncertainty Propagation in Geometric Networks"

 The implementation follows the CIS I convention used throughout the project:
   - Homogeneous transforms T ∈ SE(3) are represented as 4×4 matrices
   - Perturbations (twists) are ordered as:

         xi = [ alpha ; epsilon ] ∈ R^6

     where:
       alpha   ∈ R^3  is the rotational perturbation
       epsilon ∈ R^3  is the translational perturbation
   - Right-multiplicative perturbation model:

         T_true = T_nom ∘ Exp(xi)

 Consistency with this convention is critical for uncertainty propagation.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def skew(w: Array) -> Array:
    r"""
    Construct the skew-symmetric matrix [w]× for a vector w ∈ R^3.

    Mathematical definition:
        For w = [w_x, w_y, w_z]^T,

            [w]× = [[  0,  -w_z,  w_y],
                    [ w_z,   0,  -w_x],
                    [-w_y,  w_x,   0 ]]

    Property:
        For any vector a ∈ R^3,
            [w]× a = w × a

    This operator appears throughout SE(3) linearizations, Jacobians,
    and adjoint mappings.

    Parameters
    ----------
    w : array-like, shape (3,)
        3D vector.

    Returns
    -------
    ndarray, shape (3,3)
        Skew-symmetric matrix.
    """
    w = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ],
        dtype=np.float64,
    )


def is_se3(T: Array, atol: float = 1e-8) -> bool:
    r"""
    Lightweight structural check for a homogeneous transform T ∈ SE(3).

    This function verifies:
      - T is 4×4
      - the last row is [0, 0, 0, 1]
      - the rotation block is approximately orthonormal: RᵀR ≈ I

    Note:
        This is intentionally a *lightweight* check. We do not enforce
        det(R)=+1 strictly, since small numerical errors are expected in
        simulation and Monte Carlo sampling.

    Parameters
    ----------
    T : array-like, shape (4,4)
        Homogeneous transform.
    atol : float
        Absolute tolerance for numerical checks.

    Returns
    -------
    bool
        True if T has valid SE(3) structure.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        return False
    R = T[:3, :3]
    return np.allclose(R.T @ R, np.eye(3), atol=1e-6)


def inv_se3(T: Array) -> Array:
    r"""
    Compute the inverse of a homogeneous transform T ∈ SE(3).

    If:
        T = [[R, p],
             [0, 1]]

    then:
        T^{-1} = [[Rᵀ, -Rᵀ p],
                  [ 0,     1 ]]

    This formula follows directly from rigid-body kinematics.

    Parameters
    ----------
    T : array-like, shape (4,4)
        Homogeneous transform.

    Returns
    -------
    ndarray, shape (4,4)
        Inverse transform.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"inv_se3 expects (4,4), got {T.shape}")

    R = T[:3, :3]
    p = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def adjoint_se3(T: Array) -> Array:
    r"""
    Compute the adjoint matrix Ad_T ∈ R^{6×6} for a transform T ∈ SE(3).

    Convention (CIS I):
        Twist / perturbation vector is ordered as:
            xi = [ alpha ; epsilon ]
        where alpha is rotational and epsilon is translational.

    For:
        T = [[R, p],
             [0, 1]]

    the adjoint is defined as:

        Ad_T = [[ R,          0 ],
                [[p]× R,      R ]]

    Role in uncertainty propagation:
        Under right-multiplicative perturbations,

            T_true = T_nom ∘ Exp(xi)

        when composing two uncertain transforms,
        the perturbation of the first transform must be mapped by
        the adjoint of the inverse of the second nominal transform:

            xi_ac ≈ Ad_{T_bc^{-1}} xi_ab + xi_bc

        and covariance propagates as:

            C_ac ≈ Ad_{T_bc^{-1}} C_ab Ad_{T_bc^{-1}}ᵀ + C_bc

    Parameters
    ----------
    T : array-like, shape (4,4)
        Homogeneous transform.

    Returns
    -------
    ndarray, shape (6,6)
        Adjoint matrix.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"adjoint_se3 expects (4,4), got {T.shape}")

    R = T[:3, :3]
    p = T[:3, 3]

    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew(p) @ R
    return Ad


def make_se3(R: Array, p: Array) -> Array:
    r"""
    Construct a homogeneous transform from rotation and translation.

    Parameters
    ----------
    R : array-like, shape (3,3)
        Rotation matrix.
    p : array-like, shape (3,)
        Translation vector.

    Returns
    -------
    ndarray, shape (4,4)
        Homogeneous transform [[R, p], [0, 1]].
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    p = np.asarray(p, dtype=np.float64).reshape(3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rotz(theta_rad: float) -> Array:
    r"""
    Rotation matrix for a rotation about the z-axis.

    Parameters
    ----------
    theta_rad : float
        Rotation angle in radians.

    Returns
    -------
    ndarray, shape (3,3)
        Rotation matrix.
    """
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

def exp_so3(phi: Array) -> Array:
    r"""
    Exponential map Exp: so(3) -> SO(3) using Rodrigues' formula.

    Input:
        phi ∈ R^3 (rotation vector), with magnitude θ = ||phi||.

    Hat operator:
        [phi]× = skew(phi)

    Rodrigues:
        Exp([phi]×) = I + (sinθ/θ)[phi]× + ((1-cosθ)/θ^2)[phi]×^2

    For small θ, series expansions are used for numerical stability.

    Parameters
    ----------
    phi : array-like, shape (3,)
        Rotation vector.

    Returns
    -------
    R : ndarray, shape (3,3)
        Rotation matrix.
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-12:
        # First-order approximation: R ≈ I + [phi]×
        return I + skew(phi)

    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta * theta)
    K = skew(phi)
    return I + A * K + B * (K @ K)


def log_so3(R: Array) -> Array:
    r"""
    Logarithm map Log: SO(3) -> so(3) returning rotation vector phi ∈ R^3.

    For a rotation matrix R, define:
        cosθ = (tr(R) - 1)/2

    Then:
        phi = (θ / (2 sinθ)) * vee(R - Rᵀ)

    For small θ, use a stable approximation.

    Parameters
    ----------
    R : array-like, shape (3,3)
        Rotation matrix.

    Returns
    -------
    phi : ndarray, shape (3,)
        Rotation vector.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    # Clamp for numerical stability
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-12:
        # For very small angles, use first-order approximation:
        # R ≈ I + [phi]×  =>  [phi]× ≈ (R - Rᵀ)/2
        K = 0.5 * (R - R.T)
        return np.array([K[2, 1], K[0, 2], K[1, 0]], dtype=np.float64)

    # vee(R - Rᵀ)
    K = (R - R.T)
    vee = np.array([K[2, 1], K[0, 2], K[1, 0]], dtype=np.float64)
    return (theta / (2.0 * np.sin(theta))) * vee


def _left_jacobian_so3(phi: Array) -> Array:
    r"""
    Left Jacobian J(φ) for SO(3), used in SE(3) exponential/logarithm.

    J(φ) maps the translational part in the SE(3) exponential:
        Exp_SE3([φ; ρ]) = (Exp_SO3(φ),  J(φ) ρ)

    Closed form:
        J(φ) = I + (1-cosθ)/θ^2 [φ]× + (θ - sinθ)/θ^3 [φ]×^2

    For small θ, uses series expansions.

    Parameters
    ----------
    phi : array-like, shape (3,)

    Returns
    -------
    J : ndarray, shape (3,3)
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-12:
        # Series: J ≈ I + 1/2 [φ]×
        return I + 0.5 * skew(phi)

    K = skew(phi)
    theta2 = theta * theta
    A = (1.0 - np.cos(theta)) / theta2
    B = (theta - np.sin(theta)) / (theta2 * theta)
    return I + A * K + B * (K @ K)


def _left_jacobian_inv_so3(phi: Array) -> Array:
    r"""
    Inverse of the SO(3) left Jacobian J(φ)^{-1}.

    Used in SE(3) logarithm:
        Log_SE3(R, p) returns [φ; ρ] with ρ = J(φ)^{-1} p.

    Closed form:
        J^{-1}(φ) = I - 1/2 [φ]× + a [φ]×^2
    where:
        a = (1/θ^2) * (1 - (θ sinθ)/(2(1-cosθ)))

    For small θ, uses series expansions.

    Parameters
    ----------
    phi : array-like, shape (3,)

    Returns
    -------
    Jinv : ndarray, shape (3,3)
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    I = np.eye(3, dtype=np.float64)

    if theta < 1e-7:
        # Use series expansion to avoid cancellation in 1 - cos(theta).
        # For float64, 1 - cos(theta) underflows to 0 when theta < ~2e-8,
        # so the closed form blows up. Series: J^{-1} ≈ I - 1/2 [φ]× + 1/12 [φ]×^2
        K = skew(phi)
        return I - 0.5 * K + (1.0 / 12.0) * (K @ K)

    K = skew(phi)
    theta2 = theta * theta
    half = 0.5

    # a = 1/θ^2 * (1 - θ sinθ / (2(1-cosθ)))
    denom = 2.0 * (1.0 - np.cos(theta))
    a = (1.0 / theta2) * (1.0 - (theta * np.sin(theta)) / denom)

    return I - half * K + a * (K @ K)


def exp_se3(xi: Array) -> Array:
    r"""
    Exponential map Exp: se(3) -> SE(3) under CIS I ordering.

    Input:
        xi = [alpha; epsilon] ∈ R^6
        alpha   ∈ R^3 : rotation vector
        epsilon ∈ R^3 : translation-like component (in tangent space)

    Output:
        T = Exp_SE3(xi) ∈ SE(3)

    Under the standard matrix Lie group exponential:
        Exp_SE3([alpha; epsilon]) = (R, p)
    where:
        R = Exp_SO3(alpha)
        p = J(alpha) * epsilon

    with J(alpha) the SO(3) left Jacobian:
        J(α) = I + (1-cosθ)/θ^2 [α]× + (θ - sinθ)/θ^3 [α]×^2,
        θ = ||α||.

    This is the correct mapping for Monte Carlo simulation of right perturbations:
        T_true = T_nom ∘ Exp(xi)

    Parameters
    ----------
    xi : array-like, shape (6,)
        Twist vector [alpha; epsilon].

    Returns
    -------
    T : ndarray, shape (4,4)
        Homogeneous transform in SE(3).
    """
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    alpha = xi[:3]
    eps = xi[3:]

    R = exp_so3(alpha)
    J = _left_jacobian_so3(alpha)
    p = J @ eps

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def log_se3(T: Array) -> Array:
    r"""
    Logarithm map Log: SE(3) -> se(3) under CIS I ordering.

    Input:
        T = [[R, p],
             [0, 1]] ∈ SE(3)

    Output:
        xi = [alpha; epsilon] ∈ R^6

    Steps:
        alpha = Log_SO3(R)
        epsilon = J(alpha)^{-1} p

    where J(alpha) is the SO(3) left Jacobian.

    This is used in Monte Carlo validation to express the residual transform
    in the tangent space:
        xi_res = Log( T_nom^{-1} ∘ T_sample )

    Parameters
    ----------
    T : array-like, shape (4,4)
        Homogeneous transform.

    Returns
    -------
    xi : ndarray, shape (6,)
        Twist vector [alpha; epsilon].
    """
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]

    alpha = log_so3(R)
    Jinv = _left_jacobian_inv_so3(alpha)
    eps = Jinv @ p

    xi = np.zeros(6, dtype=np.float64)
    xi[:3] = alpha
    xi[3:] = eps
    return xi
