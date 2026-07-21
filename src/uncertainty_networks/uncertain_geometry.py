# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
This module implements the core uncertain geometric primitives described in:
  docs/math_note.pdf

Scope:
  - Forward uncertainty propagation on SE(3) using first-order approximations
  - CIS I right-multiplicative perturbation convention
  - No estimation / filtering / optimization

Convention (CIS I):
  - Nominal transform: F_nom ∈ SE(3) (4×4 homogeneous matrix)
  - Pose perturbation: eta = [alpha; epsilon] ∈ R^6,  eta ~ N(0, C)
      alpha   ∈ R^3 rotation perturbation
      epsilon ∈ R^3 translation perturbation
  - Right perturbation model:
      T_true = F_nom ∘ Exp(eta)

Core propagation rule (independent edges):
  If F_ab = {F_nom,ab, C_ab} and F_bc = {F_nom,bc, C_bc}, then

      F_nom,ac = F_nom,ab ∘ F_nom,bc
      C_ac ≈ Ad_{F_nom,bc^{-1}} C_ab Ad_{F_nom,bc^{-1}}^T + C_bc

where Ad_T is the SE(3) adjoint under CIS I twist ordering [alpha; epsilon].
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np

from .se3 import adjoint_se3, inv_se3, is_se3, skew
from .nominal_types import vct3, Rot, Frame

Array = np.ndarray


class Convention(str, Enum):
    """Perturbation convention for an uncertain transform's covariance.

    RIGHT (body-frame, CIS I default):
        T_true = F_nom @ Exp(eta)      eta ~ N(0, C)

    LEFT (world-frame):
        T_true = Exp(eta) @ F_nom      eta ~ N(0, C)

    Bridge:
        C_left  = Ad(F_nom)     @ C_right @ Ad(F_nom)^T
        C_right = Ad(F_nom^-1)  @ C_left  @ Ad(F_nom^-1)^T

    Use .as_right_convention() / .as_left_convention() to convert.
    Use .convention to query which one you have.
    """
    RIGHT = "right"
    LEFT  = "left"


@dataclass(frozen=True)
class UncertainTransform:
    r"""
    Uncertain rigid-body transformation in SE(3).

    Representation:
        F = {F_nom, C}

    where:
        - F_nom ∈ SE(3) is the nominal 4×4 homogeneous transform
        - C ∈ R^{6×6} is the covariance of the pose perturbation eta

    Perturbation model (CIS I, right-multiplicative):
        T_true = F_nom ∘ Exp(eta)
        eta = [alpha; epsilon] ~ N(0, C)

    This class supports:
        - composition with first-order uncertainty propagation
        - inversion with first-order covariance mapping
        - point transformation with CIS I Jacobian (implemented later in this file)
    """
    F_nom: Array
    C: Array
    convention: Convention = Convention.RIGHT

    @staticmethod
    def identity(C: Array | None = None) -> "UncertainTransform":
        r"""
        Construct the identity transform with optional covariance.

        Identity:
            F_nom = I_4

        If C is not provided, the covariance defaults to zero.

        Parameters
        ----------
        C : ndarray, optional, shape (6,6)
            Covariance for the identity transform.

        Returns
        -------
        UncertainTransform
            Identity uncertain transform.
        """
        if C is None:
            C = np.zeros((6, 6), dtype=np.float64)
        return UncertainTransform(np.eye(4, dtype=np.float64), C)

    @classmethod
    def from_nominal_frame(cls, frame: Frame, C: np.ndarray | None = None) -> "UncertainTransform":
        r"""
        Construct an UncertainTransform from a nominal Frame object and an optional covariance.

        Parameters
        ----------
        frame : Frame
            The *nominal* rigid-body frame (rotation + translation).
        C : ndarray, optional, shape (6,6)
            Covariance of the pose perturbation. Defaults to zero.

        Returns
        -------
        UncertainTransform
        """
        F = np.eye(4, dtype=np.float64)
        F[:3, :3] = frame.R.matrix
        F[:3, 3] = np.array([frame.p.x, frame.p.y, frame.p.z])
        if C is None:
            C = np.zeros((6, 6), dtype=np.float64)
        return cls(F, C)

    def get_nominal_frame(self) -> Frame:
        r"""
        Extract the nominal transform as a Frame object.

        Returns
        -------
        Frame
            Frame with the same rotation and translation as F_nom.
        """
        return Frame(
            R=Rot(matrix=self.F_nom[:3, :3]),
            p=vct3(float(self.F_nom[0, 3]),
                   float(self.F_nom[1, 3]),
                   float(self.F_nom[2, 3])),
        )

    def inv(self) -> "UncertainTransform":
        r"""
        Invert an uncertain transform (first-order).

        Nominal inverse:
            F_nom^{-1} = inv_se3(F_nom)

        Covariance mapping (RIGHT convention):
            T_true = F_nom @ Exp(eta_R)
            T_true^{-1} = Exp(-Ad_{F_nom} eta_R) @ F_nom^{-1}
                        = F_nom^{-1} @ Exp(-eta_R)  [re-expressed as right perturbation]
            Proof: Exp(-eta_R) = Exp(Ad_{F_nom^{-1}}(-Ad_{F_nom} eta_R))
            =>  C_inv ≈ Ad_{F_nom} C Ad_{F_nom}^T

        Covariance mapping (LEFT convention):
            T_true = Exp(eta_L) @ F_nom
            T_true^{-1} = F_nom^{-1} @ Exp(-eta_L)
                        = Exp(-Ad_{F_nom^{-1}} eta_L) @ F_nom^{-1}
            =>  C_inv ≈ Ad_{F_nom^{-1}} C Ad_{F_nom^{-1}}^T

        Returns
        -------
        UncertainTransform
            Inverse uncertain transform, same convention as self.
        """
        F_inv = inv_se3(self.F_nom)
        if self.convention == Convention.RIGHT:
            Ad = adjoint_se3(self.F_nom)
        else:
            Ad = adjoint_se3(F_inv)
        C_inv = Ad @ self.C @ Ad.T
        return UncertainTransform(F_inv, C_inv, self.convention)

    # ── private composition subroutines ──────────────────────────────────────

    @staticmethod
    def _compose_rr(
        F_ab: Array, C_ab: Array,
        F_bc: Array, C_bc: Array,
    ) -> "tuple[Array, Array, Convention]":
        r"""RIGHT @ RIGHT → RIGHT.

        T_ab = F_ab @ Exp(eta_ab),  T_bc = F_bc @ Exp(eta_bc)
        T_ac = F_ac @ Exp(Ad(F_bc^{-1}) eta_ab + eta_bc)

        C_ac = Ad(F_bc^{-1}) C_ab Ad(F_bc^{-1})^T + C_bc
        """
        F_ac = F_ab @ F_bc
        Ad_inv = adjoint_se3(inv_se3(F_bc))
        C_ac = Ad_inv @ C_ab @ Ad_inv.T + C_bc
        return F_ac, C_ac, Convention.RIGHT

    @staticmethod
    def _compose_ll(
        F_ab: Array, C_ab: Array,
        F_bc: Array, C_bc: Array,
    ) -> "tuple[Array, Array, Convention]":
        r"""LEFT @ LEFT → LEFT.

        T_ab = Exp(eta_ab) @ F_ab,  T_bc = Exp(eta_bc) @ F_bc
        T_ac = Exp(eta_ab + Ad(F_ab) eta_bc) @ F_ac

        C_ac = C_ab + Ad(F_ab) C_bc Ad(F_ab)^T
        """
        F_ac = F_ab @ F_bc
        Ad_ab = adjoint_se3(F_ab)
        C_ac = C_ab + Ad_ab @ C_bc @ Ad_ab.T
        return F_ac, C_ac, Convention.LEFT

    @staticmethod
    def _compose_rl(
        F_ab: Array, C_ab: Array,
        F_bc: Array, C_bc: Array,
    ) -> "tuple[Array, Array, Convention]":
        r"""RIGHT @ LEFT → RIGHT.

        T_ab = F_ab @ Exp(eta_R),  T_bc = Exp(eta_L) @ F_bc
        T_ac ≈ F_ab @ Exp(eta_R + eta_L) @ F_bc
             = F_ac @ Exp(Ad(F_bc^{-1})(eta_R + eta_L))

        C_ac = Ad(F_bc^{-1}) (C_ab + C_bc) Ad(F_bc^{-1})^T
        """
        F_ac = F_ab @ F_bc
        Ad_inv = adjoint_se3(inv_se3(F_bc))
        C_ac = Ad_inv @ (C_ab + C_bc) @ Ad_inv.T
        return F_ac, C_ac, Convention.RIGHT

    @staticmethod
    def _compose_lr(
        F_ab: Array, C_ab: Array,
        F_bc: Array, C_bc: Array,
    ) -> "tuple[Array, Array, Convention]":
        r"""LEFT @ RIGHT → LEFT.

        T_ab = Exp(eta_L) @ F_ab,  T_bc = F_bc @ Exp(eta_R)
        T_ac = Exp(eta_L) @ F_ac @ Exp(eta_R)
             = Exp(eta_L + Ad(F_ac) eta_R) @ F_ac

        C_ac = C_ab + Ad(F_ac) C_bc Ad(F_ac)^T
        """
        F_ac = F_ab @ F_bc
        Ad_ac = adjoint_se3(F_ac)
        C_ac = C_ab + Ad_ac @ C_bc @ Ad_ac.T
        return F_ac, C_ac, Convention.LEFT

    # ── compose dispatcher ────────────────────────────────────────────────────

    _COMPOSE_DISPATCH = None  # populated after class definition

    def compose(self, other: "UncertainTransform") -> "UncertainTransform":
        r"""
        Compose two uncertain transforms with first-order uncertainty propagation.

        Dispatches to one of four subroutines based on the convention flags of
        self and other:

            self \ other | RIGHT           | LEFT
            -------------|----------------|----------------
            RIGHT        | _compose_rr    | _compose_rl
            LEFT         | _compose_lr    | _compose_ll

        The result convention always follows the first (left) operand:
        RR → RIGHT, LL → LEFT, RL → RIGHT, LR → LEFT.

        Parameters
        ----------
        other : UncertainTransform
            The transform to compose on the right.

        Returns
        -------
        UncertainTransform
            Composed uncertain transform.
        """
        fn = UncertainTransform._COMPOSE_DISPATCH[(self.convention, other.convention)]
        F_ac, C_ac, conv = fn(self.F_nom, self.C, other.F_nom, other.C)
        C_ac = 0.5 * (C_ac + C_ac.T)
        return UncertainTransform(F_ac, C_ac, conv)

    def __matmul__(self, other: "UncertainTransform") -> "UncertainTransform":
        r"""
        Operator overload for composition:
            F_ac = F_ab @ F_bc
        """
        return self.compose(other)

    @classmethod
    def from_left_covariance(cls, F_nom: Array, C_left: Array) -> "UncertainTransform":
        r"""
        Construct an UncertainTransform from a covariance given in the left (world-frame) convention.

        Left model:   T_true = Exp(eta_L) * F_nom
        Right model:  T_true = F_nom      * Exp(eta_R)

        Relationship:
            eta_L = Ad_{F_nom} * eta_R
            =>  C_left  = Ad_{F_nom}     * C_right * Ad_{F_nom}^T
            =>  C_right = Ad_{F_nom^{-1}} * C_left  * Ad_{F_nom^{-1}}^T

        Use this when an external source (another library, a sensor driver,
        a paper that uses the world-frame convention) provides a covariance in
        the left convention and you need to feed it into this right-convention framework.

        Parameters
        ----------
        F_nom : array-like, shape (4,4)
            Nominal SE(3) transform.
        C_left : array-like, shape (6,6)
            Covariance expressed in the left (world-frame) convention.

        Returns
        -------
        UncertainTransform
            Same nominal transform, covariance converted to right convention.
        """
        F_nom  = np.asarray(F_nom,  dtype=np.float64)
        C_left = np.asarray(C_left, dtype=np.float64)
        Ad_inv = adjoint_se3(inv_se3(F_nom))
        C_right = Ad_inv @ C_left @ Ad_inv.T
        return cls(F_nom, C_right)

    def as_right_convention(self) -> "UncertainTransform":
        r"""
        Return this transform with covariance in the right (body-frame) convention.

        If already RIGHT: no-op (returns a copy).
        If LEFT: converts C_left → C_right via Ad(F^{-1}).

        Returns
        -------
        UncertainTransform
            convention == Convention.RIGHT
        """
        if self.convention == Convention.RIGHT:
            return UncertainTransform(self.F_nom, self.C.copy(), Convention.RIGHT)
        Ad_inv = adjoint_se3(inv_se3(self.F_nom))
        C_right = Ad_inv @ self.C @ Ad_inv.T
        return UncertainTransform(self.F_nom, 0.5 * (C_right + C_right.T), Convention.RIGHT)

    def as_left_convention(self) -> "UncertainTransform":
        r"""
        Return this transform with covariance in the left (world-frame) convention.

        If already LEFT: no-op (returns a copy).
        If RIGHT: converts C_right → C_left via Ad(F_nom).

        Returns
        -------
        UncertainTransform
            convention == Convention.LEFT
        """
        if self.convention == Convention.LEFT:
            return UncertainTransform(self.F_nom, self.C.copy(), Convention.LEFT)
        Ad = adjoint_se3(self.F_nom)
        C_left = Ad @ self.C @ Ad.T
        return UncertainTransform(self.F_nom, 0.5 * (C_left + C_left.T), Convention.LEFT)

    def apply_to_point(self, p: Array, Cp: Array | None = None) -> tuple[Array, Array]:
        r"""
        Apply this uncertain transform to a 3D point and propagate uncertainty using CIS I Jacobians.

        Nominal point transform:
            p'_nom = R p + t

        CIS I right-perturbation linearization:
            If T_true = F_nom ∘ Exp(eta) with eta = [alpha; epsilon],
            then to first order:

                δp' ≈ -R [p]× alpha + R epsilon

        Therefore, the Jacobians are:
            J_eta = [ -R [p]×   R ]    (shape 3×6)
            J_p   = R              (shape 3×3)

        where p is the **input** point (in the source frame).

        Covariance propagation:
            If point has intrinsic covariance Cp (in the input point's frame),
            then:

                Cp' ≈ J_eta C J_eta^T + R Cp R^T

        If Cp is None, we return the pose-induced covariance only:
                Cp' ≈ J_eta C J_eta^T

        Parameters
        ----------
        p : array-like, shape (3,)
            Input point (3D).
        Cp : ndarray, optional, shape (3,3)
            Intrinsic point covariance.

        Returns
        -------
        p_nom : ndarray, shape (3,)
            Nominal transformed point.
        Cp_out : ndarray, shape (3,3)
            Propagated covariance of transformed point.
        """
        return_point = isinstance(p, vct3)
        if return_point:
            p_arr = np.array([p.x, p.y, p.z], dtype=np.float64)
        else:
            p_arr = np.asarray(p, dtype=np.float64).reshape(3)

        R = self.F_nom[:3, :3]
        t = self.F_nom[:3, 3]

        # Nominal transformation
        p_nom = R @ p_arr + t

        # CIS I Jacobian w.r.t. pose perturbation eta = [alpha; epsilon]
        # Right convention: d p' / d eta = [-R skew(p_in), R]
        J_eta = np.zeros((3, 6), dtype=np.float64)
        J_eta[:, :3] = -R @ skew(p_arr)     # d p' / d alpha (input point)
        J_eta[:, 3:] = R                    # d p' / d epsilon

        Cp_pose = J_eta @ self.C @ J_eta.T

        if Cp is None:
            Cp_out = Cp_pose
        else:
            Cp = np.asarray(Cp, dtype=np.float64).reshape(3, 3)
            Cp_point = R @ Cp @ R.T
            Cp_out = Cp_pose + Cp_point

        # Defensive symmetrization
        Cp_out = 0.5 * (Cp_out + Cp_out.T)

        if return_point:
            return vct3(float(p_nom[0]), float(p_nom[1]), float(p_nom[2])), Cp_out
        return p_nom, Cp_out

    def __post_init__(self) -> None:
        F = np.asarray(self.F_nom, dtype=np.float64)
        C = np.asarray(self.C, dtype=np.float64)

        if F.shape != (4, 4):
            raise ValueError(f"F_nom must be shape (4,4), got {F.shape}")
        if C.shape != (6, 6):
            raise ValueError(f"C must be shape (6,6), got {C.shape}")
        if not is_se3(F):
            raise ValueError("F_nom does not appear to be a valid SE(3) homogeneous transform.")

        # Defensively symmetrize covariance (numerical stability)
        C = 0.5 * (C + C.T)

        object.__setattr__(self, "F_nom", F)
        object.__setattr__(self, "C", C)


# Populate the dispatch table now that all subroutines are defined.
UncertainTransform._COMPOSE_DISPATCH = {
    (Convention.RIGHT, Convention.RIGHT): UncertainTransform._compose_rr,
    (Convention.LEFT,  Convention.LEFT):  UncertainTransform._compose_ll,
    (Convention.RIGHT, Convention.LEFT):  UncertainTransform._compose_rl,
    (Convention.LEFT,  Convention.RIGHT): UncertainTransform._compose_lr,
}
