# Author: X.M. Christine Zhu
# Date: 06/03/2026

"""
uncertain_types.py — User-facing uncertain Cartesian types for CIS I.

Mirrors the nominal types (vct3, Rot, Frame) from spatial_math.py
but carries covariance. All heavy math delegates to UncertainTransform.

API (following Dr. Taylor's CIS I specification):

    uScalar(s, var)         uncertain scalar
    uVector(v, cov)         uncertain vector, any dimension
    uvct3(p, cov)          uncertain 3-D point
    uRot(R, cov)            uncertain rotation
    uFrame(F, cov)          uncertain rigid frame

Supported operations::

    # uScalar
    us3 = us1 + us2           (variances add)
    us3 = us1 - us2           (variances add)
    us3 = us1 * us2           (first-order product)
    us3 = us1 / us2           (first-order quotient)
    us3 = us1 ** n            (first-order power)
    us3 = a  * us1            (a is a Python float)
    us3 = -us1

    # uVector
    uv3 = uv1 + uv2
    uv3 = uv1 - uv2           (covariances add even for subtraction)
    uv3 = a  * uv1            (scalar scaling)
    uv3 = A  @ uv1            (2-D matrix: linear map,  returns uVector)
    us  = w  @ uv1            (1-D vector: dot product, returns uScalar)
    uv3 = uv1 @ A             (uv row @ matrix)
    us  = uv1.dot(w)          (explicit dot product)
    us  = uv1.norm()          (Euclidean norm)

    # uvct3
    up3 = up1 + up2
    up3 = p1  + up2

    # uRot applied to point
    up2 = uR * up1
    up2 = uR * p1

    # uRot composition
    uR3 = uR1 * uR2
    uR3 = R1  * uR2
    uR3 = uR1 * R2

    # uFrame from uncertain parts
    uF = uFrame(uR, up)
    uF = uFrame(R,  up)
    uF = uFrame(uR, p)

    # uFrame composition / point transform
    uF3 = uF1 * uF2
    uF3 = R1  * uF2    (Rot from the left;   also accepts Frame)
    uF3 = uF1 * R2     (Rot from the right;  also accepts Frame)
    up2 = uF  * up1
    up2 = uF  * p1

    # uRot from uncertain angle or axis
    uR = uRot(axis_str, uAngle)     uAngle is a uScalar or uVector of dim 1
    uR = uRot(uAxis,    angle)      uAxis  is a uVector or uvct3
    uR = uRot(uAxis,    uAngle)     both uncertain; uAngle is uScalar or uVector
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _ScipyRot

from .nominal_types import vct3, Rot, Frame
from .uncertain_geometry import UncertainTransform, Convention
from .se3 import skew
from .covariance_types import Covariance


# ── helpers ───────────────────────────────────────────────────────────────────

def _axis_angle_to_rot(axis: np.ndarray, angle: float) -> Rot:
    rotvec = axis * angle
    return Rot(matrix=_ScipyRot.from_rotvec(rotvec).as_matrix())


_AXIS_VECS = {
    'x': np.array([1., 0., 0.]),
    'y': np.array([0., 1., 0.]),
    'z': np.array([0., 0., 1.]),
}


# ── uScalar ───────────────────────────────────────────────────────────────────

class uScalar:
    """
    Uncertain scalar.

        us = uScalar(s)              zero variance
        us = uScalar(s, var)         explicit variance (NOT std dev)
        us = uScalar(other_uscalar)          copy
        us = uScalar(other_uscalar, var)     copy, override variance

    All arithmetic uses first-order Gaussian propagation.  For f(a, b):
        var_f = (df/da)^2 * var_a  +  (df/db)^2 * var_b   (independent)

    Operations::

        us3 = us1 + us2,  us1 + float,  float + us1
        us3 = us1 - us2,  us1 - float,  float - us1
        us3 = us1 * us2,  us1 * float,  float * us1
        us3 = us1 / us2,  us1 / float,  float / us1
        us3 = us1 ** n
        us3 = -us1
    """

    def __init__(self, s, var: float = None):
        if isinstance(s, uScalar):
            self.s   = s.s
            self.var = s.var if var is None else float(var)
            return
        self.s   = float(s)
        self.var = 0.0 if var is None else float(var)

    @property
    def std(self) -> float:
        """Standard deviation (positive square root of variance)."""
        return float(np.sqrt(max(self.var, 0.0)))

    # ── addition ──────────────────────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, uScalar):
            return uScalar(self.s + other.s, self.var + other.var)
        if np.isscalar(other):
            return uScalar(self.s + float(other), self.var)
        return NotImplemented

    def __radd__(self, other):
        if np.isscalar(other):
            return uScalar(float(other) + self.s, self.var)
        return NotImplemented

    # ── subtraction ───────────────────────────────────────────────────────────

    def __sub__(self, other):
        if isinstance(other, uScalar):
            # Var(a - b) = var_a + var_b  for independent a, b
            return uScalar(self.s - other.s, self.var + other.var)
        if np.isscalar(other):
            return uScalar(self.s - float(other), self.var)
        return NotImplemented

    def __rsub__(self, other):
        if np.isscalar(other):
            return uScalar(float(other) - self.s, self.var)
        return NotImplemented

    # ── negation ──────────────────────────────────────────────────────────────

    def __neg__(self):
        return uScalar(-self.s, self.var)

    # ── multiplication ────────────────────────────────────────────────────────

    def __mul__(self, other):
        if isinstance(other, uScalar):
            # d(a*b)/da = b,  d(a*b)/db = a
            return uScalar(
                self.s * other.s,
                other.s ** 2 * self.var + self.s ** 2 * other.var,
            )
        if np.isscalar(other):
            a = float(other)
            return uScalar(self.s * a, a ** 2 * self.var)
        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            a = float(other)
            return uScalar(a * self.s, a ** 2 * self.var)
        return NotImplemented

    # ── division ──────────────────────────────────────────────────────────────

    def __truediv__(self, other):
        if isinstance(other, uScalar):
            # d(a/b)/da = 1/b,  d(a/b)/db = -a/b^2
            s_out   = self.s / other.s
            var_out = (self.var / other.s ** 2
                       + (self.s / other.s ** 2) ** 2 * other.var)
            return uScalar(s_out, var_out)
        if np.isscalar(other):
            a = float(other)
            return uScalar(self.s / a, self.var / a ** 2)
        return NotImplemented

    def __rtruediv__(self, other):
        if np.isscalar(other):
            # f(s) = a/s  →  df/ds = -a/s^2
            a       = float(other)
            s_out   = a / self.s
            var_out = (a / self.s ** 2) ** 2 * self.var
            return uScalar(s_out, var_out)
        return NotImplemented

    # ── power ─────────────────────────────────────────────────────────────────

    def __pow__(self, n):
        # f(s) = s^n  →  df/ds = n * s^(n-1)
        n     = float(n)
        deriv = n * float(self.s ** (n - 1))
        return uScalar(float(self.s ** n), deriv ** 2 * self.var)

    # ── MATLAB-parity named methods (uScalar.m) ─────────────────────────────

    def plus(self, other, C: float = None) -> 'uScalar':
        """
        Same as `+`, but for two independent uScalars accepts an optional
        cross-covariance term C = 2*Cov(a,b), added to the result's variance.
        (ported from uScalar.m's plus(uS,him,C))
        """
        result = self + other
        if C is not None and isinstance(other, uScalar):
            result = uScalar(result.s, result.var + float(C))
        return result

    def minus(self, other, C: float = None) -> 'uScalar':
        """
        Same as `-`, but for two independent uScalars accepts an optional
        cross-covariance term C = 2*Cov(a,b), SUBTRACTED from the result's
        variance (since Var(a-b) = Var(a) + Var(b) - 2*Cov(a,b)).

        NOTE: MATLAB's uScalar.m minus() literally ADDS C here (identical to
        its plus()), which is inconsistent with plus()'s own use of C as a
        2*covariance term -- almost certainly a copy-paste bug from plus().
        This implements the mathematically consistent (subtracting) version.
        """
        result = self - other
        if C is not None and isinstance(other, uScalar):
            result = uScalar(result.s, result.var - float(C))
        return result

    def times(self, other) -> 'uScalar':
        """
        Product variance using the EXACT formula for independent Gaussians
        (not just the first-order approximation used by `*`):

            Var(a*b) = b^2*Var(a) + a^2*Var(b) + Var(a)*Var(b)

        (ported from uScalar.m's times(); `*`/__mul__ keeps the first-order-
        only formula used consistently everywhere else in this library, so
        this exact variant is offered as a separate, explicitly-named method
        rather than changing `*`'s existing, tested behavior.)
        """
        if isinstance(other, uScalar):
            return uScalar(
                self.s * other.s,
                other.s ** 2 * self.var + self.s ** 2 * other.var + self.var * other.var,
            )
        return self * other

    def __repr__(self):
        return f"uScalar(s={self.s:.6g}, std={self.std:.3g})"


# ── uVector ───────────────────────────────────────────────────────────────────

class uVector:
    """
    Uncertain vector of any dimension n.

        uv = uVector(v)               zero covariance
        uv = uVector(v, cov)          explicit n×n covariance
        uv = uVector(other_uvector)          copy
        uv = uVector(other_uvector, cov)     copy, override covariance

    Operations::

        uv3 = uv1 + uv2          (covariances add, independent assumption)
        uv3 = uv1 - uv2          (covariances add even for subtraction)
        uv3 = uv1 + arr          (arr is a constant ndarray, no extra variance)
        uv3 = a  * uv1           (scalar scaling)
        uv3 = A  @ uv1           (2-D matrix: linear map  C_out = A C A^T)
        us  = w  @ uv1           (1-D vector: dot product -> uScalar)
        uv3 = uv1 @ A            (uv row @ matrix)
        us  = uv1.dot(w)         (explicit dot product -> uScalar)
        us  = uv1.norm()         (Euclidean norm -> uScalar)
    """

    # Tell numpy not to handle this type with its own ufuncs/matmul.
    # Without this, `np.array @ uv` and `np.array + uv` are intercepted
    # by numpy before Python can dispatch to __rmatmul__ / __radd__.
    __array_ufunc__ = None

    def __init__(self, v, C=None):
        if isinstance(v, uVector):
            self.v = v.v.copy()
            self.C = v.C.copy() if C is None else Covariance(C)
            return
        self.v = np.asarray(v, dtype=np.float64).ravel()
        n = len(self.v)
        self.C = Covariance(np.zeros((n, n), dtype=np.float64)) if C is None else Covariance(C)

    # ── addition ──────────────────────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, uVector):
            if self.v.shape != other.v.shape:
                raise ValueError(
                    f"uVector shape mismatch in +: {self.v.shape} vs {other.v.shape}"
                )
            return uVector(self.v + other.v, self.C + other.C)
        arr = np.asarray(other, dtype=np.float64).ravel()
        return uVector(self.v + arr, self.C.copy())

    def __radd__(self, other):
        arr = np.asarray(other, dtype=np.float64).ravel()
        return uVector(arr + self.v, self.C.copy())

    # ── subtraction ───────────────────────────────────────────────────────────

    def __sub__(self, other):
        if isinstance(other, uVector):
            if self.v.shape != other.v.shape:
                raise ValueError(
                    f"uVector shape mismatch in -: {self.v.shape} vs {other.v.shape}"
                )
            # Var(a - b) = C_a + C_b for independent vectors
            return uVector(self.v - other.v, self.C + other.C)
        arr = np.asarray(other, dtype=np.float64).ravel()
        return uVector(self.v - arr, self.C.copy())

    def __rsub__(self, other):
        arr = np.asarray(other, dtype=np.float64).ravel()
        return uVector(arr - self.v, self.C.copy())

    # ── negation ──────────────────────────────────────────────────────────────

    def __neg__(self):
        return uVector(-self.v, self.C.copy())

    # ── scalar scaling ────────────────────────────────────────────────────────

    def __mul__(self, other):
        if np.isscalar(other):
            a = float(other)
            return uVector(a * self.v, a ** 2 * self.C)
        return NotImplemented

    def __rmul__(self, other):
        if np.isscalar(other):
            a = float(other)
            return uVector(a * self.v, a ** 2 * self.C)
        return NotImplemented

    # ── linear maps ───────────────────────────────────────────────────────────

    def __rmatmul__(self, A):
        """A @ uv  where A is a constant ndarray.

        A is 2-D (m, n) -> uVector  with C_out = A C A^T
        A is 1-D (n,)   -> uScalar  (dot product)
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim == 1:
            return uScalar(float(A @ self.v), float(A @ self.C @ A))
        if A.ndim == 2:
            return uVector(A @ self.v, A @ self.C @ A.T)
        raise ValueError(
            f"uVector __rmatmul__: expected 1-D or 2-D array, got {A.ndim}-D"
        )

    def __matmul__(self, A):
        """uv @ A  where A is a constant ndarray.

        A is 2-D (n, m) -> uVector  with C_out = A^T C A
        A is 1-D (n,)   -> uScalar  (dot product)
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim == 1:
            return uScalar(float(self.v @ A), float(A @ self.C @ A))
        if A.ndim == 2:
            return uVector(self.v @ A, A.T @ self.C @ A)
        raise ValueError(
            f"uVector __matmul__: expected 1-D or 2-D array, got {A.ndim}-D"
        )

    def dot(self, w: np.ndarray) -> "uScalar":
        """Dot product with a constant vector, returning a uScalar."""
        w = np.asarray(w, dtype=np.float64).ravel()
        return uScalar(float(w @ self.v), float(w @ self.C @ w))

    def norm(self) -> "uScalar":
        """Euclidean norm with first-order variance propagation.

        ||v|| -> uScalar(d, u^T C u)  where u = v / ||v||
        """
        d = float(np.linalg.norm(self.v))
        if d < 1e-12:
            return uScalar(d, float(np.trace(self.C)))
        u = self.v / d
        return uScalar(d, float(u @ self.C @ u))

    def __repr__(self):
        return f"uVector(v={self.v}, C_diag={np.diag(self.C)})"


# ── uvct3 ────────────────────────────────────────────────────────────────────

class uvct3:
    """
    Uncertain 3-D point.

    ::

        up = uvct3(p)                vct3, zero covariance
        up = uvct3(p, cov)           vct3 + 3×3 covariance
        up = uvct3([x,y,z], cov)     array-like also accepted
        up = uvct3(x, y, z)          three separate scalars, zero covariance
        up = uvct3(x, y, z, cov)     three separate scalars + 3×3 covariance
                                     (mirrors vct3(x, y, z)'s own constructor style)
        up = uvct3(other_uvct3)          copy
        up = uvct3(other_uvct3, cov)     copy, override covariance
    """

    def __init__(self, *args, C=None):
        if len(args) == 1:
            p = args[0]
        elif len(args) == 2:
            p, C = args
        elif len(args) == 3:
            p = args
        elif len(args) == 4:
            *p, C = args
        else:
            raise TypeError(
                f"uvct3() takes 1-4 positional arguments (p[, C] or x, y, z[, C]), "
                f"got {len(args)}"
            )

        if isinstance(p, uvct3):
            self._p = p._p
            self.C = p.C.copy() if C is None else Covariance(C)
            return

        if isinstance(p, vct3):
            self._p = p
        else:
            arr = np.asarray(p, dtype=np.float64).ravel()
            self._p = vct3(arr[0], arr[1], arr[2])
        self.C = Covariance(np.zeros((3, 3), dtype=np.float64)) if C is None else Covariance(C)

    # ── properties ──
    @property
    def p(self) -> vct3:
        return self._p

    @property
    def x(self) -> float:
        return self._p.x

    @property
    def y(self) -> float:
        return self._p.y

    @property
    def z(self) -> float:
        return self._p.z

    @classmethod
    def null(cls) -> 'uvct3':
        """The zero point with zero covariance."""
        return cls(vct3(), np.zeros((3, 3), dtype=np.float64))

    # ── addition ──
    def __add__(self, other):
        if isinstance(other, uvct3):
            return uvct3(self._p + other._p, self.C + other.C)
        if isinstance(other, vct3):
            return uvct3(self._p + other, self.C.copy())
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, vct3):
            return uvct3(other + self._p, self.C.copy())
        return NotImplemented

    def plus(self, other, C12: np.ndarray = None) -> 'uvct3':
        """Same as `+`, plus an optional cross-covariance term Cov(self, other)
        (only meaningful when `other` is a uvct3), ADDED to the result's
        covariance -- ported from uvct3.m's plus(v1,v2,C12)."""
        result = self + other
        if C12 is not None and isinstance(other, uvct3):
            result = uvct3(result._p, result.C + np.asarray(C12, dtype=np.float64))
        return result

    # ── subtraction ──
    def __sub__(self, other):
        if isinstance(other, uvct3):
            return uvct3(self._p - other._p, self.C + other.C)
        if isinstance(other, vct3):
            return uvct3(self._p - other, self.C.copy())
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, vct3):
            return uvct3(other - self._p, self.C.copy())
        return NotImplemented

    def minus(self, other, C12: np.ndarray = None) -> 'uvct3':
        """Same as `-`, plus an optional cross-covariance term Cov(self, other)
        (only meaningful when `other` is a uvct3), SUBTRACTED from the result's
        covariance -- ported from uvct3.m's minus(v1,v2,C12)."""
        result = self - other
        if C12 is not None and isinstance(other, uvct3):
            result = uvct3(result._p, result.C - np.asarray(C12, dtype=np.float64))
        return result

    def __neg__(self) -> 'uvct3':
        return uvct3(-self._p, self.C.copy())

    def __abs__(self) -> 'uvct3':
        """Elementwise absolute value; covariance is left unchanged (a sign
        flip alone doesn't change variance under first-order propagation) --
        matches uvct3.m's abs(): `uvct3(abs(v1.nom), v1.C)`."""
        return uvct3(abs(self._p), self.C.copy())

    # ── multiplication ──
    def __mul__(self, other):
        if isinstance(other, uvct3):
            # Dot product of two uncertain vectors -> uScalar.
            # Exact-for-independent-Gaussians variance formula (ported from
            # uvct3.m's mtimes): a'*Cb*a + b'*Ca*b + trace(Ca*Cb), plus an
            # optional cross-covariance correction is NOT supported here since
            # `*` takes no third argument -- see the standalone .dot() method.
            a = self._p.vec.ravel()
            b = other._p.vec.ravel()
            Ca, Cb = self.C, other.C
            s_out = float(a @ b)
            var_out = float(a @ Cb @ a) + float(b @ Ca @ b) + float(np.trace(Ca @ Cb))
            return uScalar(s_out, var_out)

        if isinstance(other, vct3):
            # Dot product with a certain vector -> uScalar (only self
            # contributes variance, via the quadratic form b' C b).
            a = self._p.vec.ravel()
            b = other.vec.ravel()
            s_out = float(a @ b)
            var_out = float(b @ self.C @ b)
            return uScalar(s_out, var_out)

        if isinstance(other, (int, float, np.floating)):
            a = float(other)
            return uvct3(self._p * a, a ** 2 * self.C)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return self.__mul__(other)
        return NotImplemented

    def dot(self, other, C12: np.ndarray = None) -> 'uScalar':
        """Same as `*` for the uvct3/vct3 dot-product cases, plus an optional
        cross-covariance term Cov(self, other) (only meaningful when `other`
        is a uvct3) added to the result's variance."""
        result = self * other
        if C12 is not None and isinstance(other, uvct3):
            a = self._p.vec.ravel()
            b = other._p.vec.ravel()
            C12 = np.asarray(C12, dtype=np.float64)
            extra = 2.0 * float(a @ C12 @ b) + float(np.trace(C12 @ C12))
            result = uScalar(result.s, result.var + extra)
        return result

    def __truediv__(self, scalar) -> 'uvct3':
        if not isinstance(scalar, (int, float, np.floating)):
            return NotImplemented
        a = float(scalar)
        return uvct3(self._p / a, self.C / (a * a))

    def norm(self) -> 'uScalar':
        """
        Small-noise-approximation of the vector's magnitude and its variance.

            d = ||nom||,  var = (nom' C nom) / d^2
        """
        v = self._p.vec.ravel()
        ss = float(v @ v)
        if ss < 1e-24:
            raise ValueError("uvct3.norm(): cannot compute a small-noise-approximation norm at/near zero")
        d = float(np.sqrt(ss))
        var = float(v @ self.C @ v) / ss
        return uScalar(d, var)

    def unit(self) -> 'uvct3':
        """Unit vector via the small-noise approximation (uvct3.m's unit())."""
        s = self.norm()
        return uvct3(self._p / s.s, self.C / (s.s * s.s))

    def cross(self, other, C12: np.ndarray = None) -> 'uvct3':
        """
        Cross product with first-order uncertainty propagation.

        ::

            other a vct3 (certain):  C_out = S @ self.C @ S^T,  S = skew(other)
            other a uvct3:           C_out = S2 C1 S2^T + S1 C2 S1^T
                                      (optionally minus the C12 cross terms, if given)
        """
        if isinstance(other, vct3):
            v = self._p.cross(other)
            S = other.skew()
            C = S @ self.C @ S.T
        elif isinstance(other, uvct3):
            v = self._p.cross(other._p)
            S1 = self._p.skew()
            S2 = other._p.skew()
            C = S2 @ self.C @ S2.T + S1 @ other.C @ S1.T
            if C12 is not None:
                C12 = np.asarray(C12, dtype=np.float64)
                C = C - S2 @ C12 @ S1.T - S1 @ C12.T @ S2.T
        else:
            raise TypeError("uvct3.cross() requires a vct3 or uvct3")
        return uvct3(v, 0.5 * (C + C.T))

    def __repr__(self):
        return (f"uvct3(p=({self.x:.4f}, {self.y:.4f}, {self.z:.4f}), "
                f"C_diag={np.diag(self.C)})")


# ── uRot ──────────────────────────────────────────────────────────────────────

class uRot:
    """
    Uncertain rotation.

    Constructors::

        uR = uRot(R)                        Rot, zero covariance
        uR = uRot(R, Calpha)                Rot + 3×3 rotation covariance
        uR = uRot(matrix, Calpha=None)       3x3 matrix-like, bypassing Rot()
        uR = uRot(axis_str, uAngle)         axis='x'/'y'/'z', uncertain angle
        uR = uRot(uAxis, angle)             uncertain axis (uVector/uvct3), certain angle
        uR = uRot(uAxis, uAngle)            both uncertain
        uR = uRot(other_uRot)                copy (preserves convention)
        uR = uRot(other_uRot, Calpha)         copy, override covariance
    """

    def __init__(self, first, second=None):
        if isinstance(first, uRot):
            # Copy-constructor. Preserve the source's convention explicitly —
            # every other branch below hard-codes RIGHT, which would silently
            # mislabel a LEFT-convention source if reused here.
            self._R = first._R
            self.C = first.C.copy() if second is None else Covariance(second)
            self._convention = first._convention
            return

        if isinstance(first, Rot):
            self._R = first
            C = np.zeros((3, 3), dtype=np.float64) if second is None \
                else np.asarray(second, dtype=np.float64)

        elif isinstance(first, (list, tuple, np.ndarray)):
            # uRot(matrix_like[, Calpha]) — bypasses the Rot() wrapper
            M = np.asarray(first, dtype=np.float64)
            if M.shape != (3, 3):
                raise ValueError(f"uRot: matrix-like argument must be 3x3, got shape {M.shape}")
            self._R = Rot(matrix=M)
            C = np.zeros((3, 3), dtype=np.float64) if second is None \
                else np.asarray(second, dtype=np.float64)

        elif isinstance(first, str):
            # uRot('z', uAngle)  or  uRot('z', angle_float)
            # uAngle may be a uScalar, a 1-D uVector, or a plain float.
            ax = _AXIS_VECS[first.lower()]
            if isinstance(second, uScalar):
                self._R = Rot(axis=first, angle=second.s)
                C = np.outer(ax, ax) * second.var
            elif isinstance(second, uVector):
                angle_nom = float(second.v[0])
                self._R = Rot(axis=first, angle=angle_nom)
                C = np.outer(ax, ax) * float(second.C[0, 0])
            else:
                self._R = Rot(axis=first, angle=float(second))
                C = np.zeros((3, 3), dtype=np.float64)

        elif isinstance(first, (uVector, uvct3)):
            # uRot(uAxis, angle)  or  uRot(uAxis, uAngle)
            if isinstance(first, uvct3):
                ax_nom = np.array([first.x, first.y, first.z])
                C_axis = np.asarray(first.C, dtype=np.float64)
            else:
                ax_nom = first.v[:3]
                C_axis = np.asarray(first.C, dtype=np.float64)[:3, :3]

            norm = np.linalg.norm(ax_nom)
            ax_nom = ax_nom / (norm + 1e-30)

            if isinstance(second, uScalar):
                angle_nom = second.s
                var_angle = second.var
                # Calpha = angle^2 * C_axis + (ax ⊗ ax) * var_angle  (first-order)
                C = angle_nom ** 2 * C_axis + np.outer(ax_nom, ax_nom) * var_angle
            elif isinstance(second, uVector):
                angle_nom = float(second.v[0])
                var_angle = float(second.C[0, 0])
                C = angle_nom ** 2 * C_axis + np.outer(ax_nom, ax_nom) * var_angle
            else:
                angle_nom = float(second)
                C = angle_nom ** 2 * C_axis

            self._R = _axis_angle_to_rot(ax_nom, angle_nom)

        else:
            raise TypeError(
                f"uRot: cannot construct from ({type(first).__name__}, {type(second).__name__}). "
                "Expected (Rot, cov?), (matrix_like, cov?), ('axis', uAngle), "
                "(uAxis, angle_or_uAngle), or (uRot)."
            )
        self.C = Covariance(C)
        self._convention = Convention.RIGHT

    # ── properties ──
    @property
    def convention(self) -> Convention:
        return self._convention

    @property
    def R(self) -> Rot:
        return self._R

    @property
    def matrix(self) -> np.ndarray:
        return self._R.matrix

    # ── multiplication ──
    def __mul__(self, other):
        R_mat = self._R.matrix

        if isinstance(other, uvct3):
            p_arr = np.array([other.x, other.y, other.z])
            p_nom = R_mat @ p_arr
            J_alpha = -R_mat @ skew(p_arr)
            C_out = J_alpha @ self.C @ J_alpha.T + R_mat @ other.C @ R_mat.T
            return uvct3(vct3(*p_nom), 0.5 * (C_out + C_out.T))

        if isinstance(other, vct3):
            p_arr = np.array([other.x, other.y, other.z])
            p_nom = R_mat @ p_arr
            J_alpha = -R_mat @ skew(p_arr)
            C_out = J_alpha @ self.C @ J_alpha.T
            return uvct3(vct3(*p_nom), 0.5 * (C_out + C_out.T))

        if isinstance(other, uRot):
            # Full 4-way RIGHT/LEFT composition, reduced to the already-validated
            # SE(3) machinery in UncertainTransform.compose() by embedding both
            # rotations as zero-translation transforms (verified numerically:
            # the translation block of the composed 6x6 covariance stays exactly
            # zero for all four convention combinations, so this reduction is
            # exact, not an approximation). This also fixes a real gap: the
            # previous implementation always used the RIGHT-RIGHT formula
            # regardless of either operand's actual convention.
            ut1 = self._as_zero_translation_ut()
            ut2 = other._as_zero_translation_ut()
            composed = ut1.compose(ut2)
            R3 = Rot(matrix=composed.F_nom[:3, :3])
            result = uRot(R3, composed.C[:3, :3])
            result._convention = composed.convention
            return result

        if isinstance(other, Rot):
            R2 = other.matrix
            R3 = R_mat @ R2
            if self._convention == Convention.RIGHT:
                # α3 ≈ R2^T α1  (R2 is certain)
                C3 = R2.T @ self.C @ R2
            else:
                # LEFT: R1_true = Exp(a1)@R1_nom, R2 certain => alpha3 = a1 unchanged
                C3 = self.C.copy()
            result = uRot(Rot(matrix=R3), 0.5 * (C3 + C3.T))
            result._convention = self._convention
            return result

        return NotImplemented

    def _as_zero_translation_ut(self) -> UncertainTransform:
        """Embeds this uRot as a zero-translation UncertainTransform, so
        rotation-only composition/inversion can reuse the SE(3) machinery."""
        F = np.eye(4, dtype=np.float64)
        F[:3, :3] = self._R.matrix
        C6 = np.zeros((6, 6), dtype=np.float64)
        C6[:3, :3] = self.C
        return UncertainTransform(F, C6, self._convention)

    def inv(self) -> 'uRot':
        """
        Uncertain-rotation inverse. Flips the perturbation convention
        (RIGHT<->LEFT) and keeps the covariance unchanged, since negating
        the perturbation vector doesn't change its covariance.

        NOTE: MATLAB's uRotMx.Inverse() docstring says "side := 1-side" (this
        same flip), but its actual code (`Ri = RotMx(R.el')`) returns a plain
        RotMx, silently dropping C and side entirely -- broken relative to its
        own docstring. This implements the documented intent instead.
        """
        result = uRot(self._R.inv(), self.C.copy())
        result._convention = Convention.LEFT if self._convention == Convention.RIGHT else Convention.RIGHT
        return result

    def sample(self) -> Rot:
        """
        Draws one random sample Rot from this uncertain rotation's
        distribution: alpha ~ N(0, C); returns nom @ Exp(alpha) [RIGHT] or
        Exp(alpha) @ nom [LEFT] (ported from uRotMx.Sample()).
        """
        from .se3 import exp_so3
        alpha = np.random.multivariate_normal(np.zeros(3), self.C)
        dR = exp_so3(alpha)
        if self._convention == Convention.RIGHT:
            return Rot(matrix=self._R.matrix @ dR)
        return Rot(matrix=dR @ self._R.matrix)

    def __rmul__(self, other):
        if isinstance(other, Rot):
            R3 = other.matrix @ self._R.matrix
            if self._convention == Convention.RIGHT:
                # R1 certain, R2_true = R2_nom @ Exp(a2)  =>  alpha3 = a2 unchanged
                C3 = self.C.copy()
            else:
                # LEFT: R2_true = Exp(a2)@R2_nom => R1@Exp(a2)@R2_nom ≈ Exp(R1@a2)@R3_nom
                R1 = other.matrix
                C3 = R1 @ self.C @ R1.T
            result = uRot(Rot(matrix=R3), 0.5 * (C3 + C3.T))
            result._convention = self._convention
            return result
        return NotImplemented

    # ── convention bridge ──
    @classmethod
    def from_left_covariance(cls, R: Rot, C_left: np.ndarray) -> "uRot":
        """Construct a uRot from a left-convention (world-frame) rotation covariance.

        Left model:  R_true = Exp(alpha_L) * R_nom
        Right model: R_true = R_nom * Exp(alpha_R)

        Relationship (first-order):  alpha_L = R_nom @ alpha_R
            => C_left  = R_nom @ C_right @ R_nom^T
            => C_right = R_nom^T @ C_left @ R_nom
        """
        C_left = np.asarray(C_left, dtype=np.float64)
        R_mat = R.matrix
        C_right = R_mat.T @ C_left @ R_mat
        return cls(R, 0.5 * (C_right + C_right.T))

    def as_right_convention(self) -> "uRot":
        """Return this rotation with covariance in right (body-frame) convention.

        If already RIGHT: no-op.  If LEFT: converts C_left → C_right via R^T.
        """
        if self._convention == Convention.RIGHT:
            result = uRot(self._R, self.C.copy())
        else:
            R_mat = self._R.matrix
            C_right = R_mat.T @ self.C @ R_mat
            result = uRot(self._R, 0.5 * (C_right + C_right.T))
        result._convention = Convention.RIGHT
        return result

    def as_left_convention(self) -> "uRot":
        """Return this rotation with covariance in left (world-frame) convention.

        If already LEFT: no-op.  If RIGHT: converts C_right → C_left via R.
        """
        if self._convention == Convention.LEFT:
            result = uRot(self._R, self.C.copy())
        else:
            R_mat = self._R.matrix
            C_left = R_mat @ self.C @ R_mat.T
            result = uRot(self._R, 0.5 * (C_left + C_left.T))
        result._convention = Convention.LEFT
        return result

    def __repr__(self):
        return f"uRot(C_diag={np.diag(self.C)})"


# ── uFrame ────────────────────────────────────────────────────────────────────

class uFrame:
    """
    Uncertain rigid frame.

    Constructors::

        uF = uFrame(F)              Frame, zero covariance
        uF = uFrame(F, covEta)      Frame + 6×6 pose covariance
        uF = uFrame(matrix, covEta=None)   4x4 matrix-like, bypassing Frame()
        uF = uFrame(uR, up)         uncertain rotation + uncertain point
        uF = uFrame(R,  up)         certain  rotation + uncertain point
        uF = uFrame(uR, p)          uncertain rotation + certain  point
        uF = uFrame(other_uFrame)          copy (preserves convention)
        uF = uFrame(other_uFrame, covEta)   copy, override covariance

    Operations::

        uF3 = uF1 * uF2    frame composition (full uncertainty propagation)
        uF3 = F1  * uF2    certain left frame
        uF3 = uF1 * F2     certain right frame
        up2 = uF  * up1    transform uncertain point
        up2 = uF  * p1     transform certain point
    """

    def __init__(self, first, second=None):
        if isinstance(first, uFrame):
            # Copy-constructor. Preserve the source's convention explicitly —
            # a fresh UncertainTransform(F_nom, C) without a third positional
            # arg silently resets convention to RIGHT, which would corrupt a
            # LEFT-convention source.
            src = first._ut
            C = src.C.copy() if second is None else np.asarray(second, dtype=np.float64)
            self._ut = UncertainTransform(src.F_nom.copy(), C, src.convention)

        elif isinstance(first, UncertainTransform):
            self._ut = first

        elif isinstance(first, (list, tuple, np.ndarray)):
            # uFrame(matrix_4x4[, covEta]) — bypasses the Frame() wrapper
            M = np.asarray(first, dtype=np.float64)
            if M.shape != (4, 4):
                raise ValueError(f"uFrame: matrix-like argument must be 4x4, got shape {M.shape}")
            C = np.zeros((6, 6), dtype=np.float64) if second is None \
                else np.asarray(second, dtype=np.float64)
            self._ut = UncertainTransform(M, C)

        elif isinstance(first, Frame):
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = first.R.matrix
            F_nom[:3, 3] = [first.p.x, first.p.y, first.p.z]
            C = np.zeros((6, 6), dtype=np.float64) if second is None \
                else np.asarray(second, dtype=np.float64)
            self._ut = UncertainTransform(F_nom, C)

        elif isinstance(first, uRot) and isinstance(second, uvct3):
            # uFrame(uR, up)
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = first.R.matrix
            F_nom[:3, 3] = [second.x, second.y, second.z]
            C = np.zeros((6, 6), dtype=np.float64)
            C[:3, :3] = first.C   # rotation covariance  (alpha)
            C[3:, 3:] = second.C  # translation covariance (epsilon)
            self._ut = UncertainTransform(F_nom, C)

        elif isinstance(first, Rot) and isinstance(second, uvct3):
            # uFrame(R, up)
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = first.matrix
            F_nom[:3, 3] = [second.x, second.y, second.z]
            C = np.zeros((6, 6), dtype=np.float64)
            C[3:, 3:] = second.C
            self._ut = UncertainTransform(F_nom, C)

        elif isinstance(first, uRot) and isinstance(second, vct3):
            # uFrame(uR, p)
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = first.R.matrix
            F_nom[:3, 3] = [second.x, second.y, second.z]
            C = np.zeros((6, 6), dtype=np.float64)
            C[:3, :3] = first.C
            self._ut = UncertainTransform(F_nom, C)

        else:
            raise TypeError(
                f"uFrame: unsupported arguments ({type(first).__name__}, {type(second).__name__}). "
                "Expected (Frame, cov?), (matrix_like, cov?), (uRot, uvct3), (Rot, uvct3), "
                "(uRot, vct3), or (uFrame)."
            )

    # ── properties ──
    @property
    def convention(self) -> Convention:
        return self._ut.convention

    @property
    def F(self) -> Frame:
        return self._ut.get_nominal_frame()

    @property
    def F_nom(self) -> np.ndarray:
        return self._ut.F_nom

    @property
    def C(self) -> np.ndarray:
        return self._ut.C

    def inv(self) -> "uFrame":
        return uFrame(self._ut.inv())

    def to_uncertain_transform(self) -> UncertainTransform:
        return self._ut

    # ── convention bridge ──
    @classmethod
    def from_left_covariance(cls, frame_or_F_nom, C_left: np.ndarray) -> "uFrame":
        """Construct a uFrame from a left-convention (world-frame) pose covariance.

        Left model:  T_true = Exp(eta_L) * F_nom
        Right model: T_true = F_nom * Exp(eta_R)

        Relationship (first-order):  eta_L = Ad_{F_nom} @ eta_R
            => C_right = Ad_{F_nom^{-1}} @ C_left @ Ad_{F_nom^{-1}}^T

        Accepts either a Frame object or a raw 4×4 numpy array as the nominal transform.
        """
        if isinstance(frame_or_F_nom, Frame):
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = frame_or_F_nom.R.matrix
            F_nom[:3, 3] = [frame_or_F_nom.p.x, frame_or_F_nom.p.y, frame_or_F_nom.p.z]
        else:
            F_nom = np.asarray(frame_or_F_nom, dtype=np.float64)
        return cls(UncertainTransform.from_left_covariance(F_nom, C_left))

    def as_right_convention(self) -> "uFrame":
        """No-op — covariance is already in right (body-frame) convention."""
        return uFrame(self._ut.as_right_convention())

    def as_left_convention(self) -> "uFrame":
        """Convert right-convention (body-frame) covariance to left-convention (world-frame).

        C_left = Ad_{F_nom} @ C_right @ Ad_{F_nom}^T
        """
        return uFrame(self._ut.as_left_convention())

    # ── multiplication ──
    def __mul__(self, other):
        if isinstance(other, uFrame):
            return uFrame(self._ut.compose(other._ut))

        if isinstance(other, Frame):
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = other.R.matrix
            F_nom[:3, 3] = [other.p.x, other.p.y, other.p.z]
            return uFrame(self._ut.compose(UncertainTransform(F_nom, np.zeros((6, 6)))))

        if isinstance(other, Rot):
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = other.matrix
            return uFrame(self._ut.compose(UncertainTransform(F_nom, np.zeros((6, 6)))))

        if isinstance(other, uvct3):
            p_out, C_out = self._ut.apply_to_point(other.p, Cp=other.C)
            return uvct3(p_out, C_out)

        if isinstance(other, vct3):
            p_out, C_out = self._ut.apply_to_point(other)
            return uvct3(p_out, C_out)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Frame):
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = other.R.matrix
            F_nom[:3, 3] = [other.p.x, other.p.y, other.p.z]
            return uFrame(UncertainTransform(F_nom, np.zeros((6, 6))).compose(self._ut))
        if isinstance(other, Rot):
            F_nom = np.eye(4, dtype=np.float64)
            F_nom[:3, :3] = other.matrix
            return uFrame(UncertainTransform(F_nom, np.zeros((6, 6))).compose(self._ut))
        return NotImplemented

    def __matmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        p = self._ut.F_nom[:3, 3]
        return (f"uFrame(p=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}), "
                f"C_diag={np.diag(self._ut.C)})")


# ── uPlane ────────────────────────────────────────────────────────────────────

class uPlane:
    """
    An uncertain infinite plane: an uncertain unit normal nV plus an
    uncertain point pV on the plane. Ported from uPlane.m, fixing two bugs
    found there: (1) the isa(nV,"uvct3") branch references an undefined
    lowercase `nv` instead of `nV`, meaning it errors whenever you pass an
    already-uvct3 normal -- the very case the property type implies is
    normal; (2) the "else" branch of pV coerces to a certain vct3 instead of
    uvct3, so P.pV's type silently depended on what was passed in.

    ::

        uPlane(nV, pV)   normal (uvct3/vct3/array-like) + point (same forms)
    """

    def __init__(self, nV, pV):
        self.nV = (nV if isinstance(nV, uvct3) else uvct3(nV)).unit()
        self.pV = pV if isinstance(pV, uvct3) else uvct3(pV)

    def distance(self, xV) -> uScalar:
        """Signed distance from `xV` (vct3/uvct3/array-like) to this plane."""
        dV = uvct3(xV) - self.pV
        return self.nV * dV

    def __repr__(self):
        return f"uPlane(nV={self.nV!r}, pV={self.pV!r})"


# ── uSphere ───────────────────────────────────────────────────────────────────

class uSphere:
    """
    An uncertain sphere: an uncertain center cV and an uncertain radius R
    (default uScalar(1, 0)). Ported from uSphere.m, fixing a bug there:
    the two-argument constructor calls `uscalar(R)` (lowercase) where the
    actual class is `uScalar` -- MATLAB is case-sensitive for class names,
    so passing an explicit radius would error; only the default radius
    ever actually worked in the original source.

    ::

        uSphere(cV)       center (uvct3/vct3/array-like), default radius
        uSphere(cV, R)    center + radius (uScalar/float)
    """

    def __init__(self, cV, R: uScalar = None):
        self.cV = cV if isinstance(cV, uvct3) else uvct3(cV)
        if R is None:
            self.R = uScalar(1.0, 0.0)
        else:
            self.R = R if isinstance(R, uScalar) else uScalar(R)

    def distance(self, xV) -> uScalar:
        """Distance from `xV` to the sphere's CENTER (matches the MATLAB
        source; ignores the stored radius, same as the certain Sphere)."""
        return (uvct3(xV) - self.cV).norm()

    def __repr__(self):
        return f"uSphere(cV={self.cV!r}, R={self.R!r})"


# ── uLine ─────────────────────────────────────────────────────────────────────

class uLine:
    """
    A line defined by a unit direction nV and a point pV. Ported from
    uLine.m, which your mentor explicitly marked "Still need to work on
    this" in its source comments -- its properties are plain vct3 (not
    uvct3), so no uncertainty is actually modeled here yet; this mirrors
    that same (WIP) structure rather than inventing an uncertainty model
    your mentor hasn't specified. One bug is fixed: the source's pV
    branch does `P.pV=pV` in BOTH the isa(pV,'vct3') and else cases,
    meaning a non-vct3 pV is never actually coerced -- this coerces it
    properly in both branches, matching the obviously-intended behavior
    (the same fix already applied to the certain Line class in §18).

    ::

        uLine(nV, pV)   direction (vct3/array-like) + point (vct3/array-like)
    """

    def __init__(self, nV, pV):
        self.nV = (nV if isinstance(nV, vct3) else vct3(nV)).unit()
        self.pV = pV if isinstance(pV, vct3) else vct3(pV)

    def distance(self, xV) -> float:
        """Distance from `xV` to this line."""
        xxv = xV if isinstance(xV, vct3) else vct3(xV)
        pd = self.nV.dot(xxv - self.pV)
        cp = self.nV * pd + self.pV
        return (xxv - cp).norm()

    def __repr__(self):
        return f"uLine(nV={self.nV!r}, pV={self.pV!r})"
