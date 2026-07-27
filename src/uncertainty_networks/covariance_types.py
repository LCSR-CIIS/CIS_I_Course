"""
covariance_types.py — A validated, importable covariance-matrix type.

Every uncertain type in this library (uVector.C, uvct3.C, uRot.C,
UncertainTransform.C, and uScalar.var for the 1-D case) stores its
covariance as a bare NumPy array, with validation applied inconsistently:
UncertainTransform defensively symmetrizes in __post_init__, but
uVector/uvct3/uRot/uPlane/uSphere accept whatever array they're given.

`Covariance` gives covariance matrices a first-class, importable type:
shape is validated and symmetry is defensively enforced at construction
time (same convention UncertainTransform already uses), with `.is_psd()`
available for an explicit, opt-in positive-semidefinite check.

Because `Covariance` implements `__array__`, it is a drop-in replacement
for a bare ndarray everywhere this library currently expects one -- pass
a `Covariance` instance anywhere `C=`/`cov=`/`Calpha=` is accepted
(uVector, uvct3, uRot, uFrame/UncertainTransform, uPlane, uSphere) and it
converts via `np.asarray(...)` exactly like a plain array does today, with
no changes required to any of those constructors.
"""

from __future__ import annotations

import numpy as np


class Covariance:
    """
    An n x n symmetric covariance matrix.

    ::

        Covariance(matrix)                validate + wrap an n x n array-like
        Covariance(other_covariance)      copy
        Covariance(var_x, var_y, var_z)   two-or-more separate variances -> diagonal
                                           (mirrors vct3(x, y, z)'s own constructor style)
        Covariance(std=[sx, sy, sz])      diagonal covariance from per-axis standard deviations
        Covariance.zeros(n)               the n x n zero matrix (a certain/deterministic quantity)
        Covariance.eye(n, scale=1.0)      scale * identity (isotropic covariance)
        Covariance.from_std(sigmas)       same as Covariance(std=sigmas), as a classmethod

    Validation, on every construction:
        - must be square (n, n), else raises ValueError
        - defensively symmetrized (C := 0.5*(C + C.T)) -- the same convention
          already used by UncertainTransform elsewhere in this library, since
          floating-point propagation routinely introduces tiny (~1e-16)
          asymmetries that aren't a real correctness issue
        - NOT strictly checked for positive-semidefiniteness by default,
          since many first-order-propagated covariances can have tiny
          negative eigenvalues from numerical error near zero -- use
          .is_psd() to check explicitly, or pass check_psd=True to raise
          if the matrix isn't (within tolerance)
    """

    # Tell numpy not to intercept `ndarray + Covariance` / `ndarray @ Covariance`
    # with its own ufunc machinery before Python can dispatch to our
    # __radd__/__rmatmul__ (same convention already used by uVector).
    __array_ufunc__ = None

    def __init__(self, *args, std=None, check_psd: bool = False):
        if std is not None:
            if args:
                raise TypeError("Covariance: cannot combine positional arguments with std=")
            sigmas = np.asarray(std, dtype=np.float64).ravel()
            self._mat = np.diag(sigmas ** 2)

        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, Covariance):
                self._mat = arg._mat.copy()
            else:
                M = np.asarray(arg, dtype=np.float64)
                if M.ndim != 2 or M.shape[0] != M.shape[1]:
                    raise ValueError(
                        f"Covariance: matrix must be square (n,n), got shape {M.shape}"
                    )
                self._mat = 0.5 * (M + M.T)

        elif len(args) >= 2:
            # Two-or-more separate variances -> an n x n diagonal matrix.
            variances = np.asarray(args, dtype=np.float64)
            self._mat = np.diag(variances)

        else:
            raise TypeError(
                "Covariance() requires a matrix-like/Covariance argument, "
                "two-or-more separate variances, or std=[...]"
            )

        if check_psd and not self.is_psd():
            raise ValueError("Covariance: matrix is not positive semi-definite")

    # ── alternate constructors ──────────────────────────────────────────────

    @classmethod
    def zeros(cls, n: int) -> 'Covariance':
        """The n x n zero covariance matrix (a certain/deterministic quantity)."""
        return cls(np.zeros((n, n), dtype=np.float64))

    @classmethod
    def eye(cls, n: int, scale: float = 1.0) -> 'Covariance':
        """scale * I_n -- an isotropic covariance matrix."""
        return cls(scale * np.eye(n, dtype=np.float64))

    @classmethod
    def from_std(cls, sigmas) -> 'Covariance':
        """A diagonal covariance matrix built from per-axis standard deviations."""
        return cls(std=sigmas)

    # ── access ───────────────────────────────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        """Returns the raw (n, n) NumPy array."""
        return self._mat

    def copy(self) -> 'Covariance':
        """A deep copy (mirrors ndarray.copy(), since Covariance is used as a
        drop-in replacement for a bare covariance array throughout this library)."""
        return Covariance(self)

    @property
    def n(self) -> int:
        return self._mat.shape[0]

    @property
    def shape(self) -> tuple:
        return self._mat.shape

    def __len__(self) -> int:
        return self.n

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Lets a Covariance be passed anywhere a bare ndarray is expected --
        np.asarray(cov)/np.array(cov) (and anything that calls one of those
        internally, including every uVector/uvct3/uRot/UncertainTransform
        constructor in this library) returns the underlying matrix directly.

        Accepts the `copy` keyword per the NumPy 2.x __array__ protocol
        (numpy.org/devdocs/numpy_2_0_migration_guide.html) -- without it,
        np.array(cov, ...) raises a DeprecationWarning under NumPy >= 2.0.
        `copy=False` is honored on a best-effort basis: a cast to a different
        dtype always requires a copy regardless of what was requested.
        """
        if dtype is not None and np.dtype(dtype) != self._mat.dtype:
            if copy is False:
                raise ValueError("Covariance.__array__: a dtype cast requires a copy, but copy=False was given")
            return self._mat.astype(dtype)
        return self._mat.copy() if copy else self._mat

    def __getitem__(self, key):
        """Delegates to the underlying array (e.g. cov[:3, :3] for a sub-block)."""
        return self._mat[key]

    # ── diagnostics ──────────────────────────────────────────────────────────

    def diag(self) -> np.ndarray:
        """The variances (diagonal entries)."""
        return np.diag(self._mat)

    def std(self) -> np.ndarray:
        """Per-axis standard deviations (sqrt of the diagonal)."""
        return np.sqrt(np.clip(self.diag(), 0.0, None))

    def is_symmetric(self, atol: float = 1e-8) -> bool:
        return bool(np.allclose(self._mat, self._mat.T, atol=atol))

    def is_psd(self, atol: float = 1e-9) -> bool:
        """True if every eigenvalue is >= -atol (allows for tiny numerical noise)."""
        eigvals = np.linalg.eigvalsh(self._mat)
        return bool(np.all(eigvals >= -atol))

    def trace(self) -> float:
        return float(np.trace(self._mat))

    # ── arithmetic ───────────────────────────────────────────────────────────

    def __add__(self, other) -> 'Covariance':
        """C1 + C2: sum of two covariances (valid for independent random variables)."""
        if isinstance(other, Covariance):
            return Covariance(self._mat + other._mat)
        if isinstance(other, np.ndarray):
            return Covariance(self._mat + other)
        return NotImplemented

    def __radd__(self, other) -> 'Covariance':
        return self.__add__(other)

    def __sub__(self, other) -> 'Covariance':
        if isinstance(other, Covariance):
            return Covariance(self._mat - other._mat)
        if isinstance(other, np.ndarray):
            return Covariance(self._mat - other)
        return NotImplemented

    def __rsub__(self, other) -> 'Covariance':
        """other - self, where other is a raw ndarray (e.g. a Monte-Carlo-
        estimated covariance being compared against this analytic one)."""
        if isinstance(other, np.ndarray):
            return Covariance(other - self._mat)
        return NotImplemented

    def __mul__(self, scalar) -> 'Covariance':
        """C * k: scales the covariance (symmetry preserved; PSD preserved for k >= 0)."""
        if isinstance(scalar, (int, float, np.floating)):
            return Covariance(self._mat * float(scalar))
        return NotImplemented

    def __rmul__(self, scalar) -> 'Covariance':
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> 'Covariance':
        """C / k: scales the covariance (symmetry preserved; PSD preserved for k > 0)."""
        if isinstance(scalar, (int, float, np.floating)):
            return Covariance(self._mat / float(scalar))
        return NotImplemented

    def __matmul__(self, other):
        """C @ X: generic matrix product, returned as a raw ndarray, since a
        generic product doesn't preserve the covariance's symmetry. For the
        standard covariance-propagation form A @ C @ A^T, use .congruence(A)
        instead, which keeps the result wrapped as a Covariance."""
        other_arr = other._mat if isinstance(other, Covariance) else np.asarray(other, dtype=np.float64)
        return self._mat @ other_arr

    def __rmatmul__(self, other):
        other_arr = other._mat if isinstance(other, Covariance) else np.asarray(other, dtype=np.float64)
        return other_arr @ self._mat

    def congruence(self, A) -> 'Covariance':
        """
        Returns this covariance propagated through a linear map A:

            C' = A @ C @ A^T

        This is the standard way covariance propagates through a linear
        transform (rotations, Jacobians, adjoints) throughout this library --
        exposed as a named method since a raw `A @ C @ A.T` expression drops
        the Covariance wrapper partway through (see __matmul__ above), while
        this keeps the result validated and wrapped.
        """
        A = np.asarray(A, dtype=np.float64)
        return Covariance(A @ self._mat @ A.T)

    @property
    def T(self) -> 'Covariance':
        """A covariance matrix is symmetric, so its transpose is itself."""
        return Covariance(self._mat)

    def __eq__(self, other) -> bool:
        if isinstance(other, Covariance):
            return bool(np.array_equal(self._mat, other._mat))
        if isinstance(other, np.ndarray):
            return bool(np.array_equal(self._mat, other))
        return NotImplemented

    # Elementwise numeric comparisons (unlike __eq__ above, these return a raw
    # boolean ndarray, matching how `some_ndarray > 0` behaves -- needed since
    # this library uses patterns like `np.any(x.C > 0)` on covariance fields.
    def __gt__(self, other):
        other_arr = other._mat if isinstance(other, Covariance) else other
        return self._mat > other_arr

    def __lt__(self, other):
        other_arr = other._mat if isinstance(other, Covariance) else other
        return self._mat < other_arr

    def __ge__(self, other):
        other_arr = other._mat if isinstance(other, Covariance) else other
        return self._mat >= other_arr

    def __le__(self, other):
        other_arr = other._mat if isinstance(other, Covariance) else other
        return self._mat <= other_arr

    def __repr__(self):
        return f"Covariance(n={self.n}, diag={self.diag()})"
