from dataclasses import dataclass
import math
import numpy as np

from uncertainty_networks.se3 import skew as _se3_skew, skew_axis as _se3_skew_axis

@dataclass
class vct3:
    """
    A 3-D point/vector.

    ::

        vct3()            null (zero) vector
        vct3(x, y, z)     three scalars
        vct3(other_vct3)  copy
        vct3([x, y, z])   array-like (list/tuple/ndarray of length 3)
    """
    x: float
    y: float
    z: float

    def __init__(self, x=None, y=None, z=None):
        if x is None and y is None and z is None:
            x, y, z = 0.0, 0.0, 0.0
        elif y is None and z is None:
            if isinstance(x, vct3):
                x, y, z = x.x, x.y, x.z
            else:
                arr = np.asarray(x, dtype=np.float64).ravel()
                if arr.shape[0] != 3:
                    raise ValueError(
                        f"vct3: array-like argument must have exactly 3 elements, got {arr.shape[0]}"
                    )
                x, y, z = float(arr[0]), float(arr[1]), float(arr[2])
        elif x is None or y is None or z is None:
            raise TypeError(
                "vct3() requires either a single vct3/array-like argument, or three scalars x, y, z"
            )
        # Initialize internally as a [3, 1] column vector
        self._vec = np.array([[x],
                              [y],
                              [z]], dtype=np.float64)

    @classmethod
    def null(cls) -> 'vct3':
        """The zero vector (0, 0, 0)."""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def rand(cls, low: float = -1.0, high: float = 1.0) -> 'vct3':
        """A vector with components drawn uniformly from [low, high)."""
        vals = np.random.uniform(low, high, size=3)
        return cls(vals[0], vals[1], vals[2])

    # Properties to easily read x, y, z back out as scalar values
    @property
    def x(self) -> float: return float(self._vec[0, 0])
    @property
    def y(self) -> float: return float(self._vec[1, 0])
    @property
    def z(self) -> float: return float(self._vec[2, 0])

    @property
    def vec(self) -> np.ndarray:
        """Returns the raw [3, 1] NumPy array."""
        return self._vec

    def __add__(self, other: 'vct3') -> 'vct3':
        if not isinstance(other, vct3):
            return NotImplemented
        res_vec = self._vec + other.vec
        return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])

    def __sub__(self, other: 'vct3') -> 'vct3':
        if not isinstance(other, vct3):
            return NotImplemented
        res_vec = self._vec - other.vec
        return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])

    def __neg__(self) -> 'vct3':
        res_vec = -self._vec
        return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])

    def __mul__(self, other):
        """vct3 * vct3 -> dot product (float); vct3 * scalar -> scaled vct3."""
        if isinstance(other, vct3):
            return self.dot(other)
        if isinstance(other, (int, float, np.floating)):
            res_vec = self._vec * other
            return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(self, scalar) -> 'vct3':
        if not isinstance(scalar, (int, float, np.floating)):
            return NotImplemented
        res_vec = self._vec / scalar
        return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])

    def __abs__(self) -> 'vct3':
        """Elementwise absolute value (matches uvct3.m's documented abs()
        convention: `vct3(abs(v1.el))` -- a vector, not the norm)."""
        res_vec = np.abs(self._vec)
        return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])

    def times(self, other: 'vct3') -> 'vct3':
        """Elementwise (Hadamard) product."""
        if not isinstance(other, vct3):
            raise TypeError("vct3.times() requires another vct3")
        res_vec = self._vec * other.vec
        return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])

    def dot(self, other: 'vct3') -> float:
        if not isinstance(other, vct3):
            raise TypeError("vct3.dot() requires another vct3")
        return float(np.dot(self._vec.ravel(), other.vec.ravel()))

    def cross(self, other: 'vct3') -> 'vct3':
        if not isinstance(other, vct3):
            raise TypeError("vct3.cross() requires another vct3")
        res = np.cross(self._vec.ravel(), other.vec.ravel())
        return vct3(res[0], res[1], res[2])

    def norm(self, p: int = 2) -> float:
        """The p-norm of this vector (Euclidean by default)."""
        return float(np.linalg.norm(self._vec.ravel(), ord=p))

    def distance(self, other: 'vct3') -> float:
        if not isinstance(other, vct3):
            raise TypeError("vct3.distance() requires another vct3")
        return (self - other).norm()

    def unit(self) -> 'vct3':
        """This vector normalized to unit length."""
        n = self.norm()
        if n < 1e-15:
            raise ValueError("vct3.unit(): cannot normalize a zero-length vector")
        res_vec = self._vec / n
        return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])

    def skew(self) -> np.ndarray:
        """The 3x3 skew-symmetric cross-product matrix [self]x."""
        return _se3_skew(self._vec.ravel())

    @classmethod
    def from_skew(cls, M: np.ndarray) -> 'vct3':
        """Inverse of skew(): the vct3 w such that w.skew() == M."""
        w = _se3_skew_axis(M)
        return cls(w[0], w[1], w[2])

    def homog(self) -> np.ndarray:
        """The [4, 1] homogeneous column vector [x; y; z; 1]."""
        return np.array([[self.x], [self.y], [self.z], [1.0]], dtype=np.float64)

    def __repr__(self):
        return f"vct3(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})\nInternal Vector:\n{self._vec}"


class Rot:
    """
    A 3x3 rotation matrix.

    ::

        Rot()                      identity
        Rot(axis='z', angle=0.5)   axis name ('x'/'y'/'z') + angle (radians)
        Rot(axis_vct3, angle=0.5)  arbitrary rotation axis (vct3/array-like) + angle (radians)
        Rot(matrix=M)              3x3 rotation matrix, keyword
        Rot(M)                     3x3 rotation matrix, positional
        Rot(other_rot)             copy
    """

    def __init__(self, axis=None, angle: float = None, matrix: np.ndarray = None):
        if isinstance(axis, Rot):
            if angle is not None or matrix is not None:
                raise TypeError("Rot: cannot combine a Rot copy-argument with angle/matrix")
            self._matrix = axis.matrix.copy()
            return

        if axis is None or isinstance(axis, str):
            if matrix is not None:
                self._matrix = np.array(matrix, dtype=np.float64)
            elif axis is not None and angle is not None:
                self._matrix = self._build_matrix(axis, angle)
            elif axis is None and angle is None:
                self._matrix = np.eye(3, dtype=np.float64)
            else:
                raise ValueError("Must provide either (axis and angle) or (matrix)")
            return

        # axis is neither a Rot, None, nor a string: either a positional 3x3
        # matrix, or an arbitrary rotation axis (length-3 vector) + angle=.
        if matrix is not None:
            raise TypeError("Rot: cannot combine a positional array-like argument with matrix=")

        axis_vec = axis.vec.ravel() if isinstance(axis, vct3) else np.asarray(axis, dtype=np.float64)

        if axis_vec.shape == (3, 3):
            if angle is not None:
                raise TypeError("Rot: cannot combine a positional matrix-like argument with angle")
            self._matrix = axis_vec
            return

        axis_vec = axis_vec.ravel()
        if axis_vec.shape == (3,):
            if angle is None:
                raise TypeError("Rot: an arbitrary rotation axis requires angle=...")
            from uncertainty_networks.se3 import exp_so3 as _exp_so3
            axis_norm = float(np.linalg.norm(axis_vec))
            if axis_norm < 1e-15:
                raise ValueError("Rot: rotation axis must be nonzero")
            unit_axis = axis_vec / axis_norm
            self._matrix = _exp_so3(angle * unit_axis)
            return

        raise ValueError(
            f"Rot: positional argument must be a 3x3 matrix-like or a length-3 axis "
            f"vector, got shape {axis_vec.shape}"
        )

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def _build_matrix(self, axis: str, angle: float) -> np.ndarray:
        c = np.cos(angle)
        s = np.sin(angle)
        ax = axis.lower()
        if ax == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif ax == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif ax == 'z':
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def inv(self) -> 'Rot':
        """Returns the inverse rotation nominal matrix."""
        return Rot(matrix=self._matrix.T)

    def __mul__(self, other):
        if isinstance(other, vct3):
            # [3,3] matrix @ [3,1] vector -> yields [3,1] matrix result
            res_vec = self._matrix @ other.vec
            return vct3(res_vec[0, 0], res_vec[1, 0], res_vec[2, 0])
        elif isinstance(other, Rot):
            return Rot(matrix=self._matrix @ other.matrix)
        return NotImplemented

    @classmethod
    def eye(cls) -> 'Rot':
        """The identity rotation."""
        return cls()

    @classmethod
    def xyz(cls, rx: float, ry: float, rz: float) -> 'Rot':
        """
        Composed rotation from XYZ Euler angles (radians), applied as
        R = Rot('x', rx) * Rot('y', ry) * Rot('z', rz) (X first, then Y, then Z).

        Verified against RotMx.xyz's literal closed-form matrix formula
        (from uRotMx.m's static methods) — this composition order is the one
        that reproduces it exactly; the reverse order does not.
        """
        return cls(axis='x', angle=rx) * cls(axis='y', angle=ry) * cls(axis='z', angle=rz)

    @classmethod
    def xyz_deg(cls, rx_deg: float, ry_deg: float, rz_deg: float) -> 'Rot':
        """Same as xyz(), but angles are given in degrees."""
        return cls.xyz(math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg))

    @classmethod
    def rand(cls) -> 'Rot':
        """A uniformly random rotation (random axis + random angle in [0, 2*pi))."""
        axis_vec = np.random.normal(size=3)
        axis_vec /= np.linalg.norm(axis_vec)
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        return cls(axis_vec, angle=angle)

    @classmethod
    def rand_deg(cls) -> 'Rot':
        """Alias for rand() (provided for naming parity with the MATLAB source;
        a full random SO(3) sample has no meaningful degrees-vs-radians distinction)."""
        return cls.rand()

    @classmethod
    def rand_xyz(cls, aX: float, aY: float, aZ: float) -> 'Rot':
        """
        A random rotation from bounded Euler angles, each drawn uniformly from
        [-aX, aX], [-aY, aY], [-aZ, aZ] radians. Ported from uRotMx.rand(aX,aY,aZ)
        (which, despite living on the uncertain uRotMx class in the MATLAB
        source, returns a plain certain RotMx — matched here on Rot).
        """
        rx = 2 * aX * np.random.rand() - aX
        ry = 2 * aY * np.random.rand() - aY
        rz = 2 * aZ * np.random.rand() - aZ
        return cls.xyz(rx, ry, rz)

    @classmethod
    def rand_xyz_deg(cls, aX_deg: float, aY_deg: float, aZ_deg: float) -> 'Rot':
        """Same as rand_xyz(), but the bounds and result are in degrees (uRotMx.randD)."""
        return cls.rand_xyz(math.radians(aX_deg), math.radians(aY_deg), math.radians(aZ_deg))

    def normalize(self) -> 'Rot':
        """
        Returns the nearest proper rotation matrix to this one (via SVD),
        correcting for numerical drift away from orthonormality.

        Note: the MATLAB source's literal formula (R = V*U') computes the
        nearest orthogonal matrix to this matrix's TRANSPOSE, not to this
        matrix itself (verified numerically to be ~3x farther from the
        input than the correct answer, and in fact exactly its transpose).
        This implementation uses the correct formula, U @ Vh.
        """
        U, _, Vh = np.linalg.svd(self._matrix)
        if np.linalg.det(U @ Vh) < 0:
            U = U.copy()
            U[:, -1] *= -1
        return Rot(matrix=U @ Vh)

    def axis_angle(self):
        """Returns (axis: vct3, angle: float) such that self == Rot(axis, angle=angle)."""
        from uncertainty_networks.se3 import log_so3 as _log_so3
        phi = _log_so3(self._matrix)
        angle = float(np.linalg.norm(phi))
        if angle < 1e-12:
            return vct3(0.0, 0.0, 0.0), 0.0
        axis_vec = phi / angle
        return vct3(axis_vec[0], axis_vec[1], axis_vec[2]), angle

    @classmethod
    def from_quaternion(cls, q) -> 'Rot':
        """Builds a Rot from a Quaternion instance."""
        return cls(matrix=q.to_matrix())

    def to_quaternion(self):
        """Converts this rotation to a Quaternion instance."""
        from .quaternion import Quaternion
        return Quaternion.from_matrix(self._matrix)

    @classmethod
    def cayley(cls, a) -> 'Rot':
        """
        Builds a Rot via the Cayley transform of a 3-vector `a` (vct3 or
        array-like): a rational alternative to the exponential-map
        constructor (Rot(axis, angle=...)). See se3.cayley_transform().
        """
        from .se3 import cayley_transform as _cayley_transform
        a_vec = a.vec.ravel() if isinstance(a, vct3) else np.asarray(a, dtype=np.float64).ravel()
        return cls(matrix=_cayley_transform(a_vec))

class Frame:
    """
    A rigid transform: a rotation (Rot) plus a translation (vct3).

    ::

        Frame(R, p)          Rot + vct3 (p may also be array-like)
        Frame(matrix)        4x4 homogeneous transform, single positional arg
        Frame(other_frame)   copy
        Frame(a_vct3)        identity rotation + that point
        Frame(a_rot)         that rotation + zero translation
    """

    def __init__(self, R=None, p=None):
        if isinstance(R, Frame):
            if p is not None:
                raise TypeError("Frame: cannot combine a Frame copy-argument with a p argument")
            self.R = R.R
            self.p = R.p
            return

        if p is None and isinstance(R, vct3):
            self.R = Rot()
            self.p = R
            return

        if p is None and isinstance(R, Rot):
            self.R = R
            self.p = vct3()
            return

        if p is None and R is not None and not isinstance(R, Rot):
            M = np.asarray(R, dtype=np.float64)
            if M.shape != (4, 4):
                raise ValueError(
                    f"Frame: single positional argument must be a Frame, a Rot (with p), "
                    f"or a 4x4 matrix-like; got shape {M.shape}"
                )
            self.R = Rot(matrix=M[:3, :3])
            self.p = vct3(M[0, 3], M[1, 3], M[2, 3])
            return

        if R is None or p is None:
            raise TypeError(
                "Frame() requires either a Frame instance, a 4x4 matrix-like, "
                "or both R (Rot) and p (vct3/array-like)"
            )
        # R is coerced via Rot(...) if it isn't already one (matrix-like or
        # a bare axis+angle-less array all raise clearly from within Rot()),
        # mirroring uFrame.m's two-argument constructor branch, which
        # coerces both arguments instead of just p.
        self.R = R if isinstance(R, Rot) else Rot(R)
        self.p = p if isinstance(p, vct3) else vct3(p)

    def __mul__(self, other):
        if isinstance(other, vct3):
            # F * p1 = F.R * p1 + F.p
            return (self.R * other) + self.p
        elif isinstance(other, Frame):
            # F2 * F1 = F2.R * F1 + F2.p
            return Frame(R=self.R * other.R, p=self.R * other.p + self.p)
        return NotImplemented

    def inv(self) -> 'Frame':
        """Returns the inverse homogeneous transformation frame."""
        R_inv = self.R.inv()
        # p_inv = -R^T * p
        p_inv_vec = -R_inv.matrix @ self.p.vec
        p_inv = vct3(p_inv_vec[0, 0], p_inv_vec[1, 0], p_inv_vec[2, 0])
        return Frame(R=R_inv, p=p_inv)

    @classmethod
    def eye(cls) -> 'Frame':
        """The identity transform (identity rotation, zero translation)."""
        return cls(Rot(), vct3())


class vct3Array:
    """
    A batch/array of 3-D points, stored internally as a (3, N) NumPy array.

    ::

        vct3Array()                 empty array (N=0)
        vct3Array(n)                n zero-filled points
        vct3Array(other_array)      copy
        vct3Array(a_vct3)           single-point array (N=1)
        vct3Array([v1, v2, ...])    list/tuple of vct3 instances
        vct3Array(matrix_like)      (3, N) or (N, 3) array-like
    """

    def __init__(self, arg=None):
        if arg is None:
            self._mat = np.zeros((3, 0), dtype=np.float64)
            return
        if isinstance(arg, vct3Array):
            self._mat = arg._mat.copy()
            return
        if isinstance(arg, vct3):
            self._mat = arg.vec.copy()
            return
        if isinstance(arg, (int, np.integer)) and not isinstance(arg, bool):
            self._mat = np.zeros((3, int(arg)), dtype=np.float64)
            return
        if isinstance(arg, (list, tuple)) and len(arg) > 0 and all(isinstance(v, vct3) for v in arg):
            self._mat = np.hstack([v.vec for v in arg])
            return
        M = np.asarray(arg, dtype=np.float64)
        if M.ndim != 2 or 3 not in M.shape:
            raise ValueError(f"vct3Array: matrix-like argument must be (3,N) or (N,3), got shape {M.shape}")
        if M.shape[0] != 3:
            M = M.T
        self._mat = M.copy()

    @property
    def mat(self) -> np.ndarray:
        """Returns the raw (3, N) NumPy array."""
        return self._mat

    @property
    def n(self) -> int:
        return self._mat.shape[1]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> vct3:
        col = self._mat[:, idx]
        return vct3(col[0], col[1], col[2])

    def __setitem__(self, idx: int, value: vct3):
        if not isinstance(value, vct3):
            raise TypeError("vct3Array assignment requires a vct3")
        self._mat[:, idx] = value.vec.ravel()

    @property
    def x(self) -> np.ndarray:
        return self._mat[0, :].copy()

    @property
    def y(self) -> np.ndarray:
        return self._mat[1, :].copy()

    @property
    def z(self) -> np.ndarray:
        return self._mat[2, :].copy()

    @classmethod
    def null(cls, n: int) -> 'vct3Array':
        """n zero-filled points."""
        return cls(n)

    @classmethod
    def rand(cls, n: int, low: float = -1.0, high: float = 1.0) -> 'vct3Array':
        """n points with components drawn uniformly from [low, high)."""
        arr = cls(n)
        arr._mat = np.random.uniform(low, high, size=(3, n))
        return arr

    def concatenate(self, other: 'vct3Array') -> 'vct3Array':
        result = vct3Array()
        result._mat = np.hstack([self._mat, other._mat])
        return result

    def __add__(self, other):
        if isinstance(other, vct3Array):
            if other.n != self.n:
                raise ValueError("vct3Array addition requires matching lengths")
            result = vct3Array()
            result._mat = self._mat + other._mat
            return result
        if isinstance(other, vct3):
            result = vct3Array()
            result._mat = self._mat + other.vec
            return result
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, vct3Array):
            if other.n != self.n:
                raise ValueError("vct3Array subtraction requires matching lengths")
            result = vct3Array()
            result._mat = self._mat - other._mat
            return result
        if isinstance(other, vct3):
            result = vct3Array()
            result._mat = self._mat - other.vec
            return result
        return NotImplemented

    def times(self, other: 'vct3Array') -> 'vct3Array':
        """Elementwise (Hadamard) product against another vct3Array of matching length."""
        if not isinstance(other, vct3Array) or other.n != self.n:
            raise TypeError("vct3Array.times() requires another vct3Array of matching length")
        result = vct3Array()
        result._mat = self._mat * other._mat
        return result

    def __truediv__(self, scalar) -> 'vct3Array':
        if not isinstance(scalar, (int, float, np.floating)):
            return NotImplemented
        result = vct3Array()
        result._mat = self._mat / scalar
        return result

    def __abs__(self) -> 'vct3Array':
        """Elementwise absolute value (a vct3Array), consistent with vct3's
        abs() convention -- distinct from .norm() (per-column magnitude)."""
        result = vct3Array()
        result._mat = np.abs(self._mat)
        return result

    def norm(self) -> np.ndarray:
        """Per-column Euclidean norm, shape (N,)."""
        return np.linalg.norm(self._mat, axis=0)

    def unit(self) -> 'vct3Array':
        """Each column normalized to unit length."""
        norms = self.norm()
        if np.any(norms < 1e-15):
            raise ValueError("vct3Array.unit(): cannot normalize a zero-length column")
        result = vct3Array()
        result._mat = self._mat / norms
        return result

    def mean(self) -> vct3:
        m = self._mat.mean(axis=1)
        return vct3(m[0], m[1], m[2])

    def cov(self) -> np.ndarray:
        """The 3x3 sample covariance matrix of the points (ddof=1)."""
        if self.n < 2:
            raise ValueError("vct3Array.cov() requires at least 2 points")
        return np.cov(self._mat, ddof=1)

    def cov_frame(self) -> Frame:
        """
        A Frame at the centroid of this point cloud, whose rotation's columns
        are the eigenvectors of the point covariance (largest-variance axis
        first) -- a principal-axis frame for the point cloud.
        """
        centroid = self.mean()
        C = self.cov()
        eigvals, eigvecs = np.linalg.eigh(C)
        order = np.argsort(eigvals)[::-1]
        R_mat = eigvecs[:, order].copy()
        if np.linalg.det(R_mat) < 0:
            R_mat[:, -1] *= -1
        return Frame(Rot(matrix=R_mat), centroid)

    def __repr__(self):
        return f"vct3Array(n={self.n})\n{self._mat}"


class Line:
    """
    An infinite 3-D line: a point p on the line plus a unit direction n.

    ::

        Line(p, n)          point (vct3/array-like) + direction (vct3/array-like)
        Line(other_line)    copy
    """

    def __init__(self, p=None, n=None):
        if isinstance(p, Line):
            if n is not None:
                raise TypeError("Line: cannot combine a Line copy-argument with n")
            self.p = p.p
            self.n = p.n
            return
        if p is None or n is None:
            raise TypeError("Line() requires either a Line instance, or both p (vct3) and n (vct3)")
        # NOTE: the MATLAB source has an apparent copy-paste bug where pV is
        # never coerced to vct3 in one constructor branch; both p and n are
        # coerced here, matching the obviously-intended behavior.
        self.p = p if isinstance(p, vct3) else vct3(p)
        n_vct = n if isinstance(n, vct3) else vct3(n)
        self.n = n_vct.unit()

    def closest_point(self, point: vct3) -> vct3:
        """The point on this line closest to `point`."""
        w = point - self.p
        t = w.dot(self.n)
        return self.p + self.n * t

    def distance(self, point: vct3) -> float:
        """Distance from `point` to this line."""
        return point.distance(self.closest_point(point))

    def __repr__(self):
        return f"Line(p={self.p!r}, n={self.n!r})"


class Plane:
    """
    An infinite 3-D plane: a point p on the plane plus a unit normal n.

    ::

        Plane(p, n)          point (vct3/array-like) + normal (vct3/array-like)
        Plane(other_plane)   copy
    """

    def __init__(self, p=None, n=None):
        if isinstance(p, Plane):
            if n is not None:
                raise TypeError("Plane: cannot combine a Plane copy-argument with n")
            self.p = p.p
            self.n = p.n
            return
        if p is None or n is None:
            raise TypeError("Plane() requires either a Plane instance, or both p (vct3) and n (vct3)")
        self.p = p if isinstance(p, vct3) else vct3(p)
        n_vct = n if isinstance(n, vct3) else vct3(n)
        self.n = n_vct.unit()

    def distance(self, point: vct3) -> float:
        """Signed distance from `point` to this plane (positive on the side n points toward)."""
        if not isinstance(point, vct3):
            point = vct3(point)
        return (point - self.p).dot(self.n)

    def closest_point(self, point: vct3) -> vct3:
        """The point on this plane closest to `point`."""
        if not isinstance(point, vct3):
            point = vct3(point)
        d = self.distance(point)
        return point - self.n * d

    def __repr__(self):
        return f"Plane(p={self.p!r}, n={self.n!r})"


def dist_to_plane(point: vct3, plane_point: vct3, plane_normal: vct3) -> float:
    """
    Standalone signed-distance-to-plane function, mirroring MATLAB's DistToPlane.m.

    Unlike Plane.distance(), this does NOT normalize `plane_normal` first --
    passing in a non-unit normal scales the result by its magnitude.
    """
    if not isinstance(point, vct3):
        point = vct3(point)
    if not isinstance(plane_point, vct3):
        plane_point = vct3(plane_point)
    if not isinstance(plane_normal, vct3):
        plane_normal = vct3(plane_normal)
    return (point - plane_point).dot(plane_normal)


class Sphere:
    """
    A sphere: a center point p and radius r.

    Note: MATLAB's Sphere.m names its constructor ``uSphere`` -- a mismatch
    with the classdef name ``Sphere`` that means it never actually runs in
    the original source. This is a fixed, working constructor.

    Also note: distance() below is ported exactly as written in the MATLAB
    source -- it computes distance to the CENTER and never subtracts the
    stored radius.

    ::

        Sphere(p, r)          center (vct3/array-like) + radius (float)
        Sphere(other_sphere)  copy
    """

    def __init__(self, p=None, r=None):
        if isinstance(p, Sphere):
            if r is not None:
                raise TypeError("Sphere: cannot combine a Sphere copy-argument with r")
            self.p = p.p
            self.r = p.r
            return
        if p is None or r is None:
            raise TypeError("Sphere() requires either a Sphere instance, or both p (vct3) and r (float)")
        self.p = p if isinstance(p, vct3) else vct3(p)
        self.r = float(r)

    def distance(self, point: vct3) -> float:
        """Distance from `point` to the sphere's CENTER (matches the MATLAB source; ignores radius)."""
        if not isinstance(point, vct3):
            point = vct3(point)
        return point.distance(self.p)

    def __repr__(self):
        return f"Sphere(p={self.p!r}, r={self.r})"


class vct3BoundingBox:
    """
    An axis-aligned 3-D bounding box, defined by min/max corners.

    ::

        vct3BoundingBox(vmin, vmax)     two vct3/array-like corners (order doesn't matter)
        vct3BoundingBox(a_vct3_array)   the tightest box enclosing all points
        vct3BoundingBox(other_box)      copy
    """

    def __init__(self, a=None, b=None):
        if isinstance(a, vct3BoundingBox):
            if b is not None:
                raise TypeError("vct3BoundingBox: cannot combine a copy-argument with a second argument")
            self.vmin = a.vmin
            self.vmax = a.vmax
            return
        if isinstance(a, vct3Array):
            if b is not None:
                raise TypeError("vct3BoundingBox: cannot combine a vct3Array argument with a second argument")
            mat = a.mat
            if mat.shape[1] == 0:
                raise ValueError("vct3BoundingBox: cannot bound an empty vct3Array")
            self.vmin = vct3(mat[0, :].min(), mat[1, :].min(), mat[2, :].min())
            self.vmax = vct3(mat[0, :].max(), mat[1, :].max(), mat[2, :].max())
            return
        if a is None or b is None:
            raise TypeError(
                "vct3BoundingBox() requires either a vct3BoundingBox instance, "
                "a vct3Array, or both vmin and vmax (vct3/array-like)"
            )
        vmin_in = a if isinstance(a, vct3) else vct3(a)
        vmax_in = b if isinstance(b, vct3) else vct3(b)
        self.vmin = vct3(min(vmin_in.x, vmax_in.x), min(vmin_in.y, vmax_in.y), min(vmin_in.z, vmax_in.z))
        self.vmax = vct3(max(vmin_in.x, vmax_in.x), max(vmin_in.y, vmax_in.y), max(vmin_in.z, vmax_in.z))

    def includes(self, point):
        """True if `point` (a vct3) lies within this box, inclusive of the boundary.
        If `point` is a vct3Array, returns a list of bools, one per column."""
        if isinstance(point, vct3):
            return (self.vmin.x <= point.x <= self.vmax.x and
                    self.vmin.y <= point.y <= self.vmax.y and
                    self.vmin.z <= point.z <= self.vmax.z)
        if isinstance(point, vct3Array):
            # NOTE: fixes an apparent bug in the MATLAB source, which references
            # a nonexistent `B.v` inside a per-element loop; this checks each
            # column individually, as clearly intended.
            return [self.includes(point[i]) for i in range(point.n)]
        raise TypeError("vct3BoundingBox.includes() requires a vct3 or vct3Array")

    def overlaps(self, other: 'vct3BoundingBox') -> bool:
        """True if this box and `other` share any interior/boundary points."""
        if not isinstance(other, vct3BoundingBox):
            raise TypeError("vct3BoundingBox.overlaps() requires another vct3BoundingBox")
        return (self.vmin.x <= other.vmax.x and other.vmin.x <= self.vmax.x and
                self.vmin.y <= other.vmax.y and other.vmin.y <= self.vmax.y and
                self.vmin.z <= other.vmax.z and other.vmin.z <= self.vmax.z)

    def __add__(self, other):
        """
        vct3BoundingBox + vct3BoundingBox: elementwise sum of both corners
        (a Minkowski-sum-like expansion -- NOT a union of the two boxes).
        vct3BoundingBox + scalar: expand vmin/vmax outward by that scalar
        along every axis.
        """
        if isinstance(other, vct3BoundingBox):
            return vct3BoundingBox(self.vmin + other.vmin, self.vmax + other.vmax)
        if isinstance(other, (int, float, np.floating)):
            expand = vct3(other, other, other)
            return vct3BoundingBox(self.vmin - expand, self.vmax + expand)
        return NotImplemented

    def sample(self) -> vct3:
        """A uniformly random point inside this box."""
        x = np.random.uniform(self.vmin.x, self.vmax.x)
        y = np.random.uniform(self.vmin.y, self.vmax.y)
        z = np.random.uniform(self.vmin.z, self.vmax.z)
        return vct3(x, y, z)

    def __repr__(self):
        return f"vct3BoundingBox(vmin={self.vmin!r}, vmax={self.vmax!r})"