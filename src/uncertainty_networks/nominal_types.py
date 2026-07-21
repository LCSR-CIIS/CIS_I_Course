from dataclasses import dataclass
import math
import numpy as np

@dataclass
class vct3:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        # Initialize internally as a [3, 1] column vector
        self._vec = np.array([[x],
                              [y],
                              [z]], dtype=np.float64)

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

    def __repr__(self):
        return f"vct3(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})\nInternal Vector:\n{self._vec}"


# TBD: rotations along multiple axis through the ZYX euler angle convention
# Add inverse of Rotation matrix
class Rot:
    def __init__(self, axis: str = None, angle: float = None, matrix: np.ndarray = None):
        if matrix is not None:
            self._matrix = np.array(matrix, dtype=np.float64)
        elif axis is not None and angle is not None:
            self._matrix = self._build_matrix(axis, angle)
        else:
            raise ValueError("Must provide either (axis and angle) or (matrix)")

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

# Add inverse of frame
class Frame:
    def __init__(self, R: Rot, p: vct3):
        self.R = R
        self.p = p

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