"""
Simulation for Pointer: defining a pointer's approximate
tip position on top of an already-defined marker body.
"""

from data_types.nominal_types import vct3
from data_types.covariance_types import Covariance

from ..marker import MarkerBody
from ..pointer import Pointer
from . import constants


def define_pointer(name: str, nominal_tip_position: vct3, marker_body: MarkerBody) -> Pointer:
    """Defines a pointer's approximate tip position p_tip relative to the
    pointer coordinate system, and immediately assigns it the (hidden)
    manufacturing-noise covariance every real pointer has, so it's ready
    to be read via get_actual_tip_position().
    """
    ptr = Pointer(name, nominal_tip_position, marker_body)
    ptr.set_manufacturing_cov(Covariance.eye(3, scale=constants.MARKER_MANUFACTURING_STD ** 2))
    return ptr
