"""
Simulation for the calibration object's sensing
behavior: hand-guiding the pointer near it, and reading its high-accuracy
tip sensor.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame
from data_types.covariance_types import Covariance
from uncertainty_networks.se3 import exp_se3

from ..pointer import Pointer
from ..utils import sample_normal
from .marker_body import PlacedBody
from . import constants


@dataclass
class HandGuideResult:
    """Opaque handle to one hand-guide event's actual (hidden) pointer-
    relative-to-calibration-object pose, needed by sense_tip() to know
    where the tip actually ended up."""
    F_ptr_rel_cal_actual: Frame


def hand_guide_pointer(pointer: Pointer, pointer_placed: PlacedBody,
                        cal_placed: PlacedBody, R_desired: Rot) -> HandGuideResult:
    """Simulates hand-guiding the robot so the pointer's tip is within
    approximately 5mm of the calibration object's origin, with the
    pointer's orientation approximately R_desired relative to the
    calibration object.

    Updates pointer_placed.actual_placement in place, so a subsequent
    sample_marker_bodies([..., pointer_placed, ...]) call reads the
    pointer's new physical pose.
    """
    R_desired_mat = R_desired.matrix
    p_tip_actual = pointer.get_actual_tip_position().p

    t_vec = -(R_desired_mat @ p_tip_actual.vec)
    F_ptr_rel_cal_nominal = Frame(Rot(matrix=R_desired_mat), vct3(t_vec[0, 0], t_vec[1, 0], t_vec[2, 0]))

    hand_guide_cov = Covariance(np.diag([
        constants.HAND_GUIDE_ANGLE_STD ** 2,
        constants.HAND_GUIDE_ANGLE_STD ** 2,
        constants.HAND_GUIDE_ANGLE_STD ** 2,
        constants.HAND_GUIDE_POS_STD ** 2,
        constants.HAND_GUIDE_POS_STD ** 2,
        constants.HAND_GUIDE_POS_STD ** 2,
    ]))
    eta = sample_normal(np.zeros(6), cov=hand_guide_cov)[0]
    F_ptr_rel_cal_actual = Frame(exp_se3(eta)) * F_ptr_rel_cal_nominal

    pointer_placed.actual_placement = cal_placed.actual_placement * F_ptr_rel_cal_actual
    return HandGuideResult(F_ptr_rel_cal_actual=F_ptr_rel_cal_actual)


def sense_tip(pointer: Pointer, hand_guide_result: HandGuideResult) -> Optional[vct3]:
    """Reads the calibration sensor: the pointer tip's position relative to
    the calibration object, to near-zero error, if the tip is currently
    within CAL_SENSOR_RANGE_MM of the sensor's origin. Returns None
    (an ordinary, expected outcome -- not an error) if out of range.
    """
    p_tip_actual = pointer.get_actual_tip_position().p
    p_in_cal_frame = hand_guide_result.F_ptr_rel_cal_actual * p_tip_actual

    sensor_cov = Covariance.eye(3, scale=constants.CAL_SENSOR_NOISE_STD ** 2)
    p_sensed = sample_normal(p_in_cal_frame, cov=sensor_cov)

    if p_sensed.norm() <= constants.CAL_SENSOR_RANGE_MM:
        return p_sensed
    return None
