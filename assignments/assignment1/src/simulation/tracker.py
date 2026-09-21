"""
Simulation for Tracker: CreateTracker and
SampleMarkerBodies from the assignment PDF.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from data_types.nominal_types import Frame, vct3
from data_types.uncertain_types import uvct3, uFrame
from data_types.covariance_types import Covariance
from uncertainty_networks.se3 import exp_se3

from ..tracker import Tracker
from ..utils import sample_normal
from .marker_body import PlacedBody
from . import constants


@dataclass
class ObservedBody:
    """One marker body's result from a single SampleMarkerBodies call."""
    frame: Frame              # this body's pose relative to the tracker (Fptr,j / Fcal,j)
    markers: List[uvct3]      # the individual noisy marker readings behind that pose estimate


def create_tracker(name: str, F_approx: Frame) -> Tracker:
    """CreateTracker(F, {mB1, ..., mBN}) -> trk

    Only the tracker's approximate pose is handled here; which marker
    bodies to read is decided per-call by sample_marker_bodies.
    """
    return Tracker(name, F_approx)


def sample_marker_bodies(tracker: Tracker, placed_bodies: List[PlacedBody]) -> Dict[str, ObservedBody]:
    """{F1, ..., FN} = SampleMarkerBodies(trk)

    One simulated camera snapshot of every body in placed_bodies: a single
    shared tracker-pose jiggle is drawn for this call (the whole point of
    "one call reads every tracked body at once"), then each body's markers
    are read (independent per-marker sensor noise) and registered via
    Tracker.procrustes_solver.
    """
    F_nom = tracker.F.F if isinstance(tracker.F, uFrame) else tracker.F

    jiggle_cov = Covariance(np.diag([
        constants.TRACKER_JIGGLE_ANGLE_STD ** 2,
        constants.TRACKER_JIGGLE_ANGLE_STD ** 2,
        constants.TRACKER_JIGGLE_ANGLE_STD ** 2,
        constants.TRACKER_JIGGLE_POS_STD ** 2,
        constants.TRACKER_JIGGLE_POS_STD ** 2,
        constants.TRACKER_JIGGLE_POS_STD ** 2,
    ]))
    eta = sample_normal(np.zeros(6), cov=jiggle_cov)[0]
    F_actual = F_nom * Frame(exp_se3(eta))

    sensor_noise_cov = Covariance.eye(3, scale=constants.SENSOR_NOISE_STD ** 2)

    saved_F = tracker.F
    tracker.F = F_actual
    try:
        results: Dict[str, ObservedBody] = {}
        for placed in placed_bodies:
            observed = tracker.read_marker_body(
                placed.body, sensor_noise_cov=sensor_noise_cov, body_placement=placed.actual_placement,
            )
            frame_est = tracker.procrustes_solver(placed.body.nominal_marker_positions, observed)
            results[placed.body.name] = ObservedBody(frame=frame_est, markers=observed)
        return results
    finally:
        tracker.F = saved_F
