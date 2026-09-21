"""
Simulation operations for MarkerBody: DefineMarkerBody and
PlaceMarkerBody.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from data_types.nominal_types import vct3, Frame
from data_types.covariance_types import Covariance
from uncertainty_networks.se3 import exp_se3

from ..marker import MarkerBody
from ..utils import sample_normal
from . import constants


@dataclass
class PlacedBody:
    """A MarkerBody together with its hidden, ground-truth actual placement
    in the workspace -- kept separate from MarkerBody.world_frame, which
    holds the visible, approximate (nominal) placement a student specified.

    Mutable on purpose: hand_guide_pointer() updates a pointer's
    actual_placement in place every time the pointer is re-guided.
    """
    body: MarkerBody
    actual_placement: Frame


def define_marker_body(name: str, nominal_positions: List[vct3]) -> MarkerBody:
    """DefineMarkerBody({a_m,1, ..., a_m,N}) -> mB

    Builds a MarkerBody from its approximate local marker-sphere layout and
    immediately assigns it the (hidden) manufacturing-noise covariance
    every real marker body has, so it's ready to be read by a tracker.
    """
    body = MarkerBody(name, nominal_positions, world_frame=Frame.eye())
    body.set_manufacturing_cov(Covariance.eye(3, scale=constants.MARKER_MANUFACTURING_STD ** 2))
    return body


def place_marker_body(body: MarkerBody, F_approx: Frame, precision: str = "coarse") -> PlacedBody:
    """PlaceMarkerBody(mB, Fmb)

    Records F_approx as the body's visible, nominal placement, and
    separately samples the hidden actual placement
    F*_mB = delta_F_mB * F_mB (delta composed on the left, per the PDF),
    returned as a PlacedBody for later use with sample_marker_bodies /
    hand_guide_pointer.

    precision="coarse" (default) models a roughly, manually placed static
    body (e.g. the calibration object) with the PDF's
    ||alpha_cal||<=0.1, ||epsilon_cal||<=25mm bounds. precision="precise"
    models a robot precisely moving to a *commanded* pose (e.g. step 8's
    test poses, where "delta_Ft is relatively small") instead.
    """
    if precision == "coarse":
        angle_std, pos_std = constants.BODY_PLACEMENT_ANGLE_STD, constants.BODY_PLACEMENT_POS_STD
    elif precision == "precise":
        angle_std, pos_std = constants.TEST_POSE_ANGLE_STD, constants.TEST_POSE_POS_STD
    else:
        raise ValueError(f"place_marker_body: precision must be 'coarse' or 'precise', got {precision!r}")

    body.update_world_frame(F_approx)

    cov6 = Covariance(np.diag([
        angle_std ** 2, angle_std ** 2, angle_std ** 2,
        pos_std ** 2, pos_std ** 2, pos_std ** 2,
    ]))
    eta = sample_normal(np.zeros(6), cov=cov6)[0]
    delta_F = Frame(exp_se3(eta))
    actual_placement = delta_F * F_approx

    return PlacedBody(body=body, actual_placement=actual_placement)
