# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
examples.py

Reusable network builders for simulations (no default evaluation pairs).
"""

import numpy as np
from .se3 import make_se3, rotz
from .uncertain_geometry import UncertainTransform
from .uncertain_types import uvct3
from .network import GeometricNetwork


def build_shared_infrastructure_network() -> GeometricNetwork:
    """
    Build a network containing tracker/CT/robot-base shared infrastructure.

    This function ONLY builds the network. It does not choose any default point pair.
    """
    net = GeometricNetwork()
    net.add_frame("W")

    # Shared infrastructure (connect into W)
    net.add_edge("Trk", "W", UncertainTransform(make_se3(rotz(0.02), [0.02, 0.00, 0.01]), np.diag([4e-6]*6)), add_inverse=True)
    net.add_edge("CT",  "W", UncertainTransform(make_se3(rotz(-0.015), [0.03, -0.01, 0.00]), np.diag([5e-6]*6)), add_inverse=True)
    net.add_edge("Rb",  "W", UncertainTransform(make_se3(rotz(0.01), [-0.02, 0.02, 0.00]), np.diag([6e-6]*6)), add_inverse=True)

    # Tracker branch
    net.add_edge("Trk", "Cam", UncertainTransform(make_se3(rotz(0.03), [0.10, 0.00, 0.00]), np.diag([3e-6]*6)), add_inverse=True)
    net.add_edge("Cam", "Mk",  UncertainTransform(make_se3(rotz(-0.02), [0.05, 0.02, 0.00]), np.diag([2e-6]*6)), add_inverse=True)

    # Robot branch
    net.add_edge("Rb", "Tool", UncertainTransform(make_se3(rotz(0.04), [0.00, 0.15, 0.02]), np.diag([4e-6]*6)), add_inverse=True)

    # CT branch
    net.add_edge("CT", "Anat", UncertainTransform(make_se3(rotz(0.025), [0.00, 0.00, 0.10]), np.diag([3e-6]*6)), add_inverse=True)

    # Points (names only; you can decide what to evaluate later)
    Cp = 2e-6 * np.eye(3)
    net.add_point("p_marker",   "Mk",   uvct3([0.02, 0.00,  0.00], Cp))
    net.add_point("p_tip",      "Tool", uvct3([0.00, 0.00, -0.12], Cp))
    net.add_point("p_landmark", "Anat", uvct3([0.03, -0.01, 0.02], Cp))

    return net
