"""
Author: X.M. Christine Zhu
Date: 04/04/2026

uncertainty-networks
====================
Uncertainty propagation through geometric networks on SE(3),
following the CIS I right-multiplicative perturbation convention.

Quick start
-----------
    from uncertainty_networks import GeometricNetwork, UncertainTransform
    import numpy as np

    net = GeometricNetwork()
    F   = np.eye(4); F[:3, 3] = [0.3, 0, 0]
    net.add_edge("World", "Tool", UncertainTransform(F, 0.002**2 * np.eye(6)))
    result = net.query_frame("World", "Tool")

See API_REFERENCE.md for the complete public API.
Mathematical reference: docs/math_note.pdf
"""

from .uncertain_geometry import UncertainTransform, Convention
from .nominal_types import vct3, Rot, Frame
from .uncertain_types import uScalar, uVector, uvct3, uRot, uFrame
from .network import (
    GeometricNetwork,
    PathResult,
    FusedQueryResult,
)
from .observations import (
    Observation,
    LoopObservation,
    PointObservation,
    DistanceObservation,
    condition_on_observations,
    ConditioningResult,
)
from .closed_loop import (
    LoopPosterior,
    condition_on_loop,
    fuse_gaussian_covs,
)

__all__ = [
    # Core geometry
    "UncertainTransform",
    "Convention",
    # Spatial math primitives
    "vct3", "Rot", "Frame",
    # Uncertain types (Dr. Taylor CIS I API)
    "uScalar", "uVector", "uvct3", "uRot", "uFrame",
    # Network
    "GeometricNetwork",
    "PathResult",
    "FusedQueryResult",
    # Observations / conditioning
    "Observation",
    "LoopObservation",
    "PointObservation",
    "DistanceObservation",
    "condition_on_observations",
    "ConditioningResult",
    # Loop utilities
    "LoopPosterior",
    "condition_on_loop",
    "fuse_gaussian_covs",
]
