"""
Non-interactive simulation for assignment 1. Students call these functions directly; ground-truth noise is
injected here rather than being something they need to (or can) set
themselves -- see constants.py for every noise parameter and where its
value comes from.
"""

from .marker_body import PlacedBody, define_marker_body, place_marker_body
from .pointer import define_pointer
from .tracker import ObservedBody, create_tracker, sample_marker_bodies
from .calibration_sensor import HandGuideResult, hand_guide_pointer, sense_tip
from . import constants

__all__ = [
    "PlacedBody",
    "define_marker_body",
    "place_marker_body",
    "define_pointer",
    "ObservedBody",
    "create_tracker",
    "sample_marker_bodies",
    "HandGuideResult",
    "hand_guide_pointer",
    "sense_tip",
    "constants",
]
