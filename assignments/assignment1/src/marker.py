from typing import List, Union

from data_types.nominal_types import vct3, Frame
from data_types.uncertain_types import uvct3, uFrame
from data_types.covariance_types import Covariance
from .utils import sample_normal

# Covariance in markers would be changed to a directional vector with a radius of N mm.

class MarkerBody:
    def __init__(self, name, nominal_marker_positions: List[vct3], world_frame: Union[Frame, uFrame]):
        self.name = name
        self.nominal_marker_positions = nominal_marker_positions
        self.observed_marker_positions = None  # populated by Tracker via tracker-error covariance
        # frames that need to be initialized
        self.world_frame = world_frame  # position of marker body wrt world frame
        self.actual_marker_positions = None
        self.manufacturing_cov = None  # ground-truth covariance; never exposed publicly

    def set_manufacturing_cov(self, cov):  # this should be removed
        """Instructor-only: covariance used to sample actual marker
        positions from the student-submitted nominal ones. Not part of the
        student-facing API."""
        self.manufacturing_cov = cov
        self.actual_marker_positions = None  # invalidate any cached sample
        return self

    # cov here should be a global variable extracted from a preamble
    def set_marker_positions(self, nominal_marker_positions: List[vct3], cov: Covariance): # cov here should be a 3x3, not 6x6. covariance is required and should raise RE if not present
        self.nominal_marker_positions = nominal_marker_positions
        # set the actual marker positions once we get nominal values using sampling with cov
        self.actual_marker_positions = None  # invalidate any cached sample 
        return self.actual_marker_positions

    def get_actual_marker_positions(self) -> List[uvct3]:
        if self.actual_marker_positions is None:
            if self.manufacturing_cov is None:
                raise RuntimeError("manufacturing covariance has not been set")
            sampled = sample_normal(self.nominal_marker_positions, cov=self.manufacturing_cov)
            self.actual_marker_positions = [uvct3(p) for p in sampled]
        return self.actual_marker_positions

    def get_marker_positions_world_frame(self) -> List[uvct3]:
        if self.world_frame is None:
            raise RuntimeError("marker world frame has not been set")
        return [self.world_frame * p for p in self.nominal_marker_positions], [self.world_frame * p for p in self.get_actual_marker_positions()]

    def update_world_frame(self, frame: Union[Frame, uFrame]):
        self.world_frame = frame
