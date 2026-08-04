# they can update the nominal values relative to the marker body - should be able to do that on the api side. 
from typing import List, Union

from data_types.nominal_types import vct3, Frame
from data_types.uncertain_types import uvct3, uFrame
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

    def set_manufacturing_cov(self, cov): 
        """Instructor-only: covariance used to sample actual marker
        positions from the student-submitted nominal ones. Not part of the
        student-facing API."""
        self.manufacturing_cov = cov
        self.actual_marker_positions = None  # invalidate any cached sample
        return self

    def set_marker_positions(self, nominal_marker_positions: List[vct3]):
        self.nominal_marker_positions = nominal_marker_positions
        self.actual_marker_positions = None  # invalidate any cached sample
        return self

    def add_marker_positions(self, nominal_marker_positions: List[vct3]):
        self.nominal_marker_positions.extend(nominal_marker_positions)
        self.actual_marker_positions = None  # invalidate any cached sample
        return self

    def get_actual_marker_positions(self) -> List[uvct3]:
        if self.actual_marker_positions is None:
            if self.manufacturing_cov is None:
                raise RuntimeError("manufacturing covariance has not been set")
            self._actual_marker_positions = sample_normal(
                self.nominal_marker_positions, cov=self.manufacturing_cov
            )
        return self.actual_marker_positions

    def update_world_frame(self, frame: Union[Frame, uFrame]):
        self.world_frame = frame
