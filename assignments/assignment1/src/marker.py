# they can update the nominal values relative to the marker body - should be able to do that on the api side. 
from typing import List

from uncertainty_networks.nominal_types import vct3, Frame
from .utils import sample_normal
from tracker import Tracker


class Markers:

    def __init__(self, name):
        self.name = name
        self.nominal_marker_positions = [] # vct3 or uvct3 type; error-free marker positions
        self.observed_marker_positions = [] # vct3 type
        self.C = None

    def set_cov(self, C):
        self.C = C
        return self

    def create_marker_body(self, nominal_marker_positions: List[vct3]):
        self.nominal_marker_positions = nominal_marker_positions
        # here sampling can also depend on the params of uvct3?
        self.observed_marker_positions = sample_normal(nominal_marker_positions, cov=self.C)
        return self

    def add_marker_positions(self, nominal_marker_positions: List[vct3]):
        for pos in nominal_marker_positions:
            self.nominal_marker_positions.append(pos)
            self.observed_marker_positions.append(sample_normal([pos], cov=self.C)[0])
        return self
