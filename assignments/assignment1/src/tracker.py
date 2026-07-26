from typing import List

from .utils import sample_normal
from uncertainty_networks.nominal_types import vct3, Frame
from marker import Markers


class Tracker:

    def __init__(self, name):
        self.name = name
        self.F = None  # Frame position of the tracking camera in the world coordinate system
        self.C = None  # Covariance matrix representing the uncertainty of the tracker's pose
        self.marker_positions = []

    def __repr__(self):
        return f"Tracker(name={self.name!r}, F={self.F!r}, C={self.C!r})"

    def create(self, frame: Frame, cov):
        self.F = frame
        self.C = cov
        return self

    def set_pose(self, frame: Frame):
        # when setting a new pose for the tracker, do we need to update the covariance as well?
        self.F = frame
        return self

    def set_cov(self, cov):
        # update the covariance matrix for the tracker
        self.C = cov
        return self

    def read_markers(self, marker: "Markers"): # CHANGE: here it should read a Marker object instead of raw positions
        # read the marker positions and update the tracker's state accordingly
        '''
        input: marker - a Marker object containing nominal and observed marker positions
        output: samples - list of sampled marker positions
        '''
        n = len(marker.observed_marker_positions)
        samples = []
        for i in range(n):
            marker_pos_tracker = self.F.inv() * marker.observed_marker_positions[i]
            marker_pos_tracker_sampled = sample_normal(
                marker_pos_tracker.vec.squeeze(1), cov=self.C)
            samples.append(marker_pos_tracker_sampled)
        return samples

    def add_marker_position(self, marker_pos: vct3):
        # add a new marker position in the world coordinate system to the list of seen marker positions
        self.marker_positions.append(marker_pos)
        return self
