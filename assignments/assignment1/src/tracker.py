from typing import List

from .utils import sample_normal
from uncertainty_networks.nominal_types import vct3, Frame


class Tracker:

    def __init__(self, name):
        self.name = name
        self.F = None  # Frame position of the tracking camera in the world coordinate system
        self.C = None  # Covariance matrix representing the uncertainty of the tracker's pose
        # keep track of the all marker positions that the tracker sees
        self.marker_positions = []
        # marker should be able to add new points to the list of seen marker positions
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

    def read_markers(self, marker_pos: List[vct3]):
        # read the marker positions and update the tracker's state accordingly
        '''
        input: marker_pos - list of marker positions in the world coordinate system
        output: samples - list of sampled marker positions
        '''
        n = len(marker_pos)
        samples = []
        for i in range(n):
            marker_pos_tracker = self.F.inv() * marker_pos[i]
            marker_pos_tracker_sampled = sample_normal(
                marker_pos_tracker.vec.squeeze(1), self.C)
            samples.append(marker_pos_tracker_sampled)
        return samples

    def add_marker_position(self, marker_pos: vct3):
        # add a new marker position in the world coordinate system to the list of seen marker positions
        self.marker_positions.append(marker_pos)
        return self
