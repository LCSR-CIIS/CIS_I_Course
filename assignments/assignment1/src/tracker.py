from typing import Union

from .utils import sample_normal
from data_types.nominal_types import vct3, Frame
from data_types.uncertain_types import uvct3, uFrame
from data_types.covariance_types import Covariance
from .marker import MarkerBody


class Tracker:

    def __init__(self, name, F: Union[Frame, uFrame]):
        self.name = name
        #if students can call self.F, wouldn't they be able to know the covariance of F? Work around that
        self.F = F  # Frame position of the tracking camera in the world coordinate system; set by students
        self.marker_positions = []  # observed marker positions in the tracker's coordinate frame

    def __repr__(self):
        return f"Tracker(name={self.name!r}, F={self.F!r})"

    def set_pose(self, frame: Union[Frame, uFrame]):
        self.F = uFrame(frame)
        return self

    # this method should be private, we are the ones changing the covariance of the tracker. we can let the students change the covariance while debugging,
    def _set_cov(self, cov: Covariance):  # here we might want to pass just np.ndarray, but then we avoid any checks
        self.F = uFrame(self.F, cov)
        return self

    def read_marker_body(self, marker_body: MarkerBody):
        # read the marker positions and update the tracker's state accordingly
        '''
        input: marker_body - a Marker Body object containing nominal, observed marker positions
        output: samples - list of sampled marker positions
        '''
        n = len(marker_body.nominal_marker_positions)
        self.marker_positions = marker_body.get_actual_marker_positions()
        for i in range(n):
            marker_pos_tracker_observed = self.F.inv() * self.marker_positions[i] # read actual marker positions, they have their own covariance. uFrame supports multiplying uncertain transform with uncertain point
            self.marker_positions.append(marker_pos_tracker_observed)
        return self.marker_positions

    def procrustes_solver(self):
        # or a quaternion method from the lecture notes
        pass
