from typing import Union

from .utils import sample_normal
from data_types.nominal_types import vct3
from data_types.uncertain_types import uvct3
from .marker import MarkerBody


TipPosition = Union[vct3, uvct3]


class Pointer:
    def __init__(self, name, nominal_tip_position: Union[vct3, uvct3] = None, marker_body: MarkerBody = None):
        self.name = name
        self.nominal_tip_position = nominal_tip_position
        self.marker_body = marker_body
        self._actual_tip_position = None
        self.__cov = None  # ground-truth sampling covariance; never exposed publicly

    def set_tip_position_nominal(self, position: Union[vct3, uvct3]):
        self.nominal_tip_position = position
        return self

    def _set_ground_truth_cov(self, cov):
        """Instructor-only: sets the covariance used to sample the actual
        tip position. Not part of the student-facing API."""
        self.__cov = cov
        self._actual_tip_position = None  # invalidate any previously cached sample
        return self

    def get_actual_tip_position(self):
        if self._actual_tip_position is None:
            if self.__cov is None:
                raise RuntimeError("ground-truth covariance has not been set")
            self._actual_tip_position = sample_normal([self.nominal_tip_position], cov=self.__cov)[0]
        return self._actual_tip_position


