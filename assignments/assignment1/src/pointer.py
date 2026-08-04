from typing import Union

from .utils import sample_normal
from data_types.nominal_types import vct3
from data_types.uncertain_types import uvct3
from .marker import MarkerBody


TipPosition = Union[vct3, uvct3]

# change covariances to global parameters and export them!
class Pointer:
    def __init__(self, name, nominal_tip_position: Union[vct3, uvct3] = None, marker_body: MarkerBody = None):
        self.name = name
        self.nominal_tip_position = nominal_tip_position
        self.marker_body = marker_body
        self.actual_tip_position = self.get_actual_tip_position(nominal_tip_position, cov)

    def get_actual_tip_position(self, nominal_tip_position, cov):
        if self._actual_tip_position is None:
            if self.cov is None:
                raise RuntimeError("ground-truth covariance has not been set")
            self._actual_tip_position = sample_normal([self.nominal_tip_position], cov=self.__cov)[0]
        return self._actual_tip_position


