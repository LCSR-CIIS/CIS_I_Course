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
        self.marker_body = marker_body  # marker body on the pointer; also owns the pointer's world frame
        self.actual_tip_position = None
        self.manufacturing_cov = None  # ground-truth covariance; never exposed publicly

    def set_manufacturing_cov(self, cov):  # instructor-only, mirrors MarkerBody.set_manufacturing_cov
        self.manufacturing_cov = cov
        self.actual_tip_position = None  # invalidate any cached sample
        return self

    def get_actual_tip_position(self) -> uvct3:
        if self.actual_tip_position is None:
            if self.manufacturing_cov is None:
                raise RuntimeError("manufacturing covariance has not been set")
            self.actual_tip_position = uvct3(sample_normal(self.nominal_tip_position, cov=self.manufacturing_cov))
        return self.actual_tip_position

    # The pointer's tip and the marker body's markers are both fixed offsets
    # on the same rigid object -- there's one world frame here, not two, so
    # these read self.marker_body.world_frame directly rather than caching a
    # separate copy that could drift out of sync with it.

    def get_tip_position_world_frame(self) -> vct3:
        if self.marker_body is None or self.marker_body.world_frame is None:
            raise RuntimeError("pointer's marker body has not been placed in the workspace")
        return self.marker_body.world_frame * self.nominal_tip_position

    def get_actual_tip_position_world_frame(self) -> vct3:
        if self.marker_body is None or self.marker_body.world_frame is None:
            raise RuntimeError("pointer's marker body has not been placed in the workspace")
        return self.marker_body.world_frame * self.get_actual_tip_position().p


