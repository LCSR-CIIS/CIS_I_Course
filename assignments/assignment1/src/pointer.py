from typing import List, Union

from .utils import sample_normal
from uncertainty_networks.nominal_types import vct3
from uncertainty_networks.uncertain_types import uvct3

TipPosition = Union[vct3, uvct3]


class Pointer:
    def __init__(self, name, nominal_tip_position: TipPosition = None):
        self.name = name
        self.nominal_tip_position = nominal_tip_position
        self.actual_tip_position = None

    def set_tip_position_nominal(self, position: TipPosition):
        self.nominal_tip_position = position
        return self

    def sample_tip_position_actual(self, cov):
        self.actual_tip_position = sample_normal([self.nominal_tip_position], cov=cov)[0]
        return self

    def create_pointer_body(self, nominal_tip_position: TipPosition, cov):
        self.set_tip_position_nominal(nominal_tip_position)
        self.actual_tip_position = sample_normal([self.nominal_tip_position], cov=cov)[0]
        return self

# TBD