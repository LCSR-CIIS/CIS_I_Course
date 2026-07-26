from typing import List, Union

from .utils import sample_normal
from uncertainty_networks.nominal_types import vct3
from uncertainty_networks.uncertain_types import uvct3

TipPosition = Union[vct3, uvct3]


class Pointer:
    def __init__(self, name, tip_position_nominal: TipPosition = None):
        self.name = name
        self.tip_position_nominal = tip_position_nominal
        self.tip_position_actual = None

    def set_tip_position_nominal(self, position: TipPosition):
        self.tip_position_nominal = position
        return self

    def sample_tip_position_actual(self, cov):
        self.tip_position_actual = sample_normal([self.tip_position_nominal], cov=cov)[0]
        return self

    def create_pointer_body(self, tip_position_nominal: TipPosition, cov):
        self.set_tip_position_nominal(tip_position_nominal)
        self.tip_position_actual = sample_normal([self.tip_position_nominal], cov=cov)[0]
        return self

# TBD