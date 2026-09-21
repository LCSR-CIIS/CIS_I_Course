"""
Public API for the CIS1 assignment 1 package.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from .tracker import Tracker
from .marker import MarkerBody
from .pointer import Pointer

from .utils import sample_normal
from . import simulation

from data_types.nominal_types import vct3, Rot, Frame
from data_types.uncertain_types import uvct3, uRot, uFrame
from data_types.covariance_types import Covariance

__all__ = [
	'Tracker', 'MarkerBody', 'Pointer', 'sample_normal', 'simulation',
	'vct3', 'Rot', 'Frame', 'uvct3', 'uRot', 'uFrame', 'Covariance',
]
