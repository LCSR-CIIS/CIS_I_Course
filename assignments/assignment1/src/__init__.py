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

__all__ = ['Tracker', 'MarkerBody', 'Pointer', 'sample_normal']
