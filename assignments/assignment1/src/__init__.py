"""
Public API for the CIS1 assignment 1 package.
"""

from pathlib import Path
import sys

_REPO_SRC = Path(__file__).resolve().parents[3] / 'src'
if str(_REPO_SRC) not in sys.path:
	sys.path.insert(0, str(_REPO_SRC))

from .tracker import Tracker
from .utils import sample_normal

__all__ = ['Tracker', 'sample_normal']
