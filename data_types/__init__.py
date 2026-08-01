from pathlib import Path
import sys

# nominal_types.py / uncertain_types.py import se3 and uncertain_geometry
# directly from the uncertainty_networks package (../src) rather than keeping
# duplicate copies here, so that package needs to be importable first.
_UNCERTAINTY_NETWORKS_SRC = Path(__file__).resolve().parent.parent / 'src'
if str(_UNCERTAINTY_NETWORKS_SRC) not in sys.path:
    sys.path.insert(0, str(_UNCERTAINTY_NETWORKS_SRC))
