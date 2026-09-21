"""
Ground-truth noise parameters for the assignment 1 simulation.

All lengths are in millimeters, all angles in radians

Every constant below is a per-axis standard deviation for an isotropic 3-D
Gaussian. Where the assignment gives a typical std directly, that value is used
as-is. Where the assignment instead gives a bound of the form ||delta|| <= B (a
radius on the *norm* of a 3-vector, not a per-axis std), the bound is
converted to a per-axis std by treating B as roughly the 95th-percentile
radius of a 3-dof chi distribution (radius = std * chi3_95 = std * 2.795),
i.e. std = B / 2.795. This is the standard trick for turning a "the error
is essentially always within B" spec into a Gaussian std.
"""

# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

TRACKER_JIGGLE_ANGLE_STD = 0.01    # sigma_Ta: tracker pose angular jiggle, rad
TRACKER_JIGGLE_POS_STD = 5.0       # sigma_Tp: tracker pose translational jiggle, mm
SENSOR_NOISE_STD = 0.02            # sigma_b: per-marker camera detection noise, mm
SAMPLE_DELTA_T_SEC = 0.03          # delta_t_samp: nominal time cost of one SampleMarkerBodies call

_CHI3_95 = 2.795  # 95th percentile radius of a 3-dof chi distribution, in std-devs

# ---------------------------------------------------------------------------
# Marker body manufacturing ( ||delta_a_m,k|| <= 2mm, for both a_ptr,k and a_cal,k )
# ---------------------------------------------------------------------------

MARKER_MANUFACTURING_STD = 2.0 / _CHI3_95  # ~0.72mm per-axis std on each marker sphere's true position

# ---------------------------------------------------------------------------
# Marker body placement ( ||alpha_cal|| <= 0.1 rad, ||epsilon_cal|| <= 25mm ).
# The assignment only gives numeric bounds for the calibration object specifically;
# any other placed marker body (e.g. extra bodies a student adds near the
# test volume) reuses the same bounds, since the assignment's general
# PlaceMarkerBody section leaves delta_mBa/delta_mBe as unspecified symbols.
# ---------------------------------------------------------------------------

BODY_PLACEMENT_ANGLE_STD = 0.1 / _CHI3_95    # ~0.036 rad
BODY_PLACEMENT_POS_STD = 25.0 / _CHI3_95     # ~8.94mm

# ---------------------------------------------------------------------------
# Hand-guiding the pointer near the calibration object.
# ---------------------------------------------------------------------------

HAND_GUIDE_POS_STD = 5.0 / _CHI3_95
HAND_GUIDE_ANGLE_STD = 0.05          

# ---------------------------------------------------------------------------
# Step 8's test poses: "The system will position the pointer at
# F*t = Ft * delta_Ft, where delta_Ft is relatively small." This is the same
# robot precisely moving to a *commanded* pose as hand-guiding is, not a
# roughly-placed static object -- so it reuses hand-guide-scale precision,
# tightened slightly since "relatively small" reads as more precise than
# hand-guiding's "approximately 5mm". Distinct from BODY_PLACEMENT_*_STD
# above, which models a coarsely, manually placed static body (the
# calibration object).
# ---------------------------------------------------------------------------

TEST_POSE_POS_STD = 2.0 / _CHI3_95    # ~0.72mm
TEST_POSE_ANGLE_STD = 0.02            # rad 

# ---------------------------------------------------------------------------
# Calibration sensor.
# ---------------------------------------------------------------------------

CAL_SENSOR_RANGE_MM = 5.0          
CAL_SENSOR_NOISE_STD = SENSOR_NOISE_STD / 10.0  # ~0.002mm -- "negligible error"
