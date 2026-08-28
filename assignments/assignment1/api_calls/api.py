"""HTTP API that lets students perform pivot-calibration data collection
against a simulated optical tracker, per the CIS I Assignment 1 spec:

  - defineFTracker / CreateTracker: specify the approximate pose of the
    tracking camera relative to the workspace.
  - DefineMarkerBody / UpdateMarkerBody / PlaceMarkerBody: specify the
    approximate marker-sphere layout (pointer, calibration object, or any
    other user-defined marker body) and its approximate placement.
  - AddMarkerBodies / RemoveMarkerBodies: mutate a tracker's tracked-body
    list.
  - SampleMarkerBodies / MostRecentObservedFrame / MostRecentObservedMarkers:
    take one noisy tracker reading of every tracked marker body, solving
    rigid registration (Procrustes) per body server-side, and retrieve the
    results.
  - Pointer + calibration-sensor routes: define a pointer's marker body and
    approximate tip position, hand-guide it near a calibration object at a
    desired orientation, and read the calibration sensor's high-accuracy
    tip position when in range (<=5mm).

Every route above is student-facing. Ground-truth error covariances that
turn these approximate/nominal inputs into realistic noisy readings are
hidden in _instructor_config.py (gitignored, never distributed to
students). A handful of additional /debug/* routes exist purely for
instructor/student debugging and are only mounted when the server is
started with CIS1_MODE=debug (see _instructor_config.InstructorSession).

Run locally (from the assignments/assignment1/ directory, so that `src` and
`api_calls` are both importable):
    python -m uvicorn api_calls.api:app --reload
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

# src's __init__.py puts the repo root on sys.path, which is what makes
# data_types/uncertainty_networks (below) importable -- must be imported
# first.
from src.marker import MarkerBody
from src.pointer import Pointer
from src.tracker import Tracker
from src.utils import sample_normal

from data_types.nominal_types import vct3, Rot, Frame
from data_types.uncertain_types import uFrame
from data_types.covariance_types import Covariance
from uncertainty_networks.se3 import exp_se3

from ._instructor_config import build_tracker, InstructorSession

app = FastAPI(title="CIS1 Tracker API")

_session = build_tracker()  # instructor-only hidden state; mode fixed for this server's lifetime


# ─────────────────────────────────────────────────────────────────────────
# (de)serialization models
# ─────────────────────────────────────────────────────────────────────────

class Vct3Model(BaseModel):
    x: float
    y: float
    z: float

    def to_vct3(self) -> vct3:
        return vct3(self.x, self.y, self.z)

    @classmethod
    def from_vct3(cls, v: vct3) -> "Vct3Model":
        return cls(x=v.x, y=v.y, z=v.z)


class FrameModel(BaseModel):
    R: List[List[float]]
    p: Vct3Model

    def to_frame(self) -> Frame:
        return Frame(Rot(matrix=np.array(self.R, dtype=np.float64)), self.p.to_vct3())

    @classmethod
    def from_frame(cls, f: Frame) -> "FrameModel":
        return cls(R=f.R.matrix.tolist(), p=Vct3Model.from_vct3(f.p))


class PoseRequest(BaseModel):
    """Covers defineFTracker's several overloads: a full Frame, R+p, R alone
    (translation defaults to the origin), or p alone (rotation defaults to
    identity)."""
    F: Optional[FrameModel] = None
    R: Optional[List[List[float]]] = None
    p: Optional[Vct3Model] = None

    def resolve(self) -> Frame:
        if self.F is not None:
            if self.R is not None or self.p is not None:
                raise HTTPException(400, "PoseRequest: cannot combine F with R/p")
            return self.F.to_frame()
        if self.R is not None and self.p is not None:
            return Frame(Rot(matrix=np.array(self.R, dtype=np.float64)), self.p.to_vct3())
        if self.R is not None:
            return Frame(Rot(matrix=np.array(self.R, dtype=np.float64)), vct3())
        if self.p is not None:
            return Frame(Rot(), self.p.to_vct3())
        raise HTTPException(400, "PoseRequest: must provide F, or R and/or p")


class CreateTrackerRequest(BaseModel):
    name: str
    F: PoseRequest
    marker_body_names: List[str] = []


class DefineMarkerBodyRequest(BaseModel):
    name: str
    nominal_positions: List[Vct3Model]
    is_calibration: bool = False


class UpdateMarkerBodyRequest(BaseModel):
    nominal_positions: List[Vct3Model]


class MarkerBodyNamesRequest(BaseModel):
    marker_body_names: List[str]


class DefinePointerRequest(BaseModel):
    name: str
    nominal_tip_position: Vct3Model
    marker_body_name: str


class HandGuideRequest(BaseModel):
    cal_name: str
    R_desired: List[List[float]]


class SenseTipRequest(BaseModel):
    pointer_name: str


class DebugConfigRequest(BaseModel):
    jiggle_cov: Optional[List[List[float]]] = None
    sensor_noise_cov: Optional[List[List[float]]] = None
    marker_manufacturing_cov: Optional[List[List[float]]] = None


# ─────────────────────────────────────────────────────────────────────────
# server-side state
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Registry:
    marker_bodies: Dict[str, MarkerBody] = field(default_factory=dict)
    trackers: Dict[str, Tracker] = field(default_factory=dict)
    tracked_body_names: Dict[str, List[str]] = field(default_factory=dict)
    pointers: Dict[str, Pointer] = field(default_factory=dict)
    calibration_body_names: set = field(default_factory=set)
    # hidden actual placement of each marker body (see _instructor_config's
    # scope note on PlaceMarkerBody / hand-guide) -- never serialized outside
    # the /debug/* routes.
    marker_body_placement: Dict[str, Frame] = field(default_factory=dict)
    # hidden actual pointer-relative-to-calibration-object pose from the most
    # recent hand-guide call, keyed by (pointer_name, cal_name).
    pointer_rel_cal: Dict[Tuple[str, str], Frame] = field(default_factory=dict)
    last_observed_frame: Dict[str, Frame] = field(default_factory=dict)
    last_observed_markers: Dict[str, List[vct3]] = field(default_factory=dict)


app.state.registry = Registry()
app.state.session = _session


def get_registry(request: Request) -> Registry:
    return request.app.state.registry


def get_session(request: Request) -> InstructorSession:
    return request.app.state.session


def _require(d: dict, key: str, kind: str):
    if key not in d:
        raise HTTPException(status_code=404, detail=f"{kind} '{key}' not found")
    return d[key]


# ─────────────────────────────────────────────────────────────────────────
# tracker pose
# ─────────────────────────────────────────────────────────────────────────

@app.post("/trackers")
def create_tracker(req: CreateTrackerRequest, registry: Registry = Depends(get_registry)):
    """CreateTracker(F, {mB1, ..., mBN})"""
    if req.name in registry.trackers:
        raise HTTPException(409, f"tracker '{req.name}' already exists")
    for mb_name in req.marker_body_names:
        _require(registry.marker_bodies, mb_name, "marker_body")

    tracker = Tracker(req.name, req.F.resolve())
    registry.trackers[req.name] = tracker
    registry.tracked_body_names[req.name] = list(req.marker_body_names)
    return {"name": req.name, "tracked_marker_bodies": registry.tracked_body_names[req.name]}


@app.post("/trackers/{name}/pose")
def define_tracker_pose(name: str, req: PoseRequest, registry: Registry = Depends(get_registry)):
    """defineFTracker(...)"""
    tracker = _require(registry.trackers, name, "tracker")
    tracker.F = req.resolve()
    return {"name": name}


@app.post("/trackers/{name}/marker-bodies:add")
def add_marker_bodies(name: str, req: MarkerBodyNamesRequest, registry: Registry = Depends(get_registry)):
    """AddMarkerBodies(trk, {mB1, ..., mBN}) -> PreviousList"""
    _require(registry.trackers, name, "tracker")
    for mb_name in req.marker_body_names:
        _require(registry.marker_bodies, mb_name, "marker_body")
    previous = list(registry.tracked_body_names[name])
    for mb_name in req.marker_body_names:
        if mb_name not in registry.tracked_body_names[name]:
            registry.tracked_body_names[name].append(mb_name)
    return {"previous_list": previous}


@app.post("/trackers/{name}/marker-bodies:remove")
def remove_marker_bodies(name: str, req: MarkerBodyNamesRequest, registry: Registry = Depends(get_registry)):
    """RemoveMarkerBodies(trk, {mB1, ..., mBN}) -> PreviousList"""
    _require(registry.trackers, name, "tracker")
    previous = list(registry.tracked_body_names[name])
    registry.tracked_body_names[name] = [n for n in previous if n not in req.marker_body_names]
    return {"previous_list": previous}


@app.post("/trackers/{name}/sample")
def sample_marker_bodies(name: str, registry: Registry = Depends(get_registry),
                          session: InstructorSession = Depends(get_session)):
    """SampleMarkerBodies(trk) -> {F1, ..., FN}"""
    tracker = _require(registry.trackers, name, "tracker")

    frames = {}
    for mb_name in registry.tracked_body_names.get(name, []):
        mb = registry.marker_bodies[mb_name]
        body_placement = registry.marker_body_placement.get(mb_name)
        observed = tracker.read_marker_body(
            mb, jiggle_cov=session.jiggle_cov, sensor_noise_cov=session.sensor_noise_cov,
            body_placement=body_placement,
        )
        F_est = tracker.procrustes_solver(mb.nominal_marker_positions, observed)
        registry.last_observed_frame[mb_name] = F_est
        registry.last_observed_markers[mb_name] = [o.p for o in observed]
        frames[mb_name] = FrameModel.from_frame(F_est)

    time.sleep(session.delta_tsamp)
    return {"frames": frames, "elapsed_sec": session.delta_tsamp}


# ─────────────────────────────────────────────────────────────────────────
# marker bodies
# ─────────────────────────────────────────────────────────────────────────

@app.post("/marker-bodies")
def define_marker_body(req: DefineMarkerBodyRequest, registry: Registry = Depends(get_registry),
                        session: InstructorSession = Depends(get_session)):
    """DefineMarkerBody({a_m,1, ..., a_m,N}) -> mB"""
    if req.name in registry.marker_bodies:
        raise HTTPException(409, f"marker body '{req.name}' already exists")

    positions = [v.to_vct3() for v in req.nominal_positions]
    mb = MarkerBody(req.name, positions, Frame.eye())
    session.apply_marker_manufacturing_noise(mb)
    registry.marker_bodies[req.name] = mb
    if req.is_calibration:
        registry.calibration_body_names.add(req.name)
    return {"name": req.name, "n_markers": len(positions)}


@app.put("/marker-bodies/{name}")
def update_marker_body(name: str, req: UpdateMarkerBodyRequest, registry: Registry = Depends(get_registry),
                        session: InstructorSession = Depends(get_session)):
    """UpdateMarkerBody(mB, {a_m,1, ..., a_m,N}) -> PreviousValues"""
    mb = _require(registry.marker_bodies, name, "marker_body")
    previous = mb.nominal_marker_positions
    new_positions = [v.to_vct3() for v in req.nominal_positions]
    mb.set_marker_positions(new_positions, session.marker_manufacturing_cov)
    return {"previous_positions": [Vct3Model.from_vct3(p) for p in previous]}


@app.post("/marker-bodies/{name}/place")
def place_marker_body(name: str, req: FrameModel, registry: Registry = Depends(get_registry),
                       session: InstructorSession = Depends(get_session)):
    """PlaceMarkerBody(mB, Fmb)"""
    mb = _require(registry.marker_bodies, name, "marker_body")
    F_nominal = req.to_frame()
    mb.update_world_frame(F_nominal)
    is_cal = name in registry.calibration_body_names
    registry.marker_body_placement[name] = session.sample_placement(F_nominal, is_calibration=is_cal)
    return {"placed": True}


@app.get("/marker-bodies/{name}/frame")
def most_recent_observed_frame(name: str, registry: Registry = Depends(get_registry)):
    """Fm = MostRecentObservedFrame(mB)"""
    _require(registry.marker_bodies, name, "marker_body")
    if name not in registry.last_observed_frame:
        raise HTTPException(404, f"marker body '{name}' has not been sampled yet")
    return FrameModel.from_frame(registry.last_observed_frame[name])


@app.get("/marker-bodies/{name}/markers")
def most_recent_observed_markers(name: str, registry: Registry = Depends(get_registry)):
    """{a_m,1, ..., a_m,N} = MostRecentObservedMarkers(mB)"""
    _require(registry.marker_bodies, name, "marker_body")
    if name not in registry.last_observed_markers:
        raise HTTPException(404, f"marker body '{name}' has not been sampled yet")
    return {"markers": [Vct3Model.from_vct3(p) for p in registry.last_observed_markers[name]]}


# ─────────────────────────────────────────────────────────────────────────
# pointer + calibration sensor
# ─────────────────────────────────────────────────────────────────────────

@app.post("/pointers")
def define_pointer(req: DefinePointerRequest, registry: Registry = Depends(get_registry),
                    session: InstructorSession = Depends(get_session)):
    if req.name in registry.pointers:
        raise HTTPException(409, f"pointer '{req.name}' already exists")
    mb = _require(registry.marker_bodies, req.marker_body_name, "marker_body")

    ptr = Pointer(req.name, req.nominal_tip_position.to_vct3(), mb)
    session.apply_pointer_manufacturing_noise(ptr)
    registry.pointers[req.name] = ptr
    return {"name": req.name}


@app.post("/pointers/{name}/hand-guide")
def hand_guide_pointer(name: str, req: HandGuideRequest, registry: Registry = Depends(get_registry),
                        session: InstructorSession = Depends(get_session)):
    """
    Simulates hand-guiding the robot so the pointer's tip is within
    approximately 5mm of the calibration object's origin, with the
    pointer's orientation approximately R_desired relative to the
    calibration object (PDF page 5, step 4 -- "a special Python function is
    provided to enable you to simulate this function").
    """
    ptr = _require(registry.pointers, name, "pointer")
    if req.cal_name not in registry.calibration_body_names:
        raise HTTPException(400, f"'{req.cal_name}' is not a defined calibration object")
    F_cal_actual = registry.marker_body_placement.get(req.cal_name)
    if F_cal_actual is None:
        raise HTTPException(400, f"calibration object '{req.cal_name}' has not been placed yet")

    R_desired_mat = np.array(req.R_desired, dtype=np.float64)
    p_tip_actual = ptr.get_actual_tip_position().p
    # Choose the translation so the tip lands exactly at the calibration
    # object's origin under the desired orientation, before hand-guide noise:
    # R_desired * p_tip + t = 0.
    t_vec = -(R_desired_mat @ p_tip_actual.vec)
    F_ptr_rel_cal_nominal = Frame(Rot(matrix=R_desired_mat), vct3(t_vec[0, 0], t_vec[1, 0], t_vec[2, 0]))

    eta = sample_normal(np.zeros(6), cov=session.hand_guide_cov)[0]
    F_ptr_rel_cal_actual = Frame(exp_se3(eta)) * F_ptr_rel_cal_nominal

    F_ptr_actual_world = F_cal_actual * F_ptr_rel_cal_actual
    registry.marker_body_placement[ptr.marker_body.name] = F_ptr_actual_world
    registry.pointer_rel_cal[(name, req.cal_name)] = F_ptr_rel_cal_actual
    return {"hand_guided": True}


@app.post("/calibration-object/{name}/sense-tip")
def sense_tip(name: str, req: SenseTipRequest, registry: Registry = Depends(get_registry),
              session: InstructorSession = Depends(get_session)):
    """
    If the pointer's tip is within 5mm of the calibration object's origin
    (most recently established via hand-guide), reports the tip's position
    relative to the calibration object to near-zero error. Otherwise, a
    normal (HTTP 200) out-of-range result -- not an error.
    """
    if name not in registry.calibration_body_names:
        raise HTTPException(400, f"'{name}' is not a defined calibration object")
    ptr = _require(registry.pointers, req.pointer_name, "pointer")

    F_ptr_rel_cal_actual = registry.pointer_rel_cal.get((req.pointer_name, name))
    if F_ptr_rel_cal_actual is None:
        return {"in_range": False, "sensed_tip": None}

    p_tip_actual = ptr.get_actual_tip_position().p
    p_in_cal_frame = F_ptr_rel_cal_actual * p_tip_actual
    p_sensed = sample_normal(p_in_cal_frame, cov=session.cal_sensor_cov)
    in_range = p_sensed.norm() <= 0.005
    return {"in_range": in_range, "sensed_tip": Vct3Model.from_vct3(p_sensed) if in_range else None}


# ─────────────────────────────────────────────────────────────────────────
# debug-only routes (only mounted when the server is started with
# CIS1_MODE=debug -- see _instructor_config.InstructorSession)
# ─────────────────────────────────────────────────────────────────────────

if _session.is_debug:

    @app.get("/debug/ground-truth")
    def debug_ground_truth(registry: Registry = Depends(get_registry),
                            session: InstructorSession = Depends(get_session)):
        marker_bodies = {}
        for mb_name, mb in registry.marker_bodies.items():
            entry = {}
            if mb.actual_marker_positions is not None:
                entry["actual_local_positions"] = [Vct3Model.from_vct3(p.p) for p in mb.actual_marker_positions]
            if mb_name in registry.marker_body_placement:
                entry["actual_placement"] = FrameModel.from_frame(registry.marker_body_placement[mb_name])
            marker_bodies[mb_name] = entry

        pointers = {}
        for ptr_name, ptr in registry.pointers.items():
            if ptr.actual_tip_position is not None:
                pointers[ptr_name] = {"actual_tip_position": Vct3Model.from_vct3(ptr.actual_tip_position.p)}

        trackers = {}
        for trk_name, trk in registry.trackers.items():
            F_nom = trk.F.F if isinstance(trk.F, uFrame) else trk.F
            trackers[trk_name] = {
                "nominal_F": FrameModel.from_frame(F_nom),
                "jiggle_cov_diag": session.jiggle_cov.diag().tolist(),
            }

        return {"marker_bodies": marker_bodies, "pointers": pointers, "trackers": trackers}

    @app.post("/debug/config")
    def debug_override_config(req: DebugConfigRequest, session: InstructorSession = Depends(get_session)):
        if req.jiggle_cov is not None:
            session.jiggle_cov = Covariance(req.jiggle_cov)
        if req.sensor_noise_cov is not None:
            session.sensor_noise_cov = Covariance(req.sensor_noise_cov)
        if req.marker_manufacturing_cov is not None:
            session.marker_manufacturing_cov = Covariance(req.marker_manufacturing_cov)
        return {"updated": True}


@app.get("/health")
def health():
    return {"status": "ok", "mode": _session.mode}
