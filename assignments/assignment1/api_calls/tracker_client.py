"""
Pythonic client for the CIS I Assignment 1 Tracker API (api_calls/api.py).

Start the server first (see api_calls/README.md):

    cd assignments/assignment1
    python -m uvicorn api_calls.api:app --reload

then drive it from a plain Python script instead of hand-building JSON and
calling `requests.post(...)` yourself:

    from api_calls.tracker_client import TrackerClient
    from data_types.nominal_types import vct3, Rot, Frame

    client = TrackerClient()
    client.define_marker_body("ptr_mb", nominal_positions=[
        vct3(0, 0, 0), vct3(50, 0, 0), vct3(0, 50, 0), vct3(0, 0, 50), vct3(30, 20, 10),
    ])

Every method below is a thin wrapper around one HTTP route: it builds that
route's exact request body and parses its response back into vct3/Rot/Frame
objects. The hidden noise the assignment relies on stays entirely
server-side (api_calls/_instructor_config.py) -- this is a client, not a
bypass.
"""

from typing import Dict, List, Optional

import numpy as np
import requests

# src/__init__.py inserts the repo root onto sys.path, which is what makes
# `data_types` importable below -- must be imported first (same convention
# api_calls/api.py itself uses).
import src  # noqa: F401
from data_types.nominal_types import vct3, Rot, Frame


class TrackerAPIError(Exception):
    """Raised when the Tracker API responds with a non-2xx status.

    Carries the server's own error message (`detail`) instead of forcing
    every caller to inspect a `requests.Response` themselves.
    """

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Tracker API error {status_code}: {detail}")


def _vct3_to_json(v: vct3) -> dict:
    return {"x": v.x, "y": v.y, "z": v.z}


def _json_to_vct3(d: dict) -> vct3:
    return vct3(d["x"], d["y"], d["z"])


def _frame_to_json(F: Frame) -> dict:
    return {"R": F.R.matrix.tolist(), "p": _vct3_to_json(F.p)}


def _json_to_frame(d: dict) -> Frame:
    return Frame(Rot(matrix=d["R"]), _json_to_vct3(d["p"]))


def _cov_to_json(cov) -> Optional[list]:
    if cov is None:
        return None
    return np.asarray(cov, dtype=np.float64).tolist()


def _pose_json(F: Optional[Frame] = None, R: Optional[Rot] = None, p: Optional[vct3] = None) -> dict:
    """Builds a PoseRequest-shaped payload ({"F": ...} or {"R": ..., "p": ...}).

    Mirrors PoseRequest.resolve()'s overload rules in api_calls/api.py: pass
    a full Frame, or R and/or p (R alone defaults p to the origin, p alone
    defaults R to identity, both server-side).
    """
    if F is not None:
        if R is not None or p is not None:
            raise ValueError("_pose_json: cannot combine F with R/p")
        return {"F": _frame_to_json(F)}
    if R is None and p is None:
        raise ValueError("_pose_json: must provide F, or R and/or p")
    payload = {}
    if R is not None:
        payload["R"] = R.matrix.tolist()
    if p is not None:
        payload["p"] = _vct3_to_json(p)
    return payload


class TrackerClient:
    """Thin HTTP client for the Tracker API. One method per route, named
    after the assignment's pseudocode (DefineMarkerBody -> define_marker_body,
    etc.), taking and returning vct3/Rot/Frame instead of raw JSON."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _request(self, method: str, path: str, json: Optional[dict] = None) -> dict:
        response = self._session.request(method, f"{self.base_url}{path}", json=json)
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise TrackerAPIError(response.status_code, detail)
        return response.json()

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, json: Optional[dict] = None) -> dict:
        return self._request("POST", path, json=json if json is not None else {})

    def _put(self, path: str, json: Optional[dict] = None) -> dict:
        return self._request("PUT", path, json=json if json is not None else {})

    # -- marker bodies --------------------------------------------------

    def define_marker_body(self, name: str, nominal_positions: List[vct3],
                            is_calibration: bool = False) -> None:
        """DefineMarkerBody({a_m,1, ..., a_m,N})"""
        self._post("/marker-bodies", {
            "name": name,
            "nominal_positions": [_vct3_to_json(v) for v in nominal_positions],
            "is_calibration": is_calibration,
        })

    def update_marker_body(self, name: str, nominal_positions: List[vct3]) -> List[vct3]:
        """UpdateMarkerBody(mB, {a_m,1, ..., a_m,N}) -> PreviousValues"""
        result = self._put(f"/marker-bodies/{name}", {
            "nominal_positions": [_vct3_to_json(v) for v in nominal_positions],
        })
        return [_json_to_vct3(v) for v in result["previous_positions"]]

    def place_marker_body(self, name: str, F: Frame) -> None:
        """PlaceMarkerBody(mB, Fmb)"""
        self._post(f"/marker-bodies/{name}/place", _frame_to_json(F))

    def most_recent_observed_frame(self, name: str) -> Frame:
        """Fm = MostRecentObservedFrame(mB)"""
        return _json_to_frame(self._get(f"/marker-bodies/{name}/frame"))

    def most_recent_observed_markers(self, name: str) -> List[vct3]:
        """{a_m,1, ..., a_m,N} = MostRecentObservedMarkers(mB)"""
        result = self._get(f"/marker-bodies/{name}/markers")
        return [_json_to_vct3(v) for v in result["markers"]]

    # -- tracker ----------------------------------------------------------

    def create_tracker(self, name: str, marker_body_names: List[str],
                        F: Optional[Frame] = None, R: Optional[Rot] = None,
                        p: Optional[vct3] = None) -> List[str]:
        """CreateTracker(F, {mB1, ..., mBN}) -> tracked marker body names"""
        result = self._post("/trackers", {
            "name": name,
            "marker_body_names": marker_body_names,
            "F": _pose_json(F, R, p),
        })
        return result["tracked_marker_bodies"]

    def define_tracker_pose(self, name: str, F: Optional[Frame] = None,
                             R: Optional[Rot] = None, p: Optional[vct3] = None) -> None:
        """defineFTracker(...)"""
        self._post(f"/trackers/{name}/pose", _pose_json(F, R, p))

    def add_marker_bodies(self, name: str, marker_body_names: List[str]) -> List[str]:
        """AddMarkerBodies(trk, {mB1, ..., mBN}) -> PreviousList"""
        result = self._post(f"/trackers/{name}/marker-bodies:add",
                             {"marker_body_names": marker_body_names})
        return result["previous_list"]

    def remove_marker_bodies(self, name: str, marker_body_names: List[str]) -> List[str]:
        """RemoveMarkerBodies(trk, {mB1, ..., mBN}) -> PreviousList"""
        result = self._post(f"/trackers/{name}/marker-bodies:remove",
                             {"marker_body_names": marker_body_names})
        return result["previous_list"]

    def sample_marker_bodies(self, name: str) -> Dict[str, Frame]:
        """SampleMarkerBodies(trk) -> {F1, ..., FN}, one noisy reading of
        every tracked body's pose relative to the tracker."""
        result = self._post(f"/trackers/{name}/sample")
        return {mb_name: _json_to_frame(f) for mb_name, f in result["frames"].items()}

    # -- pointer + calibration sensor --------------------------------------

    def define_pointer(self, name: str, marker_body_name: str,
                        nominal_tip_position: vct3) -> None:
        self._post("/pointers", {
            "name": name,
            "marker_body_name": marker_body_name,
            "nominal_tip_position": _vct3_to_json(nominal_tip_position),
        })

    def hand_guide_pointer(self, name: str, cal_name: str, R_desired: Rot) -> None:
        """Simulates hand-guiding the pointer so its tip lands within ~5mm
        of the calibration object's origin, at orientation R_desired."""
        self._post(f"/pointers/{name}/hand-guide", {
            "cal_name": cal_name,
            "R_desired": R_desired.matrix.tolist(),
        })

    def sense_tip(self, cal_name: str, pointer_name: str) -> Optional[vct3]:
        """Reads the calibration sensor: the pointer tip's position relative
        to the calibration object, or None if currently out of range."""
        result = self._post(f"/calibration-object/{cal_name}/sense-tip",
                             {"pointer_name": pointer_name})
        if not result["in_range"]:
            return None
        return _json_to_vct3(result["sensed_tip"])

    # -- debug (only available when the server was started with
    #    CIS1_MODE=debug -- otherwise these 404, surfaced as TrackerAPIError) --

    def debug_ground_truth(self) -> dict:
        return self._get("/debug/ground-truth")

    def debug_override_config(self, jiggle_cov=None, sensor_noise_cov=None,
                               marker_manufacturing_cov=None) -> None:
        self._post("/debug/config", {
            "jiggle_cov": _cov_to_json(jiggle_cov),
            "sensor_noise_cov": _cov_to_json(sensor_noise_cov),
            "marker_manufacturing_cov": _cov_to_json(marker_manufacturing_cov),
        })

    def health(self) -> dict:
        return self._get("/health")
