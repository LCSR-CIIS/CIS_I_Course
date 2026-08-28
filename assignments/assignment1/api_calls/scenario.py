"""
Load a Tracker API setup (marker bodies, tracker, pointer) from a YAML file
instead of writing it out as TrackerClient calls by hand.

    from api_calls.tracker_client import TrackerClient
    from api_calls.scenario import load_scenario

    client = TrackerClient()
    load_scenario(client, "api_calls/scenario_example.yaml")

Only covers the one-time *setup* calls -- defining/placing marker bodies,
creating the tracker, defining the pointer. The per-observation loop
(hand_guide_pointer / sample_marker_bodies / sense_tip) is inherently a
runtime loop, not static config, so it stays a plain Python `for` loop
calling TrackerClient methods directly (see api_calls/README.md).

YAML shape (see scenario_example.yaml for a full example):

    marker_bodies:
      - name: ptr_mb
        positions: [[0, 0, 0], [50, 0, 0], ...]
        is_calibration: false   # optional, default false
        place:                 # optional -- calls place_marker_body
          R: [[1,0,0],[0,1,0],[0,0,1]]   # optional, default identity
          p: [500, 0, 0]                 # optional, default [0,0,0]

    trackers:
      - name: trk1
        marker_body_names: [ptr_mb, cal_mb]
        pose:                  # optional -- R and/or p, default identity/origin
          R: [[1,0,0],[0,1,0],[0,0,1]]
          p: [0, 0, 0]

    pointers:
      - name: ptr1
        marker_body_name: ptr_mb
        nominal_tip_position: [0, 0, -150]
"""

from typing import List

import yaml

import src  # noqa: F401
from data_types.nominal_types import Rot, vct3, Frame

from .tracker_client import TrackerClient


def _to_vct3(value) -> vct3:
    if isinstance(value, dict):
        return vct3(value["x"], value["y"], value["z"])
    return vct3(value)


def _to_positions(values) -> List[vct3]:
    return [_to_vct3(v) for v in values]


def _to_frame(pose: dict) -> Frame:
    R = Rot(matrix=pose["R"]) if "R" in pose else Rot()
    p = _to_vct3(pose["p"]) if "p" in pose else vct3()
    return Frame(R, p)


def load_scenario(client: TrackerClient, path: str) -> None:
    """Reads `path` (YAML) and issues the equivalent TrackerClient setup calls."""
    with open(path) as f:
        scenario = yaml.safe_load(f) or {}

    for mb in scenario.get("marker_bodies", []):
        client.define_marker_body(
            mb["name"],
            nominal_positions=_to_positions(mb["positions"]),
            is_calibration=mb.get("is_calibration", False),
        )
        if "place" in mb:
            client.place_marker_body(mb["name"], _to_frame(mb["place"]))

    for trk in scenario.get("trackers", []):
        pose = trk.get("pose", {})
        R = Rot(matrix=pose["R"]) if "R" in pose else None
        p = _to_vct3(pose["p"]) if "p" in pose else None
        client.create_tracker(
            trk["name"],
            marker_body_names=trk.get("marker_body_names", []),
            R=R if R is not None else Rot(),
            p=p if p is not None else vct3(),
        )

    for ptr in scenario.get("pointers", []):
        client.define_pointer(
            ptr["name"],
            marker_body_name=ptr["marker_body_name"],
            nominal_tip_position=_to_vct3(ptr["nominal_tip_position"]),
        )
