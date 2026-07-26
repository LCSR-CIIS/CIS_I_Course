"""HTTP API that lets students call `Tracker.read_markers` without ever
seeing the tracker's pose (F) or covariance (C).

Students only get to POST marker positions and receive sampled readings back;
the actual `Tracker` instance is built once at startup from
`_instructor_config.py`, which is not distributed to students.

Run locally with:
    python -m uvicorn api_calls.api:app --reload
"""

# everything has to work on both mac and windows. 

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import vct3

from ._instructor_config import build_tracker

app = FastAPI(title='CIS1 Tracker API')

_tracker = build_tracker()

# students can be allowed to change their version of the nominal types of markers
# students can update the name of the marker 
# solve the procrustes problem for them.
# students' assignment is to estimate the covariance of the tracker's pose based on the marker readings

class MarkerPosition(BaseModel): # this should be a vct3
    x: float
    y: float
    z: float


class ReadMarkersRequest(BaseModel):
    markers: List[MarkerPosition] = Field(..., description='Marker positions in world coordinates')


class ReadMarkersResponse(BaseModel):
    samples: List[List[float]]


@app.post('/read_markers', response_model=ReadMarkersResponse)
def read_markers(request: ReadMarkersRequest) -> ReadMarkersResponse:
    marker_positions = [vct3(m.x, m.y, m.z) for m in request.markers]
    samples = _tracker.read_markers(marker_positions)
    return ReadMarkersResponse(samples=[sample.reshape(-1).tolist() for sample in samples])
