# Tracker API

An HTTP API that simulates the optical tracker described in the Assignment 1
handout. Students submit *approximate/nominal* geometry (marker-sphere
layouts, the tracker's pose, a calibration object's placement); the server
applies hidden, instructor-configured noise (see `_instructor_config.py`,
not shared with students) and returns realistic noisy readings.

## Start the backend

Run this from the `assignments/assignment1/` directory (one level up from
this folder) — that matters because `python -m` adds the current directory
to `sys.path`, which is how `api_calls.api` is able to `import src`.

```bash
cd assignments/assignment1
python -m uvicorn api_calls.api:app --reload
```

If `uvicorn` isn't on your `PATH`, use the interpreter that has `fastapi`/
`uvicorn` installed directly, e.g. `python3 -m pip install -r requirements.txt`
first, or invoke that interpreter's `python -m uvicorn ...` explicitly. This
works the same way on macOS, Linux, and Windows.

By default the server listens on `http://127.0.0.1:8000`. Add
`--host 0.0.0.0 --port 8000` if you need it reachable from other machines.

### Debug mode

By default the server runs in **eval** mode: ground-truth values are never
exposed, matching a real grading run. Set `CIS1_MODE=debug` before starting
the server to additionally mount `/debug/ground-truth` (read cached actual
marker/tip positions and tracker pose) and `/debug/config` (override the
instructor's noise covariances for your own testing):

```bash
CIS1_MODE=debug python -m uvicorn api_calls.api:app --reload
```

Debug routes return 404 if the server wasn't started with `CIS1_MODE=debug`
— this is decided once, at server startup, not per-request.

## Interactive docs

FastAPI auto-generates Swagger UI while the server is running:
`http://127.0.0.1:8000/docs` — the quickest way to explore every route's
exact request/response schema.

## Python client (recommended)

Instead of calling the routes below directly, use `tracker_client.TrackerClient`
(`api_calls/tracker_client.py`, alongside this README) -- one Python method
per route, taking and returning the assignment's own `vct3`/`Rot`/`Frame`
types instead of raw JSON dicts. See the "Example" section below.

## Routes

| Method + path | Assignment capability |
|---|---|
| `POST /marker-bodies` | `DefineMarkerBody({a_m,1, ..., a_m,N})` — define a marker body (pointer, calibration object, or any other) from its approximate local marker-sphere positions. Pass `"is_calibration": true` for the calibration object. |
| `PUT /marker-bodies/{name}` | `UpdateMarkerBody(mB, {...})` — overwrite a marker body's nominal positions (e.g. after averaging several observations); returns the previous values. |
| `POST /marker-bodies/{name}/place` | `PlaceMarkerBody(mB, Fmb)` — place a marker body at an approximate pose. |
| `GET /marker-bodies/{name}/frame` | `MostRecentObservedFrame(mB)` |
| `GET /marker-bodies/{name}/markers` | `MostRecentObservedMarkers(mB)` |
| `POST /trackers` | `CreateTracker(F, {mB1, ..., mBN})` |
| `POST /trackers/{name}/pose` | `defineFTracker(...)` — update a tracker's approximate pose (accepts a full frame, `R`+`p`, `R` alone, or `p` alone). |
| `POST /trackers/{name}/marker-bodies:add` | `AddMarkerBodies(trk, {...})` |
| `POST /trackers/{name}/marker-bodies:remove` | `RemoveMarkerBodies(trk, {...})` |
| `POST /trackers/{name}/sample` | `SampleMarkerBodies(trk)` — one noisy reading of every tracked body; returns each body's pose relative to the tracker. |
| `POST /pointers` | Define a pointer: its approximate tip position and which marker body it's attached to. |
| `POST /pointers/{name}/hand-guide` | Simulates hand-guiding the pointer so its tip is within ~5mm of a calibration object's origin, at a desired orientation. |
| `POST /calibration-object/{name}/sense-tip` | Reads the calibration sensor: the pointer tip's position relative to the calibration object, to near-zero error, if currently within 5mm. |
| `GET /debug/ground-truth`, `POST /debug/config` | debug-mode only (see above) |
| `GET /health` | liveness check |

## Example: end-to-end pivot-calibration data collection

### Using `TrackerClient`

```python
from api_calls.tracker_client import TrackerClient
from data_types.nominal_types import vct3, Rot, Frame

client = TrackerClient()

# 1. Define marker bodies (approximate local marker-sphere layouts)
client.define_marker_body("ptr_mb", nominal_positions=[
    vct3(0, 0, 0), vct3(50, 0, 0), vct3(0, 50, 0), vct3(0, 0, 50), vct3(30, 20, 10),
])
client.define_marker_body("cal_mb", is_calibration=True, nominal_positions=[
    vct3(0, 0, 0), vct3(60, 0, 0), vct3(0, 60, 0), vct3(0, 0, 60),
])

# 2. Place the calibration object and create the tracker (approximate poses)
client.place_marker_body("cal_mb", Frame(Rot(), vct3(500, 0, 0)))
client.create_tracker("trk1", marker_body_names=["ptr_mb", "cal_mb"], p=vct3(0, 0, 0))

# 3. Define the pointer
client.define_pointer("ptr1", marker_body_name="ptr_mb", nominal_tip_position=vct3(0, 0, -150))

# 4-6. For each observation j: hand-guide near the calibration object,
# sample both bodies' poses, and read the accurate calibration-sensor tip.
for _ in range(10):
    client.hand_guide_pointer("ptr1", cal_name="cal_mb", R_desired=Rot())
    sample = client.sample_marker_bodies("trk1")   # {"ptr_mb": Frame, "cal_mb": Frame}
    sensed = client.sense_tip("cal_mb", pointer_name="ptr1")   # vct3 or None
    # ... accumulate (sample["ptr_mb"], sample["cal_mb"], sensed) for pivot calibration
```

### Using a config file for setup

The one-time setup calls above (steps 1-3) can instead come from a YAML
file via `scenario.load_scenario` -- see `api_calls/scenario_example.yaml`
for the file equivalent of steps 1-3. The observation loop (steps 4-6) is
still plain Python, since it's a runtime loop, not static config:

```python
from api_calls.tracker_client import TrackerClient
from api_calls.scenario import load_scenario

client = TrackerClient()
load_scenario(client, "api_calls/scenario_example.yaml")   # does steps 1-3 above

for _ in range(10):
    client.hand_guide_pointer("ptr1", cal_name="cal_mb", R_desired=Rot())
    sample = client.sample_marker_bodies("trk1")
    sensed = client.sense_tip("cal_mb", pointer_name="ptr1")
```

### Raw HTTP (advanced / debugging)

`TrackerClient` is just a thin wrapper -- if you want to see exactly what's
on the wire, or poke at a route from `/docs`, the same example works with
plain `requests` calls:

```python
import requests
BASE = "http://127.0.0.1:8000"

def post(path, **json): return requests.post(f"{BASE}{path}", json=json).json()

# 1. Define marker bodies (approximate local marker-sphere layouts)
post("/marker-bodies", name="ptr_mb", nominal_positions=[
    {"x": 0, "y": 0, "z": 0}, {"x": 50, "y": 0, "z": 0},
    {"x": 0, "y": 50, "z": 0}, {"x": 0, "y": 0, "z": 50}, {"x": 30, "y": 20, "z": 10},
])
post("/marker-bodies", name="cal_mb", is_calibration=True, nominal_positions=[
    {"x": 0, "y": 0, "z": 0}, {"x": 60, "y": 0, "z": 0},
    {"x": 0, "y": 60, "z": 0}, {"x": 0, "y": 0, "z": 60},
])

# 2. Place the calibration object and create the tracker (approximate poses)
requests.post(f"{BASE}/marker-bodies/cal_mb/place",
               json={"R": [[1,0,0],[0,1,0],[0,0,1]], "p": {"x": 500, "y": 0, "z": 0}})
post("/trackers", name="trk1", marker_body_names=["ptr_mb", "cal_mb"],
     F={"F": {"R": [[1,0,0],[0,1,0],[0,0,1]], "p": {"x": 0, "y": 0, "z": 0}}})

# 3. Define the pointer
post("/pointers", name="ptr1", marker_body_name="ptr_mb",
     nominal_tip_position={"x": 0, "y": 0, "z": -150})

# 4-6. For each observation j: hand-guide near the calibration object,
# sample both bodies' poses, and read the accurate calibration-sensor tip.
for _ in range(10):
    post("/pointers/ptr1/hand-guide", cal_name="cal_mb",
         R_desired=[[1,0,0],[0,1,0],[0,0,1]])
    sample = post("/trackers/trk1/sample")   # {"frames": {"ptr_mb": ..., "cal_mb": ...}, "elapsed_sec": ...}
    sensed = post("/calibration-object/cal_mb/sense-tip", pointer_name="ptr1")
    # ... accumulate (sample["frames"]["ptr_mb"], sample["frames"]["cal_mb"], sensed) for pivot calibration
```
