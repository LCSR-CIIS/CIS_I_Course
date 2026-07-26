# Tracker API

An HTTP API that lets you call `Tracker.read_markers` without seeing the
tracker's internal pose (`F`) or covariance (`C`). Those are set once, at
startup, in `_instructor_config.py` (not shared with students).

## Start the backend

Run these commands from the `cis1/` directory (one level up from this
folder) — this matters because `python -m` adds the current directory to
`sys.path`, which is how `api_calls.api` is able to `import src`.

```bash
cd cis1
python -m uvicorn api_calls.api:app --reload # change to be also compatible with windows 
```

If `uvicorn` isn't on your `PATH`, use the interpreter that has `fastapi`/
`uvicorn` installed directly, e.g.:

```bash
/opt/miniforge3/envs/default/bin/python -m uvicorn api_calls.api:app --reload
```

By default the server listens on `http://127.0.0.1:8000`. Add
`--host 0.0.0.0 --port 8000` if you need it reachable from other machines.

## Call the API

### `POST /read_markers`

Request body:

```json
{
  "markers": [
    {"x": 1.0, "y": 2.0, "z": 3.0},
    {"x": 4.0, "y": 5.0, "z": 6.0}
  ]
}
```

Response body:

```json
{
  "samples": [
    [1.02, 1.97, 3.05],
    [4.03, 4.98, 5.99]
  ]
}
```

Each entry in `samples` is a noisy reading of the corresponding input marker,
in the tracker's own coordinate frame.

### curl

```bash
curl -s -X POST http://127.0.0.1:8000/read_markers \
  -H "Content-Type: application/json" \
  -d '{"markers": [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}]}'
```

### Python

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/read_markers",
    json={"markers": [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}]},
)
print(response.json()["samples"])
```

### Interactive docs

FastAPI auto-generates Swagger UI while the server is running:
`http://127.0.0.1:8000/docs`
