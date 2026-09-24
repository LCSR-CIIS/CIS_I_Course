# CIS I — Assignment 1: Pivot Calibration

This assignment is done entirely in a Jupyter notebook. You'll design a
pointer and a calibration object (marker-sphere layouts + approximate
workspace placement), then implement pivot calibration against a simulated
optical tracker.

## What's in this folder

```
assignment1/
├── requirements.txt
├── src/                          the simulation library (don't need to edit this)
│   ├── marker.py, pointer.py, tracker.py, utils.py   -- core classes
│   └── simulation/                -- the functions you'll actually call
└── notebooks/
    ├── assignment1_template.ipynb   <- start here, this is your homework
```

You do not need to read or modify anything under `src/` — you only call
functions from `src.simulation` inside the notebook.

## 1. Set up your environment

You need **Python 3.9 or newer**. Check with:

- Linux/macOS: `python3 --version`
- Windows: `python --version` (or `py --version`)

All commands below are run from the `assignment1/` folder (the one this
README is in).

### Linux / macOS

```bash
cd assignment1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

Command Prompt:

```bat
cd assignment1
py -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

PowerShell:

```powershell
cd assignment1
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PowerShell refuses to run the activation script ("running scripts is
> disabled on this system"), run this once in an **admin** PowerShell, then
> retry: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Once activated, your terminal prompt should show `(venv)`. You'll need to
re-run the activate command (not the venv creation) every time you open a
new terminal to work on this assignment.

## 2. Launch the notebook

With the `venv` still active, pick one:

**Jupyter, from the terminal** (works the same on all OSes):

```bash
jupyter notebook
```

(or `jupyter lab` if you prefer the newer interface). This opens a browser
tab — navigate into `notebooks/` and open `assignment1_template.ipynb`.

**VS Code**: open the `assignment1` folder, open
`notebooks/assignment1_template.ipynb`, and click the kernel picker in the
top-right corner of the notebook. Select the Python interpreter inside your
`venv` (Linux/macOS: `venv/bin/python`; Windows: `venv\Scripts\python.exe`)
— VS Code sometimes needs the "Python" and "Jupyter" extensions installed
first, which it will prompt you to install if missing.

Either way, run the first code cell (the imports) first — if it fails with
a `ModuleNotFoundError`, you likely installed `requirements.txt` into a
different Python environment than the one the notebook's kernel is using;
double check the kernel picker matches your `venv`.

## 3. What you have to do

Open `assignment1_template.ipynb` and work through it top to bottom — it's
organized into sections matching the handout's numbered steps. Markdown
cells explain each section; code cells marked `# TODO` are yours to fill
in, everything else is provided plumbing. In order, you will:

1. **Design** the pointer's and calibration object's marker-sphere layouts,
   and the tracker's/calibration object's approximate workspace poses
   (Steps 1-2 of the handout). This is a real design decision — read the
   markdown notes on marker spread and conditioning before picking numbers.
2. **Implement marker-geometry refinement** — averaging multiple tracker
   readings to tighten your estimate of the true marker positions (Step 3).
3. Run the provided data-collection loop (hand-guide + sample + sense-tip,
   Steps 4-6) — you choose how many observations and how to vary the
   pointer's orientation between them.
4. **Implement pivot calibration itself** (Step 7) — this is the core of
   the assignment; the markdown above that cell walks through the
   derivation and the key design questions to think through (and answer in
   your report), but the math is yours to write.
5. **Validate** against the handout's test-pose grid (Step 8) and report
   your achieved accuracy against the 2mm spec.

The notebook will run cleanly up to wherever your next unfilled TODO is,
then stop with `NotImplementedError` — that's expected, not a bug.

**Don't forget**: the notebook is only the code portion. The handout also
requires a written report (design rationale, math derivation, algorithmic
approach, validation discussion, performance tables, etc.) — see the
assignment for the full report requirements and submission format.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'scipy'` (or `numpy`, etc.)** —
  your kernel isn't using the `venv` you installed `requirements.txt` into.
  Re-check the kernel picker (VS Code) or that you activated `venv` before
  running `jupyter notebook` (terminal).
- **`ModuleNotFoundError: No module named 'src'`** — the notebook assumes
  it's running with its own folder (`notebooks/`) as the working directory,
  which is the default for both Jupyter and VS Code. If you moved or
  renamed files, or launched Jupyter in an unusual way, this can break.
- **`pip install` fails or `venv` module missing (Linux)** — install your
  distro's Python venv package first, e.g. `sudo apt install python3-venv`
  on Debian/Ubuntu, then retry.
