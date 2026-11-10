# PSM test suite

This suite runs the [PSM](https://github.com/Qualcomm-MDP/PSM) pipeline on one area, then runs PSM's evaluation scripts on the result. Each run saves a report and a `summary.json`, so you can compare runs across machines, areas and PSM commits.

It **does not include PSM.** It runs an existing PSM checkout through PSM's own CLI (`src/main.py --profile`) and never modifies that checkout.

## Prerequisites

- **Python 3.11+** to run the suite. It uses only the standard library, so there is nothing to install.
- **A PSM checkout with its dependencies installed.** Follow PSM's README (`poetry install`). The geometry evaluation also needs `laspy[lazrs]` in that same environment: `poetry run pip install "laspy[lazrs]"`.
- **A Mapillary token** in `<PSM>/.env`, as `MAPILLARY_ACCESS_TOKEN=...`. PSM's `.env` is git-ignored; never put the token in this repo.
- **Network access** for Mapillary, OSM, elevation tiles and USGS LiDAR. The first run with segmentation on also downloads about 1 GB of model weights.

## Setup

```
git clone <this repo> psm-test-suite && cd psm-test-suite
cp config.example.toml config.toml        # local, git-ignored
```

Edit `config.toml`:

```toml
[psm]
dir = "../PSM"          # your PSM checkout (relative to config.toml, or absolute)
python = ""             # empty = PSM's .venv or Poetry environment, found automatically

[region]
name = "Ann Arbor"
bbox = [-83.74998, 42.28025, -83.74695, 42.28250]   # west, south, east, north

[pipeline]
terrain = true
segmentation = true     # false = --no-seg (faster, no model download)
poses = "raw"           # raw | computed | colmap
```

Every option is described in `config.example.toml`. Each pipeline setting maps to a real `src/main.py` flag. Anything else goes in `extra_args`.

**How the PSM checkout is chosen.** The first match wins:
1. `--psm-dir`
2. `$PSM_REPO`
3. `psm.dir` in the config file
4. `../PSM` or `../qualcomm/PSM` next to the suite

The interpreter is chosen the same way: `--python`, then `$PSM_PYTHON`, then `psm.python`, then automatic discovery.

## Run

```
python3 run_test.py --config config.toml --dry-run   # checks paths, token, hardware; runs nothing
python3 run_test.py --config config.toml
```

Command-line options override the config for one run:
- `--bbox W S E N --name "Main St"`, or `--location "Ann Arbor, Michigan" --size-km 0.5` (geocoded)
- `--no-seg`, `--no-terrain`, `--no-evals`
- `-- <raw main.py args>`

Run the self-checks with `python3 -m unittest discover -s tests`. They are offline and don't need PSM.

## Results

Each run creates a new folder, `results/<date>_<area>_<machine>/`. Earlier runs are never overwritten. It contains:

- **`summary.json`**, with the same fields for every run:
  - hardware: CPU, RAM, GPUs, whether torch can use CUDA or MPS, and the device PSM would pick
  - the exact bbox and area
  - the PSM commit and the suite commit
  - the resolved settings and `main.py` arguments
  - the pipeline status
  - the PSM profiler's per-stage wall time, CPU time and memory
  - the evaluation scores
- **`report.md`**: the same information, readable.
- **`logs/`**: the pipeline and evaluation logs.
- **`artifacts/`**: `model.glb`, `profile.json` and `mapillary.json`.
- **`eval/`**: each evaluation's own output files.

## Comparing machines

Run the same `config.toml` on each machine, at the same PSM commit, with a `machine_label` set or `$PSM_MACHINE_LABEL`. Collect the `results/` folders in one place, then run:

```
python3 compare.py results/*/summary.json --md compare.md --csv compare.csv
```

The table lists runtimes, memory and geometry scores, and says when the bbox, PSM commit, suite commit or options differ between runs.

## Evaluations

| evaluation | runs when |
|---|---|
| Geometry vs. LiDAR (`scripts/eval/eval_geometry_lidar.py`): building height M1, terrain M2, roof elevation M3 | Automatically after every run. Until the script is merged into PSM `main`, it is read from `origin/feat/geometry-eval` with `git show`, so `git fetch` the PSM checkout first. Needs `laspy[lazrs]` and USGS LiDAR coverage (US only; or set `evaluation.laz`) |
| Pose alignment stats | `poses = "colmap"` (needs COLMAP) |
| UV texture quality, texture confidence, render-vs-photo metrics | Reported as SKIPPED, with the reason: not implemented in PSM, unmerged, or they need Blender and single-building exports |

A skipped or failed evaluation never fails the run. The reason is recorded in the report.
