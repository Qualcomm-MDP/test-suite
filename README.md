# PSM Test Suite

A benchmarking tool for the [Photorealistic Street Maps (PSM)](https://github.com/Qualcomm-MDP/PSM) pipeline. Run PSM on a geographic area, collect performance and evaluation metrics, and compare results across machines and configurations.

The suite runs a separate local PSM checkout; it does not include or modify PSM.

## Setup

**Requirements:**
- Python 3.11+
- A local [PSM](https://github.com/Qualcomm-MDP/PSM) checkout with dependencies installed (`poetry install`)
- A Mapillary access token in `<PSM>/.env` (`MAPILLARY_ACCESS_TOKEN=...`)
- Internet access for imagery, map data, and elevation data

For geometry evaluation, install `laspy[lazrs]` in PSM's environment:

```bash
poetry run pip install "laspy[lazrs]"
```

Clone the test suite and create your local configuration:

```bash
git clone <repo-url>
cd psm-test-suite
cp config.example.toml config.toml
```

Edit `config.toml` to specify your PSM checkout, test region, and pipeline options:

```toml
[psm]
dir = "../PSM"

[region]
name = "Ann Arbor"
bbox = [-83.74998, 42.28025, -83.74695, 42.28250] #west south east north

[pipeline]
terrain = true
segmentation = true
poses = "raw"
```

See `config.example.toml` for all available settings. The PSM path can also be specified with `--psm-dir`.

## Run a Test

```bash
# Check configuration without running PSM
python3 run_test.py --config config.toml --dry-run

# Run the pipeline and evaluations
python3 run_test.py --config config.toml
```

You can override settings for individual runs:

```bash
# Test another bounding box
python3 run_test.py --config config.toml --bbox W S E N

# Run without segmentation
python3 run_test.py --config config.toml --no-seg
```

Additional PSM arguments can be passed after `--`.

The first full run may download approximately 1 GB of model weights.

## Results

Each run creates a separate folder under `results/`. Previous runs are never overwritten.

```text
results/<run-id>/
├── summary.json       # Structured metrics and run metadata
├── report.md          # Human-readable results
├── logs/              # Pipeline and evaluation logs
├── artifacts/         # GLB model, profiler data, imagery metadata
└── eval/              # Evaluation scores and supporting files
```

Results include:
- **Hardware:** CPU, GPU, RAM, and available compute backend
- **Configuration:** geographic area, pipeline settings, and Git commits
- **Performance:** total and per-stage runtime, CPU time, and memory
- **Evaluation:** geometry accuracy and other available metrics

## Compare Runs

Run the same configuration and PSM commit on different machines, then collect their result folders in one place.

```bash
python3 compare.py results/*/summary.json --md compare.md --csv compare.csv
```

The comparison includes runtime, memory, and geometry scores, and flags differences in configurations or commits.

## Evaluations

| Evaluation | Availability |
|---|---|
| Geometry vs. LiDAR (building height, terrain, roof elevation) | Automatic when LiDAR data and dependencies are available |
| Pose alignment | Requires COLMAP and `poses = "colmap"` |
| Texture quality and image comparison | Skipped when implementations or dependencies are unavailable |

**Geometry evaluation setup:** Until `eval_geometry_lidar.py` is merged into PSM's main branch, fetch `origin/feat/geometry-eval` in your PSM checkout. Geometry evaluation requires USGS LiDAR coverage or locally supplied LAZ data.

Unavailable evaluations are marked `SKIPPED` with an explanation.

## Development

Run the offline tests with:

```bash
python3 -m unittest discover -s tests
```
