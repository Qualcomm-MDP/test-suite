# PSM Test Suite

A benchmarking tool for the [Photorealistic Street Maps (PSM)](https://github.com/Qualcomm-MDP/PSM) pipeline. Run PSM on a geographic area, collect performance and evaluation metrics, and compare results across machines and configurations.

The suite uses a separate local PSM checkout and does not modify PSM.

## Setup

**Requirements:**
- Python 3.11+
- A local [PSM](https://github.com/Qualcomm-MDP/PSM) checkout with dependencies installed (`poetry install`)
- A Mapillary access token in `<PSM>/.env` (`MAPILLARY_ACCESS_TOKEN=...`)
- Internet access for imagery, map data, and elevation data

For geometry evaluation, install `laspy[lazrs]` in PSM's Python environment:

```bash
cd /path/to/PSM
poetry run pip install "laspy[lazrs]"
```

Clone the test suite and create your local configuration:

```bash
git clone https://github.com/Qualcomm-MDP/test-suite.git
cd test-suite
cp config.example.toml config.toml
```

Edit `config.toml` to specify your PSM checkout, test region, and pipeline options:

```toml
[psm]
dir = "../PSM"

[region]
name = "Ann Arbor"
bbox = [-83.74998, 42.28025, -83.74695, 42.28250] # west, south, east, north

[pipeline]
terrain = true
segmentation = true
poses = "raw"
```

See `config.example.toml` for all available settings. You can also specify the PSM path using `--psm-dir`.

## Run a Test

```bash
# Check configuration without running PSM
python3 run_test.py --config config.toml --dry-run

# Run the pipeline and evaluations
python3 run_test.py --config config.toml
```

Override configuration settings for individual runs:

```bash
# Test another bounding box
python3 run_test.py --config config.toml --bbox W S E N

# Run without segmentation
python3 run_test.py --config config.toml --no-seg
```

Additional PSM arguments can be passed after `--`.

The first full run may download approximately 1 GB of model weights.

## Results and Data Files

Each run creates a new folder under `results/`. Previous runs are never overwritten.

The repository also includes a completed segmentation-enabled Ann Arbor benchmark in [`examples/ann-arbor-full/`](examples/ann-arbor-full/), containing real performance and geometry evaluation data.

### Example Results

| File | Contents |
|---|---|
| [`summary.json`](examples/ann-arbor-full/summary.json) | Overall run results, including hardware, configuration, runtime, memory, and evaluation scores |
| [`report.md`](examples/ann-arbor-full/report.md) | Human-readable summary of the benchmark |
| [`artifacts/profile.json`](examples/ann-arbor-full/artifacts/profile.json) | Per-stage wall time, CPU time, and available memory measurements |
| [`eval/geometry/summary.json`](examples/ann-arbor-full/eval/geometry/summary.json) | Aggregate geometry accuracy statistics for M1 (building height), M2 (terrain/base), and M3 (roof elevation) |
| [`eval/geometry/per_building.csv`](examples/ann-arbor-full/eval/geometry/per_building.csv) | Individual building measurements and geometry errors |

**Geometry data:** `eval/geometry/per_building.csv` contains one row per building, including OSM identifiers, height sources and confidence, footprint area, LiDAR measurements, model elevations, and M1–M3 errors. The geometry `summary.json` aggregates these measurements by height confidence and terrain slope, reporting RMSE, MAE, bias, confidence intervals, and accuracy thresholds.

**Performance data:** `artifacts/profile.json` records execution time and memory measurements for each pipeline stage, such as fetching data, terrain generation, mesh construction, texturing, and export. Measurements are per stage, not per image. GPU memory measurements may be unavailable on some platforms, including Apple MPS.

**Example limitation:** The Ann Arbor run completed successfully using an OpenStreetMap API fallback after Overpass became unavailable. Its runtime includes initial model downloads and should not be treated as a controlled hardware benchmark.

### Additional Files from Full Runs

Locally generated runs under `results/` also include:

| File | Contents |
|---|---|
| `artifacts/model.glb` | Generated 3D model |
| `artifacts/mapillary.json` | Mapillary image metadata |
| `artifacts/model.overlays.json` | Camera positions and overlay information |
| `eval/geometry/per_building_errors.geojson` | Building-level errors with geographic footprints, viewable in QGIS |
| Geometry evaluation plots | Visualizations of geometry accuracy and error distributions |
| `eval/geometry/summary.md` | Detailed geometry evaluation report |
| `logs/` | Pipeline and evaluation logs |
| LiDAR and terrain caches | Reference data used for geometry evaluation |

The `results/` directory is Git-ignored to avoid committing large models, cached data, and logs. The smaller `examples/ann-arbor-full/` directory contains the numerical results needed for analysis.

## Compare Runs

For meaningful comparisons, use the same geographic area, pipeline configuration, and PSM commit across machines. Cached models and input data should also be consistent when comparing runtime.

Collect the result folders in one place, then run:

```bash
python3 compare.py results/*/summary.json --md compare.md --csv compare.csv
```

The comparison includes runtime, memory, and geometry scores, and flags differences in configurations or Git commits.

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
