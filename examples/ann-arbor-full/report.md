# PSM test: Ann Arbor on macbook-pro-m2

**Overall: COMPLETED_DEGRADED**  (run `2026-11-10_ann-arbor_macbook-pro-m2_2`, 2026-11-10T01:02:09+00:00)

| | |
|---|---|
| Area | -83.74998, 42.28025, -83.74695, 42.2825 (W S E N), 0.0624 km² — bbox |
| Machine | Apple M2 Pro, 12 cores, 16 GB RAM, macOS-27.0.1-arm64-arm-64bit-Mach-O |
| Compute | CPU + GPU (MPS usable by torch); ML device PSM would pick: mps |
| PSM | `767fbaa` (main) |
| Options | `--terrain` |
| Pipeline | completed_degraded (exit 0), wall 714.0 s, degraded: overpass_fallback |
| Buildings left out of mesh | 1 |

## Performance (PSM profiler)

Total 706.14 s (188.6 min/km²); largest process RSS 802 MB (OS).

| stage | status | wall s | cpu s | py heap peak MB | CUDA Δ MB |
|---|---|--:|--:|--:|--:|
| fetch | ok | 151.36 | 2.20 | 9.3 | unavailable |
| align_poses | ok | 0.00 | 0.00 | 0.1 | unavailable |
| terrain | ok | 0.65 | 0.11 | 5.5 | unavailable |
| build_mesh | ok | 0.25 | 0.24 | 0.8 | unavailable |
| naip | skipped | 0.00 | 0.00 | 0.0 | unavailable |
| texturing | ok | 553.25 | 320.02 | 356.5 | unavailable |
| export | ok | 0.46 | 0.41 | 78.5 | unavailable |

- py_heap_* is tracemalloc (Python heap only: no torch/OpenCV buffers, no subprocesses)
- cuda_alloc_delta_mb is CUDA only; MPS use is not recorded

## Evaluations

| metric | status | source | result |
|---|---|---|---|
| geometry_vs_lidar | OK | `scripts/eval/eval_geometry_lidar.py from origin/feat/geometry-eval @ 3911640 (unmerged; read with git show)` | terrain_QR06a=PASS, terrain_share_within_tol=1.000, terrain_share_ci95=[1.0, 1.0], terrain_n=56, height_share_within_tol=0.536, height_share_ci95=[0.4107142857142857, 0.6607142857142857], height_n=56, M1_height_err_n=56 |
| texture_quality_uv_fid | SKIPPED | `none` | the UV texture-quality metric (O, D, FID) is not implemented on any PSM branch; LPIPS/DreamSim scripts are different metrics and are not substituted |
| texture_confidence | SKIPPED | `src/common/per_patch_confidence.py` | only on unmerged branch feat/colmap-pose-alignment-dervin-test; writes confidence into texture alpha and emits no score file |
| pose_alignment_stats | SKIPPED | `src/pose_alignment (outputs/alignment/mapillary_roi_*)` | only produced with --poses colmap (needs COLMAP 4.2+ and LightGlue); this run did not use it |
| render_vs_photo_metrics | SKIPPED | `scripts/eval/{simple_compare,metric_study,pose_ab,pose_outline_score}.py` | these score single-building exports (scripts/export_single_building.py) rendered from Mapillary cameras in Blender, not a bbox model; not wired in. Prerequisites here: blender=False, eval-venv=False |

### Geometry vs. LiDAR (all evaluated buildings, metres, model − LiDAR)

| metric | n | median | RMSE | MAE |
|---|--:|--:|--:|--:|
| M1_height_err | 56 | -1.665 | 7.556 | 4.531 |
| M2_terrain_err | 56 | -0.009 | 0.036 | 0.026 |
| M2_base_err | 56 | -0.006 | 0.218 | 0.120 |
| M3_roof_err | 56 | -1.135 | 7.312 | 4.384 |

Building status: {'ok': 56, 'too_small': 1, 'not_in_mesh': 1}

- **geometry_vs_lidar** (metres (errors); fraction 0-1 (shares)): M1 height, M2 terrain, M3 roof-elevation error per building (model - LiDAR, m; + = model too high). QR-06a terrain: share of buildings with |M2| <= 1 m; QR-06b height: share with |M1| <= 3 m (a baseline, not pass/fail). Flat models (no --terrain) only get M1.
