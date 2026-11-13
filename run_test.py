#!/usr/bin/env python3
"""Run the PSM pipeline on one area, then every available evaluation, into one results folder.

  python3 run_test.py --config config.toml                     # the standard command
  python3 run_test.py --baseline                               # the standard benchmark (configs/baseline.toml)
  python3 run_test.py --config config.toml --dry-run           # checks only, runs nothing
  python3 run_test.py --config config.toml --bbox W S E N --name "Main St"
  python3 run_test.py --config config.toml --location "Ann Arbor, Michigan" --size-km 0.5
  python3 run_test.py --config config.toml --no-seg -- --texture best   # after --: raw main.py args
  python3 run_test.py --config config.toml --glb old/model.glb # evaluate a model, no pipeline run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from psmtest import config, evals, hardware, location, network, pipeline, report

HERE = Path(__file__).resolve().parent
# psm-test/1: first runs (absolute artifact paths; run 1 also lacks "suite").
# psm-test/2: paths relative to the run folder, "files" index, pipeline.ml_devices,
#             pipeline.model_downloads, machine.pipeline_python/torch_version/torch_cuda_build,
#             geometry bias/nmad/within_*, stages[].cuda_start/cuda_end. Only additions otherwise.
#             Later additions (still /2): network (before/after/during), pipeline.photos,
#             pipeline.alignment_reused, performance.timing, suite.machine_config_file.
SCHEMA = "psm-test/2"


def rel(path, run_dir: Path):
    """A path inside the run folder as a relative, '/'-separated string; anything else unchanged."""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def file_index(run_dir: Path) -> dict:
    """The run's main files, relative to the run folder; only the ones that exist."""
    wanted = {"summary": "summary.json", "report": "report.md", "pipeline_log": "logs/pipeline.log",
              "profile": "artifacts/profile.json", "glb": "artifacts/model.glb",
              "overlays": "artifacts/model.overlays.json", "mapillary": "artifacts/mapillary.json",
              "geometry_summary": "eval/geometry/summary.json",
              "geometry_per_building": "eval/geometry/per_building.csv",
              "geometry_per_building_geojson": "eval/geometry/per_building_errors.geojson",
              "geometry_log": "logs/eval_geometry.log",
              "texture_summary": "artifacts/texture_eval/summary.json",
              "texture_per_group": "artifacts/texture_eval/per_group.csv"}
    return {k: v for k, v in wanted.items() if k in ("summary", "report") or (run_dir / v).exists()}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--config", help="settings file (default: config.toml next to run_test.py, if present)")
    which.add_argument("--baseline", action="store_true",
                       help="use the standard benchmark settings, configs/baseline.toml (every PSM feature on); "
                            "machine paths still come from your config.toml")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--location", help="place name, geocoded with OSM Nominatim (network)")
    where.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    where.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--size-km", type=float, help="side of a square box around --location/--center")
    ap.add_argument("--max-area-km2", type=float)
    ap.add_argument("--name", help="short area name for the report and results folder")
    ap.add_argument("--psm-dir", "--psm-repo", dest="psm_dir", help="PSM checkout (overrides $PSM_REPO, config)")
    ap.add_argument("--python", help="PSM interpreter (overrides $PSM_PYTHON, config)")
    ap.add_argument("--results", help="results folder (overrides config)")
    ap.add_argument("--machine-label", help="overrides $PSM_MACHINE_LABEL, config, hostname")
    ap.add_argument("--no-terrain", action="store_true")
    ap.add_argument("--no-seg", action="store_true")
    ap.add_argument("--poses", choices=config.POSES,
                    help="camera poses (overrides pipeline.poses); colmap needs COLMAP 4.2+")
    ap.add_argument("--no-evals", action="store_true", help="skip evaluations")
    ap.add_argument("--no-texture-eval", action="store_true",
                    help="don't pass --evaluate-texture to PSM (no UV texture score)")
    ap.add_argument("--no-network-test", action="store_true",
                    help="skip the latency checks and speed test before and after the run "
                         "(bytes received per stage are still recorded)")
    ap.add_argument("--glb", help="evaluate this existing .glb instead of running the pipeline")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and checks; run nothing")
    ap.add_argument("pipeline_args", nargs=argparse.REMAINDER, help="after --: extra src/main.py options")
    a = ap.parse_args(argv)
    a.pipeline_args = [x for x in a.pipeline_args if x != "--"]
    if a.baseline:
        a.config = str(config.BASELINE)
    elif a.config is None and (HERE / "config.toml").exists():
        a.config = str(HERE / "config.toml")
    return a


def settings(a) -> tuple[dict, Path]:
    """Config file + CLI overrides -> final settings (CLI wins)."""
    cfg, base = config.load(a.config)
    a.machine_config = config.layer_machine(cfg, a.config)
    r, p, t = cfg["region"], cfg["pipeline"], cfg["test"]
    if a.bbox or a.location or a.center:  # a region on the command line replaces the config's
        r.update(bbox=a.bbox or [], location=a.location or "", center=a.center or [], size_km=0.0, name="")
    for key, val in (("size_km", a.size_km), ("max_area_km2", a.max_area_km2), ("name", a.name)):
        if val is not None:
            r[key] = val
    if a.no_terrain:
        p["terrain"] = False
    if a.poses:
        p["poses"] = a.poses
    if a.no_seg:
        p["segmentation"] = False
    p["extra_args"] = list(p["extra_args"]) + a.pipeline_args
    if a.no_evals:
        t["run_evaluations"] = False
    if a.no_network_test:
        t["network_test"] = False
    if a.no_evals or a.no_texture_eval:
        p["evaluate_texture"] = False
    config.validate(cfg)
    return cfg, base


def suite_git() -> dict:
    head = pipeline.git(HERE, "rev-parse", "HEAD")
    return {"commit": head, "dirty": bool(pipeline.git(HERE, "status", "--porcelain", "--untracked-files=no"))
            if head else None}


def texture_eval_plan(args: list[str], repo: Path) -> dict:
    """Whether PSM will score the texture in this run, and why not. The flag is dropped for PSM
    checkouts whose src/main.py predates it (they would exit on the unknown option)."""
    if "--evaluate-texture" not in args:
        return {"requested": False, "dropped": False,
                "reason": "turned off (pipeline.evaluate_texture = false, --no-texture-eval or --no-evals)"}
    if not pipeline.main_supports(repo, "--evaluate-texture"):
        return {"requested": False, "dropped": True,
                "reason": "this PSM checkout's src/main.py has no --evaluate-texture "
                          "(added in PSM 9f5dbb6, PR #56); update PSM to score the texture"}
    return {"requested": True, "dropped": False, "reason": None}


def colmap_plan(cfg: dict, base: Path, repo: Path) -> dict:
    """COLMAP for poses = colmap: where it is, its version and CUDA support. Without a usable
    one PSM still runs but keeps Mapillary's poses (the run is then completed_degraded)."""
    if cfg["pipeline"]["poses"] != "colmap":
        return {"used": False, "check": f"not used (poses = {cfg['pipeline']['poses']})"}
    exe, source = config.resolve_colmap(cfg, base, repo)
    info = {"used": True, "source": source, **pipeline.colmap_info(exe)}
    if not exe:
        info["warning"] = ("poses = colmap but COLMAP was not found (set psm.colmap in config.toml, "
                           "$COLMAP_BIN or COLMAP_BIN in <PSM>/.env); PSM will keep Mapillary's poses")
    elif not info["version_ok"]:
        info["warning"] = (f"COLMAP at {exe} reports version {info['version'] or 'unknown'}; PSM needs "
                           f"{'.'.join(map(str, pipeline.MIN_COLMAP))}+ and its alignment may fail")
    info["check"] = (f"{exe} ({source}): COLMAP {info['version']}, "
                     f"{'with' if info['cuda'] else 'without'} CUDA" if exe else "NOT FOUND")
    return info


def new_run_dir(results: Path, stem: str) -> Path:
    """A fresh folder; never reuses one from an earlier run."""
    results.mkdir(parents=True, exist_ok=True)
    for i in range(1000):
        d = results / (stem if i == 0 else f"{stem}_{i + 1}")
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"too many runs named {stem}")


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        cfg, base = settings(a)
        r, p, t, ev = cfg["region"], cfg["pipeline"], cfg["test"], cfg["evaluation"]
        repo, repo_src = config.resolve_psm(a.psm_dir, cfg, base)
        py, py_src = config.resolve_python(a.python, cfg, base, repo)
        colmap = colmap_plan(cfg, base, repo)
        loc = location.resolve(r["location"] or None, r["bbox"] or None, r["center"] or None,
                               r["size_km"] or None, r["max_area_km2"])
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    loc["name"] = r["name"] or loc["query"]
    args = config.pipeline_args(p)
    texture_eval = texture_eval_plan(args, repo)
    if texture_eval["dropped"]:
        args = [x for x in args if x != "--evaluate-texture"]
    results_dir = Path(a.results).resolve() if a.results else config._path(t["results_dir"], base)
    label = a.machine_label or os.environ.get("PSM_MACHINE_LABEL") or t["machine_label"] or None
    if label:
        os.environ["PSM_MACHINE_LABEL"] = label  # hardware.collect reads it
    gitinfo = pipeline.git_info(repo)
    machine = hardware.collect(py, repo, probe=bool(py))
    glb_given = Path(a.glb).resolve() if a.glb else None
    checks = {"psm_dir": f"{repo} ({repo_src})", "pipeline_python": f"{py} ({py_src})",
              "mapillary_token": pipeline.has_token(repo) if not glb_given else "not needed",
              "evaluations": t["run_evaluations"],
              "texture_evaluation": "on" if texture_eval["requested"] else texture_eval["reason"],
              "colmap": colmap["check"],
              "network_test": "on (latency + ~25 MB speed test, before and after)" if t["network_test"]
              else "off (bytes received per stage still recorded)"}
    print(json.dumps({"config": a.config, "machine_config": a.machine_config, "location": loc, "psm": gitinfo, "checks": checks,
                      "compute": machine["compute"], "expected_ml_device": machine["expected_ml_device"],
                      "pipeline_args": args, "results_dir": str(results_dir)}, indent=1))
    if colmap.get("warning"):
        print(f"warning: {colmap['warning']}", file=sys.stderr)
    if not py:
        print("error: PSM interpreter not found (set psm.python, $PSM_PYTHON or --python)", file=sys.stderr)
        return 2
    if a.dry_run:
        return 0
    if checks["mapillary_token"] is False:
        print("error: no MAPILLARY_ACCESS_TOKEN in the environment or <psm>/.env", file=sys.stderr)
        return 2

    now = datetime.now(timezone.utc)
    run_dir = new_run_dir(results_dir, f"{now:%Y-%m-%d}_{location.slug(loc['name'] or 'bbox')}_"
                                       f"{location.slug(machine['label'])}")
    logs, art = run_dir / "logs", run_dir / "artifacts"
    logs.mkdir()
    art.mkdir()
    env = pipeline.utf8_env({**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PSM_ENV_DIR": str(repo)})
    if colmap.get("path"):  # PSM's align_poses reads COLMAP_BIN (it overrides <PSM>/.env)
        env["COLMAP_BIN"] = colmap["path"]

    before = pipeline.repo_snapshot(repo)
    started = time.time()
    if glb_given:
        cmd, res = None, {"exit_code": None, "wall_time_s": None, "max_process_rss_mb": None}
        glb, perf = glb_given, {"available": False, "reason": "--glb: pipeline not run"}
        scan = {"degraded": [], "flagged_buildings": None, "error_lines": []}
        devices = {"expected": machine["expected_ml_device"], "observed": {}, "used": [],
                   "status": "not_run", "note": "--glb: pipeline not run"}
        downloads = {"status": "not_run", "note": "--glb: pipeline not run"}
        photos = {"status": "not_run", "note": "--glb: pipeline not run"}
        net = {"note": "--glb: pipeline not run"}
        status = "not_run"
    else:
        cmd = pipeline.command(py, repo, loc["bbox"], art, args)
        net = {}
        if t["network_test"]:
            print("network check (latency to PSM's services, speed test) ...", flush=True)
            net["before"] = network.probe()
        cache_before = pipeline.cache_snapshot(env)
        with network.Sampler() as sampler:
            res = pipeline.run(cmd, run_dir, env, logs / "pipeline.log")
        cache_after = pipeline.cache_snapshot(env)
        if t["network_test"]:
            net["after"] = network.probe()
        glb = art / "model.glb"
        perf = pipeline.parse_profile(art / "profile.json")
        net["during"] = network.stage_traffic(sampler.samples, perf.get("started_at"), perf.get("stages", []))
        log_text = (logs / "pipeline.log").read_text(encoding="utf-8", errors="replace")
        scan = pipeline.scan_log(log_text)
        devices = pipeline.ml_devices(log_text, machine["expected_ml_device"])
        downloads = pipeline.model_downloads(log_text, cache_before, cache_after)
        photos = pipeline.photo_sources(log_text, args)
        status = pipeline.classify(res["exit_code"], glb.exists(), perf, scan["degraded"])
        if perf.get("available"):
            perf["timing"] = pipeline.timing(perf, photos, log_text)
            if perf["timing"].get("available") and loc["area_km2"]:
                # per km² of processing: downloads and scoring left out, like the headline time
                perf["min_per_km2"] = round(perf["timing"]["processing_s"] / 60 / loc["area_km2"], 2)

    ctx = {"repo": repo, "python": py, "run_dir": run_dir, "logs": logs, "glb": glb, "env": env,
           "bbox": loc["bbox"], "pipeline_args": args, "started": started, "laz": ev["laz"],
           "geometry_ref": ev["geometry_ref"], "cache_dir": HERE / "cache", "eval_timeout": ev["timeout_s"],
           "texture_eval": texture_eval if not glb_given else
           {"requested": False, "reason": "--glb: PSM's texture evaluation runs inside a pipeline run "
            "(it needs the reference photos the run saves)"}}
    if not t["run_evaluations"]:
        results = evals.skip_all("evaluations turned off")
    elif glb.exists():
        results = evals.run_all(ctx)
    else:
        results = evals.skip_all("no model: the pipeline did not produce a .glb")
    after = pipeline.repo_snapshot(repo)
    for e in results:  # relative to the run folder, so collected result folders stay readable
        e["artifacts"] = [rel(x, run_dir) for x in e.get("artifacts", [])]
        e["log"] = rel(e.get("log"), run_dir)

    summary = {
        "schema": SCHEMA, "run_id": run_dir.name, "created_utc": now.isoformat(timespec="seconds"),
        "status": status, "machine": machine, "location": loc,
        "suite": {**suite_git(), "config_file": a.config,
                  "machine_config_file": a.machine_config, "settings": cfg},
        "pipeline": {"git": gitinfo, "psm_dir": str(repo), "python": py, "command": cmd,
                     "pipeline_args": args, "status": status, **res,
                     **{k: scan[k] for k in ("degraded", "flagged_buildings", "error_lines")},
                     "ml_devices": devices, "model_downloads": downloads, "photos": photos,
                     "alignment_reused": photos.get("alignment_reused_from"),
                     "colmap": {k: v for k, v in colmap.items() if k != "check"},
                     "glb": rel(glb, run_dir) if glb.exists() else None,
                     "repo_changes": sorted(after - before) if before is not None and after is not None else None},
        "performance": perf, "network": net, "evaluations": results,
        "files": file_index(run_dir),  # relative to this summary's folder
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(report.markdown(summary), encoding="utf-8")
    print(f"\n{status.upper()}  ->  {run_dir}/report.md")
    for e in results:
        print(f"  {e['status']:8} {e['name']}")
    return 0 if status in ("completed", "completed_degraded", "not_run") else 1


if __name__ == "__main__":
    sys.exit(main())
