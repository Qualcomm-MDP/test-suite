"""Evaluations that run after the pipeline. Each one calls PSM's own script/code, or reports
SKIPPED with the reason. To add a newly merged PSM evaluation, write one function
`def name(ctx) -> dict` that returns result(...) and append it to EVALUATIONS."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .pipeline import git

GEOMETRY_PATH = "scripts/eval/eval_geometry_lidar.py"
GEOMETRY_METRICS = ["M1_height_err", "M2_terrain_err", "M2_base_err", "M3_roof_err"]
GEOMETRY_DEPS = ["numpy", "scipy", "shapely", "pyproj", "requests", "matplotlib", "laspy", "lazrs"]


def result(name: str, status: str, source: str, *, reason: str | None = None, scores: dict | None = None,
           units: str = "", interpretation: str = "", artifacts: list | None = None,
           log: str | None = None) -> dict:
    assert status in ("OK", "SKIPPED", "FAILED"), status
    return {"name": name, "status": status, "source": source, "reason": reason, "scores": scores or {},
            "units": units, "interpretation": interpretation, "artifacts": artifacts or [], "log": log}


def missing_modules(python: str, modules: list[str]) -> list[str]:
    code = ("import importlib.util as u, sys\n"
            "print(' '.join(m for m in sys.argv[1:] if u.find_spec(m) is None))")
    try:
        r = subprocess.run([python, "-c", code, *modules], capture_output=True, text=True, timeout=60)
        return r.stdout.split() if r.returncode == 0 else modules
    except (OSError, subprocess.SubprocessError):
        return modules


def locate_script(repo: Path, rel: str, ref: str, dest: Path) -> tuple[Path | None, str]:
    """The script in the checkout; else read from `ref` with `git show` (the checkout is
    untouched) into the run folder, so unmerged evaluations can still be used."""
    if (repo / rel).exists():
        return repo / rel, f"{rel} (working tree)"
    sha = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    text = git(repo, "show", f"{ref}:{rel}") if sha else None
    if not text:
        return None, f"{rel} is not in the checkout or in {ref}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text + "\n")
    return dest, f"{rel} from {ref} @ {sha[:7]} (unmerged; read with git show)"


def _run(cmd: list[str], cwd: Path, env: dict, log: Path, timeout: float) -> int:
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            return subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout:.0f} s\n")
            return -1


def _tail(path: Path, n: int = 8) -> str:
    lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip()]
    return " | ".join(lines[-n:])


def geometry_lidar(ctx: dict) -> dict:
    name = "geometry_vs_lidar"
    meaning = ("M1 height, M2 terrain, M3 roof-elevation error per building (model - LiDAR, m; "
               "+ = model too high). QR-06a terrain: share of buildings with |M2| <= 1 m; "
               "QR-06b height: share with |M1| <= 3 m (a baseline, not pass/fail). Flat models "
               "(no --terrain) only get M1.")
    script, source = locate_script(ctx["repo"], GEOMETRY_PATH, ctx["geometry_ref"],
                                   ctx["run_dir"] / "tools" / "eval_geometry_lidar.py")
    if script is None:
        return result(name, "SKIPPED", source, reason=source)
    if not ctx["glb"].exists():
        return result(name, "SKIPPED", source, reason="no .glb from the pipeline")
    miss = missing_modules(ctx["python"], GEOMETRY_DEPS)
    if miss:
        return result(name, "SKIPPED", source, reason=(
            f"missing in the pipeline env: {', '.join(miss)} "
            f"(install: {ctx['python']} -m pip install \"laspy[lazrs]\")"))
    out = ctx["run_dir"] / "eval" / "geometry"
    b = ctx["bbox"]
    cmd = [ctx["python"], str(script), str(ctx["glb"]), "--bbox",
           *(str(b[k]) for k in ("west", "south", "east", "north")),
           "--out", str(out), "--cache-dir", str(ctx["cache_dir"] / "lidar")]
    if ctx.get("laz"):
        cmd += ["--laz", *ctx["laz"]]
    log = ctx["logs"] / "eval_geometry.log"
    env = {**ctx["env"], "PYTHONPATH": str(ctx["repo"] / "src")}
    rc = _run(cmd, ctx["run_dir"], env, log, ctx["eval_timeout"])
    summ = out / "summary.json"
    if rc != 0 or not summ.exists():
        return result(name, "FAILED", source, log=str(log),
                      reason=f"exit {rc}: {_tail(log)}")
    return result(name, "OK", source, scores=geometry_scores(json.loads(summ.read_text())),
                  units="metres (errors); fraction 0-1 (shares)", interpretation=meaning,
                  artifacts=[str(out / f) for f in ("summary.md", "summary.json", "per_building.csv")],
                  log=str(log))


def geometry_scores(s: dict) -> dict:
    req = s.get("requirements", {})
    t, h = req.get("terrain", {}), req.get("height", {})
    scores = {"terrain_QR06a": t.get("result"), "terrain_share_within_tol": t.get("share"),
              "terrain_share_ci95": t.get("ci95"), "terrain_n": t.get("n"),
              "height_share_within_tol": h.get("share"), "height_share_ci95": h.get("ci95"),
              "height_n": h.get("n"), "building_status_counts": s.get("status_counts")}
    allg = s.get("groups", {}).get("all", {})
    for m in GEOMETRY_METRICS:
        d = allg.get(m, {})
        for k in ("n", "median", "rmse", "mae"):
            scores[f"{m}_{k}"] = d.get(k)
    return scores


def alignment_stats(ctx: dict) -> dict:
    """PSM's own alignment outputs (pose_alignment/align.py summary.json, stats.json)."""
    name, source = "pose_alignment_stats", "src/pose_alignment (outputs/alignment/mapillary_roi_*)"
    if "colmap" not in " ".join(ctx["pipeline_args"]):
        return result(name, "SKIPPED", source, reason="only produced with --poses colmap "
                      "(needs COLMAP 4.2+ and LightGlue); this run did not use it")
    roots = sorted((ctx["repo"] / "outputs" / "alignment").glob("mapillary_roi_*"),
                   key=lambda p: p.stat().st_mtime)
    roots = [r for r in roots if r.stat().st_mtime >= ctx["started"]]
    if not roots:
        return result(name, "FAILED", source, reason="no new alignment folder after the run")
    dest = ctx["run_dir"] / "eval" / "alignment"
    scores, arts = {}, []
    for f in list(roots[-1].rglob("summary.json")) + list(roots[-1].rglob("stats.json")):
        rel = f.relative_to(roots[-1])
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / rel)
        arts.append(str(dest / rel))
        data = json.loads(f.read_text())
        if isinstance(data, dict):
            scores.update({f"{rel.parent}/{k}".lstrip("./"): v for k, v in data.items()
                           if isinstance(v, (int, float))})
    return result(name, "OK", source, scores=scores, artifacts=arts,
                  units="see PSM align.py (metres, counts, seconds)",
                  interpretation="points_to_footprint_* = distance of reconstructed points to OSM "
                  "walls (lower is better); frames_aligned/reliable = images given a COLMAP pose")


def image_metrics(ctx: dict) -> dict:
    have = {"blender": bool(shutil.which("blender")),
            "eval-venv": (Path.home() / "eval-venv" / "bin" / "python").exists()}
    return result("render_vs_photo_metrics", "SKIPPED",
                  "scripts/eval/{simple_compare,metric_study,pose_ab,pose_outline_score}.py",
                  reason="these score single-building exports (scripts/export_single_building.py) "
                  "rendered from Mapillary cameras in Blender, not a bbox model; not wired in. "
                  f"Prerequisites here: {', '.join(f'{k}={v}' for k, v in have.items())}")


def texture_confidence(ctx: dict) -> dict:
    present = (ctx["repo"] / "src/common/per_patch_confidence.py").exists()
    return result("texture_confidence", "SKIPPED", "src/common/per_patch_confidence.py",
                  reason=("present, but it writes confidence into texture alpha and emits no score"
                          if present else "only on unmerged branch "
                          "feat/colmap-pose-alignment-dervin-test; writes confidence into texture "
                          "alpha and emits no score file"))


def texture_uv_fid(ctx: dict) -> dict:
    return result("texture_quality_uv_fid", "SKIPPED", "none",
                  reason="the UV texture-quality metric (O, D, FID) is not implemented on any PSM "
                  "branch; LPIPS/DreamSim scripts are different metrics and are not substituted")


EVALUATIONS = [geometry_lidar, texture_uv_fid, texture_confidence, alignment_stats, image_metrics]


def run_all(ctx: dict) -> list[dict]:
    out = []
    for fn in EVALUATIONS:
        try:
            out.append(fn(ctx))
        except Exception as e:  # one broken evaluation must not stop the others
            out.append(result(fn.__name__, "FAILED", fn.__module__, reason=f"{type(e).__name__}: {e}"))
    return out
