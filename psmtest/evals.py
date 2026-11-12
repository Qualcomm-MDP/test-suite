"""Evaluations that run after the pipeline. Each one calls PSM's own script/code, or reports
SKIPPED with the reason. To add a newly merged PSM evaluation, write one function
`def name(ctx) -> dict` that returns result(...) and append it to EVALUATIONS."""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path

from .pipeline import git, utf8_env

GEOMETRY_PATH = "scripts/eval/eval_geometry_lidar.py"
GEOMETRY_METRICS = ["M1_height_err", "M2_terrain_err", "M2_base_err", "M3_roof_err"]
# Per-metric statistics copied into summary.json (the full set, with CIs and the per-confidence
# and per-slope groups, stays in eval/geometry/summary.json). within_* are shares 0-1.
GEOMETRY_STATS = ("n", "median", "rmse", "mae", "bias_mean", "nmad", "within_1m", "within_3m")
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
        r = subprocess.run([python, "-c", code, *modules], capture_output=True, encoding="utf-8", errors="replace", timeout=60,
                           env=utf8_env(dict(os.environ)))
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
    dest.write_text(text + "\n", encoding="utf-8")
    return dest, f"{rel} from {ref} @ {sha[:7]} (unmerged; read with git show)"


def _run(cmd: list[str], cwd: Path, env: dict, log: Path, timeout: float) -> int:
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            return subprocess.run(cmd, cwd=cwd, env=utf8_env(env), stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout:.0f} s\n")
            return -1


def _tail(path: Path, n: int = 8) -> str:
    lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
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
    return result(name, "OK", source, scores=geometry_scores(json.loads(summ.read_text(encoding="utf-8"))),
                  units="metres (errors); fraction 0-1 (shares)", interpretation=meaning,
                  artifacts=[str(out / f) for f in ("summary.md", "summary.json", "per_building.csv",
                                                     "per_building_errors.geojson") if (out / f).exists()],
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
        for k in GEOMETRY_STATS:
            scores[f"{m}_{k}"] = d.get(k)
    return scores



def alignment_stats(ctx: dict) -> dict:
    """PSM's COLMAP pose alignment (--poses colmap): its outputs/alignment/mapillary_roi_N/ folder
    (the one this run created, or --alignment-dir). Copies data/alignment/summary.json and the
    per-leg stats.json into the run; scores are the headline counts and the step timings."""
    name, source = "pose_alignment_stats", "src/pose_alignment (outputs/alignment/mapillary_roi_*)"
    if "--poses" not in ctx["pipeline_args"] or "colmap" not in ctx["pipeline_args"]:
        return result(name, "SKIPPED", source, reason="only produced with --poses colmap "
                      "(needs COLMAP 4.2+); this run did not use it")
    args = ctx["pipeline_args"]
    if "--alignment-dir" in args:
        given = Path(args[args.index("--alignment-dir") + 1])
        roots = [given if given.is_absolute() else ctx["repo"] / given]
    else:
        roots = sorted((ctx["repo"] / "outputs" / "alignment").glob("mapillary_roi_*"),
                       key=lambda p: p.stat().st_mtime)
        roots = [r for r in roots if r.stat().st_mtime >= ctx["started"]]
    summ = roots[-1] / "data" / "alignment" / "summary.json" if roots else None
    if summ is None or not summ.exists():
        log = ctx["logs"] / "pipeline.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        m = re.search(r"align_poses: alignment failed \((.+)\); every image keeps", text)
        return result(name, "FAILED", source, log=str(log),
                      reason=f"PSM: alignment failed: {m.group(1)}" if m else
                      ("no alignment folder was created during the run" if not roots else
                       f"{roots[-1].name} has no data/alignment/summary.json"))
    dest = ctx["run_dir"] / "eval" / "alignment"
    arts = []
    for f in [summ, *sorted(summ.parent.glob("*/stats.json"))]:
        rel = f.relative_to(summ.parent)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / rel)
        arts.append(str(dest / rel))
    s = json.loads(summ.read_text(encoding="utf-8"))
    return result(name, "OK", source, scores=alignment_scores(s, roots[-1].name), artifacts=arts,
                  units="counts; shares 0-1; seconds",
                  interpretation="frames = photos given to COLMAP; frames_aligned = given an aligned "
                  "pose; frames_reliable = aligned and passing PSM's reliability checks. A leg is a "
                  "stretch of street; a piece is one capture sequence's photos on a leg. wall_fit_* = "
                  "median over aligned pieces of the reconstructed points' distance to the OSM walls "
                  "(m, lower is better) and share within 1 m, with Mapillary's poses (before) and "
                  "aligned (after). fit_s_* = "
                  "PSM's final fitting step (align.py: reconstructed legs onto OSM walls) only; the "
                  "align_poses stage time also covers the photo download, masking and the COLMAP "
                  "reconstruction, which run as subprocesses")


def wall_fit(pieces: list[dict]) -> dict:
    """How well each aligned piece's reconstructed points sit on the OSM walls (align.py
    points_to_footprint), before (Mapillary's poses) and after alignment: the median over pieces
    of each piece's median distance (m) and of its share of points within 1 m. Lower m is better."""
    def med(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(statistics.median(vals), 3) if vals else None
    ptf = [p.get("points_to_footprint") or {} for p in pieces]
    return {"wall_fit_median_m_before": med((x.get("before") or {}).get("median_m") for x in ptf),
            "wall_fit_median_m_after": med((x.get("after") or {}).get("median_m") for x in ptf),
            "wall_within_1m_before": med((x.get("before") or {}).get("within_1m") for x in ptf),
            "wall_within_1m_after": med((x.get("after") or {}).get("within_1m") for x in ptf)}


def alignment_scores(s: dict, roi: str | None = None) -> dict:
    frames, aligned, reliable = s.get("frames"), s.get("frames_aligned"), s.get("frames_reliable")
    pieces = s.get("legs") or []  # align.py: one entry per (leg, capture sequence)
    share = lambda n: round(n / frames, 4) if frames and isinstance(n, (int, float)) else None  # noqa: E731
    return {"roi_folder": roi, "frames": frames, "frames_aligned": aligned, "frames_reliable": reliable,
            "aligned_share": share(aligned), "reliable_share": share(reliable),
            "legs": len({p.get("leg") for p in pieces}),
            "legs_aligned": len({p.get("leg") for p in pieces if p.get("aligned")}),
            "pieces": len(pieces), "pieces_aligned": sum(1 for p in pieces if p.get("aligned")),
            **wall_fit([p for p in pieces if p.get("aligned")]),
            "features": (s.get("reconstruction") or {}).get("features"),
            "mapper": (s.get("reconstruction") or {}).get("mapper"),
            # aggregate step times only (align.py also writes one "fit legNN" entry per leg)
            **{f"fit_s_{k}": v for k, v in (s.get("seconds") or {}).items()
               if isinstance(v, (int, float)) and re.fullmatch(r"[a-z_]+", k)}}


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


TEXTURE_NOT_RUN = re.compile(r"Texture evaluation (?P<what>not available|failed): (?P<reason>.+?)"
                             r"(?:\. The model is kept\.)?\s*$", re.M)


def texture_uv_fid(ctx: dict) -> dict:
    """PSM's UV texture-quality score (src/evaluation/texture_uv.py), run by PSM itself as its
    last pipeline stage (--evaluate-texture), which also saves the reference street photos the
    FID needs. This reads texture_eval/ next to the .glb; PSM never fails the run over it."""
    name, source = "texture_quality_uv_fid", "src/evaluation/texture_uv.py (main.py --evaluate-texture)"
    plan = ctx.get("texture_eval") or {"requested": False, "reason": "not requested"}
    if not plan["requested"]:
        return result(name, "SKIPPED", source, reason=plan["reason"])
    out = ctx["glb"].parent / "texture_eval"
    summ = out / "summary.json"
    if summ.exists():
        return result(name, "OK", source, scores=texture_scores(json.loads(summ.read_text(encoding="utf-8"))),
                      units="0-1 (confidence and O/D/F terms); FID unitless",
                      interpretation=("Confidence = 1 - (a*O + b*D + c*F), 1 = perfect. O obstruction: "
                                      "share of the UV surface without photo texture; D distortion: share "
                                      "of the surface stretched beyond s_max; F realism: 1 - exp(-FID/tau), "
                                      "FID of texture patches vs the run's street photos. partial = F left "
                                      "out (weights renormalised over O and D)."),
                      artifacts=[str(out / f) for f in ("summary.md", "summary.json", "per_group.csv")
                                 if (out / f).exists()])
    log = ctx["logs"] / "pipeline.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    m = None
    for m in TEXTURE_NOT_RUN.finditer(text):
        pass  # keep the last one
    if m:
        status = "SKIPPED" if m.group("what") == "not available" else "FAILED"
        return result(name, status, source, reason=f"PSM: texture evaluation {m.group('what')}: "
                      f"{m.group('reason').strip()}", log=str(log))
    return result(name, "FAILED", source, log=str(log),
                  reason=f"--evaluate-texture was passed but {summ.name} was not written in {out}")


def texture_scores(s: dict) -> dict:
    """texture_uv's summary.json -> the headline numbers (the full file stays in texture_eval/)."""
    terms, counts = s.get("terms") or {}, s.get("counts") or {}
    return {"confidence": s.get("confidence"), "partial": s.get("partial"),
            "O_obstruction": terms.get("O"), "D_distortion": terms.get("D"), "F_realism": terms.get("F"),
            "fid": s.get("fid"), "fid_reason": s.get("fid_reason"), "tau": s.get("tau"),
            "o_texels": s.get("o_texels"), "weights_used": s.get("weights_used"),
            "device": s.get("device"), "seconds": s.get("seconds"), "mode": s.get("mode"),
            "faces": counts.get("faces"), "distorted_faces": counts.get("distorted"),
            "groups": (s.get("web") or {}).get("groups")}


EVALUATIONS = [geometry_lidar, texture_uv_fid, texture_confidence, alignment_stats, image_metrics]
# The name each evaluation reports under, also when it is skipped or crashes, so a metric has one
# name in every summary.json (before 2026-11-12 skipped entries used the function names).
NAMES = {geometry_lidar: "geometry_vs_lidar", texture_uv_fid: "texture_quality_uv_fid",
         texture_confidence: "texture_confidence", alignment_stats: "pose_alignment_stats",
         image_metrics: "render_vs_photo_metrics"}


def skip_all(reason: str) -> list[dict]:
    return [result(NAMES[fn], "SKIPPED", "-", reason=reason) for fn in EVALUATIONS]


def run_all(ctx: dict) -> list[dict]:
    out = []
    for fn in EVALUATIONS:
        try:
            out.append(fn(ctx))
        except Exception as e:  # one broken evaluation must not stop the others
            out.append(result(NAMES.get(fn, fn.__name__), "FAILED", fn.__module__,
                              reason=f"{type(e).__name__}: {e}"))
    return out
