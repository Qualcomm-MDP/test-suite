"""Run PSM's own CLI (src/main.py) and read what it produces: the PipelineProfiler JSON
(--profile), the .glb and the log. Nothing here measures stages itself."""

from __future__ import annotations

import os
import re
try:
    import resource  # not on Windows
except ImportError:
    resource = None
import subprocess
import sys
import time
from pathlib import Path

# Log lines PSM prints when a step falls back instead of failing the run.
DEGRADED = {
    "alignment_failed": r"align_poses: alignment failed",
    "colmap_missing": r"COLMAP not found",
    "captures_fallback": r"Choosing captures failed|No street has a dense enough capture",
    "terrain_failed": r"Terrain failed",
    "naip_failed": r"NAIP failed",
    "lama_cpu_fallback": r"LaMa on \S+ failed",
    "overpass_fallback": r"All Overpass endpoints failed",  # OSM fetched from the main OSM API

}
FLAGGED = re.compile(r"(\d+) building\(s\) were left out of the mesh")  # PSM >= 05e30e5

# Lines PSM prints when it actually places a model on a device (segmentation/model.py,
# segmentation/obstruction.py, segmentation/street_masks.py) and when LaMa falls back to the CPU.
DEVICE_LINES = [
    ("segmentation", re.compile(r"Loading segmentation model \((?P<model>[^)]+)\) on (?P<dev>[\w:]+)")),
    ("lama", re.compile(r"Loading LaMa inpainting model on (?P<dev>[\w:]+)")),
    ("street_masks", re.compile(r"^Model (?P<model>\S+) on (?P<dev>[\w:]+)\s*$", re.M)),
]
LAMA_FALLBACK = re.compile(r"LaMa on (?P<dev>[\w:]+) failed")
# Model-weight downloads that libraries print: torch.hub ('Downloading: "https://..." to ...') and
# huggingface_hub progress bars ('model.safetensors:  40%|'). Not Mapillary's "Downloading images".
DOWNLOAD_LINES = re.compile(r'Downloading: "(?P<url>[^"]+)"|'
                            r"(?P<file>[\w.-]+\.(?:safetensors|bin|pt|pth|ckpt|onnx)):\s+\d+%\|")


def git(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(repo: Path) -> dict:
    head = git(repo, "rev-parse", "HEAD")
    if head is None:
        return {"commit": None, "note": "not a git checkout"}
    behind = git(repo, "rev-list", "--count", "HEAD..origin/main")
    return {"commit": head, "short": head[:7], "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "subject": git(repo, "log", "-1", "--format=%s"),
            "dirty": bool(git(repo, "status", "--porcelain", "--untracked-files=no")),
            "origin_main": git(repo, "rev-parse", "--short", "origin/main"),
            "behind_origin_main": int(behind) if behind and behind.isdigit() else None}


def repo_snapshot(repo: Path) -> set[str] | None:
    out = git(repo, "status", "--porcelain", "--ignored")
    return None if out is None else set(out.splitlines())


def find_python(repo: Path) -> str | None:
    """Automatic discovery only: <repo>/.venv, then `poetry env info -e` (PATH or suite-local)."""
    for cand in (repo / ".venv/bin/python", repo / ".venv/Scripts/python.exe"):
        if cand.exists():
            return str(cand)
    local_poetry = Path(__file__).resolve().parents[1] / ".tools" / "poetry" / "bin" / "poetry"
    for poetry in ("poetry", str(local_poetry)):
        try:
            env = subprocess.run([poetry, "-C", str(repo), "env", "info", "-e"], capture_output=True,
                                 encoding="utf-8", errors="replace", timeout=30).stdout.strip()
            if env and Path(env).exists():
                return env
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def utf8_env(env: dict) -> dict:
    """Child Pythons write UTF-8 to pipes. Without this, Windows uses the ANSI code page
    (cp1252) when output is piped, and PSM dies printing '\u2713' (common/ProgressMonitor.py)."""
    return {**env, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def _echo(line: str) -> None:
    """Print to the suite's own terminal, which may not be UTF-8 (Windows console code page)."""
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        sys.stdout.write(line.encode(enc, errors="replace").decode(enc))


def main_supports(repo: Path, flag: str) -> bool:
    """Whether this checkout's src/main.py defines `flag` (older PSM exits on unknown options)."""
    try:
        return f'"{flag}"' in (repo / "src" / "main.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def env_file_value(repo: Path, key: str) -> str | None:
    """KEY=value from <PSM>/.env (PSM loads it with python-dotenv), unquoted; None if absent."""
    env = repo / ".env"
    if not env.exists():
        return None
    m = re.search(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=\s*(.*?)\s*$",
                  env.read_text(encoding="utf-8", errors="replace"), re.M)
    if not m:
        return None
    return m.group(1).strip("\"'") or None


MIN_COLMAP = (4, 2)  # PSM's reconstruct.py uses COLMAP 4 pose priors / sensor types (README: 4.2+)


def colmap_info(exe: str | None) -> dict:
    """Version and CUDA support from `colmap help`'s first line, e.g.
    'COLMAP 4.2.1 (Commit Unknown on Unknown with CUDA, std hash maps)'."""
    if not exe:
        return {"path": None, "version": None, "cuda": None, "version_ok": False}
    try:
        r = subprocess.run([exe, "help"], capture_output=True, encoding="utf-8", errors="replace",
                           timeout=60)
        first = next((l for l in (r.stdout + r.stderr).splitlines() if l.startswith("COLMAP")), "")
    except (OSError, subprocess.SubprocessError) as e:
        return {"path": exe, "version": None, "cuda": None, "version_ok": False, "error": str(e)}
    m = re.match(r"COLMAP (\d+)\.(\d+)(?:\.(\d+))?", first)
    ver = tuple(int(x) for x in m.groups() if x is not None) if m else None
    return {"path": exe, "version": ".".join(map(str, ver)) if ver else None,
            "cuda": ("with CUDA" in first) if first else None,
            "version_ok": bool(ver and ver[:2] >= MIN_COLMAP), "banner": first or None}


def has_token(repo: Path) -> bool:
    if os.environ.get("MAPILLARY_ACCESS_TOKEN"):
        return True
    env = repo / ".env"
    return env.exists() and bool(re.search(r"^\s*MAPILLARY_ACCESS_TOKEN\s*=\s*\S", env.read_text(encoding="utf-8", errors="replace"), re.M))


def command(python: str, repo: Path, bbox: dict, art: Path, extra: list[str]) -> list[str]:
    return [python, str(repo / "src" / "main.py"),
            "--bbox", *(str(bbox[k]) for k in ("west", "south", "east", "north")),
            "--output", str(art / "model.glb"), "--profile", str(art / "profile.json"),
            "--dump-mapillary", str(art / "mapillary.json"), *extra]


def run(cmd: list[str], cwd: Path, env: dict, log: Path) -> dict:
    """Run cmd, copying its output to log and the terminal."""
    rusage = (lambda: resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) if resource else (lambda: 0)
    before = rusage()
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        with subprocess.Popen(cmd, cwd=cwd, env=utf8_env(env), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding="utf-8", errors="replace") as p:
            for line in p.stdout:
                f.write(line)
                _echo(line)
            rc = p.wait()
    wall = time.perf_counter() - t0
    peak = rusage()
    unit = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    return {"exit_code": rc, "wall_time_s": round(wall, 2),
            # ru_maxrss of children: the largest single process, valid only if it grew
            "max_process_rss_mb": round(peak * unit / 2**20, 1) if peak > before else None}


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_profile(path: Path) -> dict:
    """PSM PipelineProfiler.summary() -> the suite's performance block. Values the profiler
    marks "N/A" (e.g. GPU fields without CUDA) become None."""
    import json

    if not path.exists():
        return {"available": False,
                "reason": "no profile JSON (PSM writes it only after a successful run)"}
    p = json.loads(path.read_text(encoding="utf-8"))

    def cuda(snap) -> dict:  # profiler's gpu_start/gpu_end: allocated, reserved, free (CUDA only)
        snap = snap if isinstance(snap, dict) else {}
        return {f"{k}_mb": _num(snap.get(f"{k}_mb")) for k in ("allocated", "reserved", "free")}

    stages = [{"name": s.get("name"), "status": s.get("status"),
               "wall_time_s": _num(s.get("wall_time_s")), "cpu_time_s": _num(s.get("cpu_time_s")),
               "py_heap_peak_mb": _num(s.get("peak_memory_mb")),
               "py_heap_delta_mb": _num(s.get("memory_delta_mb")),
               "cuda_alloc_delta_mb": _num(s.get("gpu_delta_allocated_mb")),
               "cuda_start": cuda(s.get("gpu_start")), "cuda_end": cuda(s.get("gpu_end")),
               "throughput": s.get("throughput") or {}, "error": s.get("error")}
              for s in p.get("stages", [])]
    return {"available": True, "source": "PSM common/profiler.py PipelineProfiler (--profile)",
            "total_wall_time_s": _num(p.get("total_wall_time_s")),
            "total_cpu_time_s": round(sum(s["cpu_time_s"] or 0 for s in stages), 3) if stages else None,
            "started_at": p.get("started_at"), "profiler_system": p.get("system"), "stages": stages,
            "notes": ["py_heap_* is tracemalloc (Python heap only: no torch/OpenCV buffers, no "
                      "subprocesses)", "cuda_* are CUDA only (start/end snapshots, no peak); MPS "
                      "use is not recorded by PSM's profiler",
                      "cpu_time_s and cuda_* cover PSM's own process only: with --poses colmap the "
                      "alignment (feature matching, COLMAP) runs in subprocesses, so align_poses "
                      "shows far less CPU time and GPU memory than it really used"]}


def ml_devices(text: str, expected: str | None) -> dict:
    """Devices PSM *reported* placing models on (confirmed), kept apart from the device
    select_device() would pick before the run (expected)."""
    observed = {}
    for comp, pat in DEVICE_LINES:
        m = pat.search(text)
        if m:
            observed[comp] = {"device": m.group("dev"), "model": m.groupdict().get("model"),
                              "evidence": m.group(0).strip()}
    fb = LAMA_FALLBACK.search(text)
    if fb and "lama" in observed:
        observed["lama"].update(device="cpu", fell_back_from=fb.group("dev"),
                                evidence=observed["lama"]["evidence"] + " | " + fb.group(0))
    used = sorted({o["device"] for o in observed.values()})
    return {"expected": expected, "observed": observed, "used": used,
            "status": "confirmed" if observed else "not_observed",
            "note": None if observed else "no model-loading line in the log: either no ML model "
                    "ran (e.g. --no-seg) or PSM did not report its device"}


def model_cache_dirs(env: dict) -> dict[str, Path]:
    """Where huggingface_hub and torch.hub keep weights, honouring their environment variables."""
    home = Path.home()
    xdg = Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME") else home / ".cache"
    hf = (Path(env["HF_HUB_CACHE"]) if env.get("HF_HUB_CACHE")
          else Path(env["HF_HOME"]) / "hub" if env.get("HF_HOME") else xdg / "huggingface" / "hub")
    th = Path(env["TORCH_HOME"]) if env.get("TORCH_HOME") else xdg / "torch"
    return {"huggingface": hf, "torch_hub": th / "hub" / "checkpoints"}


def cache_snapshot(env: dict) -> dict | None:
    """Weight files in the model caches (a missing cache folder counts as empty). None if a
    cache folder exists but cannot be read."""
    files = set()
    try:
        for name, d in model_cache_dirs(env).items():
            if not d.exists():
                continue
            if name == "huggingface":
                for model in d.glob("models--*"):
                    files.update(f"{name}/{model.name}/{b.name}" for b in (model / "blobs").glob("*")
                                 if not b.name.endswith(".incomplete"))
            else:
                files.update(f"{name}/{f.name}" for f in d.iterdir() if f.is_file())
    except OSError:
        return None
    return {"dirs": {k: str(v) for k, v in model_cache_dirs(env).items()}, "files": files}


def model_downloads(text: str, before: dict | None, after: dict | None) -> dict:
    """status: 'downloaded'      confirmed (a download line in the log, or new weight files in a cache)
               'not_downloaded'  both cache snapshots taken, nothing new, no download line
               'unknown'         no snapshot and no download line: cache state cannot be told"""
    log_hits = []
    for m in DOWNLOAD_LINES.finditer(text):
        hit = m.group("url") or m.group("file")
        if hit not in log_hits:
            log_hits.append(hit)
    new = sorted(after["files"] - before["files"]) if before and after else None
    if log_hits or new:
        status = "downloaded"
    elif new is not None:
        status = "not_downloaded"
    else:
        status = "unknown"
    models = sorted({f.split("/")[1] for f in new or [] if f.startswith("huggingface/")})
    return {"status": status, "log_evidence": log_hits[:10],
            "new_cache_files": None if new is None else len(new),
            "new_cache_models": models + [f.split("/", 1)[1] for f in new or [] if f.startswith("torch_hub/")],
            "cache_dirs": (after or before or {}).get("dirs"),
            "note": "timings include one-time weight downloads (cold start)" if status == "downloaded"
                    else "weights loaded from the cache (warm start)" if status == "not_downloaded"
                    else "cache state not recorded"}


def scan_log(text: str) -> dict:
    m = FLAGGED.search(text)
    tail = [l for l in text.splitlines() if l.strip()][-25:]
    err = [l for l in tail if re.search(r"Error|Exception|Traceback|failed", l)]
    return {"degraded": [k for k, pat in DEGRADED.items() if re.search(pat, text)],
            # PSM before 05e30e5 never prints this line, so 0 there means "not reported"
            "flagged_buildings": int(m.group(1)) if m else 0,
            "error_lines": err[-10:], "log_tail": tail}


def classify(exit_code: int, glb_exists: bool, profile: dict, degraded: list[str]) -> str:
    stage_errors = [s for s in profile.get("stages", []) if s["status"] not in ("ok", "skipped")]
    if exit_code == 0 and glb_exists and not stage_errors:
        return "completed_degraded" if degraded else "completed"
    if glb_exists or any(s["status"] == "ok" for s in profile.get("stages", [])):
        return "partial"
    return "failed"


# ------------------------------------------------------------------ photos: downloaded or cached

# Texturing's photo loop (texturing/tex_projection.py): a tqdm bar "Downloading images: n/N" over
# every photo it uses, "Reused from the pose alignment: N photos, ..." for photos read from the
# alignment's store instead of downloaded, and "Download failed [id]" per photo it could not get.
# The alignment's own timing report (pose_alignment/run_pipeline.py) has "download N frames  T s"
# and its fit report "align time: osm_buildings T s, ...".
PHOTOS_TOTAL = re.compile(r"Downloading images:\s+\d+%\|[^|\n]*\|\s*\d+/(\d+)")
PHOTOS_REUSED = re.compile(r"Reused from the pose alignment: (\d+) photos")
PHOTO_FAILED = re.compile(r"Download failed \[")
ALIGN_DOWNLOAD = re.compile(r"^\s*download (\d+) frames\s+([\d.]+) s\s*$", re.M)
ALIGN_OSM = re.compile(r"align time: osm_buildings ([\d.]+) s")


def photo_sources(text: str, pipeline_args: list[str]) -> dict:
    """Where this run's street photos came from: downloaded during the run (by the alignment, or
    by texturing while it processes), or cached on disk from an earlier run (an alignment folder
    reused with --alignment-dir). PSM keeps no photo cache otherwise: raw-pose runs download every
    photo every time."""
    args = list(pipeline_args)
    align_dir = args[args.index("--alignment-dir") + 1] if "--alignment-dir" in args[:-1] else None
    poses = args[args.index("--poses") + 1] if "--poses" in args[:-1] else "raw"
    m = PHOTOS_TOTAL.search(text)
    out = {"poses": poses, "alignment_reused_from": align_dir}
    if not m:
        return {**out, "status": "unknown", "total": None,
                "note": "texturing's photo count is not in the log (texturing did not run, or an "
                        "older PSM log format)"}
    total = int(m.group(1))
    reused = sum(int(x) for x in PHOTOS_REUSED.findall(text))
    failed = len(PHOTO_FAILED.findall(text))
    dl = [(int(n), float(s)) for n, s in ALIGN_DOWNLOAD.findall(text) if int(n) > 0]
    align_n, align_s = (sum(n for n, _ in dl), round(sum(s for _, s in dl), 2)) if dl else (0, None)
    # photos texturing read from the alignment's store came from an earlier run when the alignment
    # folder was reused and the alignment downloaded nothing this time; otherwise from this run's
    cached = reused if align_dir and not align_n else 0
    in_texturing = max(total - reused - failed, 0)
    downloaded = (reused - cached) + in_texturing
    status = ("none" if not total else "all_cached" if cached and not downloaded
              else "partly_cached" if cached else "all_downloaded" if downloaded else "unknown")
    notes = []
    if cached:
        notes.append(f"{cached} photos read from the reused alignment folder {align_dir} (downloaded "
                     "by an earlier run)")
    if reused - cached:
        notes.append(f"{reused - cached} photos downloaded by this run's alignment"
                     + (f" ({align_n} frames in {align_s} s, timed by PSM)" if align_s is not None else ""))
    if in_texturing:
        notes.append(f"{in_texturing} photos downloaded during texturing, in parallel with its "
                     "processing: that download time is inside texturing_s and can't be separated")
    if failed:
        notes.append(f"{failed} photos could not be downloaded")
    return {**out, "status": status, "total": total, "cached": cached,
            "downloaded_by_alignment": reused - cached, "downloaded_in_texturing": in_texturing,
            "failed": failed, "alignment_download_frames": align_n or None,
            "alignment_download_s": align_s,
            "download_overlaps_processing": in_texturing > 0, "note": "; ".join(notes)}


def timing(perf: dict, photos: dict, text: str) -> dict:
    """Split the run's time into processing (what the machine does to build the model), the
    downloads PSM times on their own, and scoring. Processing never includes those downloads."""
    stages = {s["name"]: s.get("wall_time_s") or 0.0 for s in perf.get("stages", [])}
    total = perf.get("total_wall_time_s")
    if total is None:
        return {"available": False, "reason": "no profile"}
    osm = ALIGN_OSM.findall(text)
    downloads = {"fetch": round(stages.get("fetch", 0.0), 3),  # OSM buildings + Mapillary metadata
                 "alignment_photos": photos.get("alignment_download_s") or 0.0,
                 "alignment_osm": round(sum(float(x) for x in osm), 3) if osm else 0.0}
    scoring = round(sum(stages.get(k, 0.0) for k in ("evaluate_geometry", "evaluate_texture")), 3)
    processing = round(total - sum(downloads.values()) - scoring, 3)
    notes = ["processing_s = profiler total - downloads_s - scoring_s",
             "terrain and naip download their tiles inside those stages (a few seconds); they stay "
             "in processing_s"]
    if photos.get("download_overlaps_processing"):
        notes.append("includes photo downloads done in parallel with texturing (see photos); PSM "
                     "does not time them separately")
    return {"available": True, "processing_s": processing, "downloads_s": downloads,
            "download_total_s": round(sum(downloads.values()), 3), "scoring_s": scoring,
            "wall_clock_s": total,
            "processing_includes_photo_downloads": bool(photos.get("download_overlaps_processing")),
            "notes": notes}
