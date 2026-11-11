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
                      "use is not recorded by PSM's profiler"]}


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
