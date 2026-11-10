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


def git(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
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
                                 text=True, timeout=30).stdout.strip()
            if env and Path(env).exists():
                return env
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def has_token(repo: Path) -> bool:
    if os.environ.get("MAPILLARY_ACCESS_TOKEN"):
        return True
    env = repo / ".env"
    return env.exists() and bool(re.search(r"^\s*MAPILLARY_ACCESS_TOKEN\s*=\s*\S", env.read_text(), re.M))


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
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace") as p:
            for line in p.stdout:
                f.write(line)
                sys.stdout.write(line)
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
    p = json.loads(path.read_text())
    stages = [{"name": s.get("name"), "status": s.get("status"),
               "wall_time_s": _num(s.get("wall_time_s")), "cpu_time_s": _num(s.get("cpu_time_s")),
               "py_heap_peak_mb": _num(s.get("peak_memory_mb")),
               "py_heap_delta_mb": _num(s.get("memory_delta_mb")),
               "cuda_alloc_delta_mb": _num(s.get("gpu_delta_allocated_mb")),
               "throughput": s.get("throughput") or {}, "error": s.get("error")}
              for s in p.get("stages", [])]
    return {"available": True, "source": "PSM common/profiler.py PipelineProfiler (--profile)",
            "total_wall_time_s": _num(p.get("total_wall_time_s")),
            "total_cpu_time_s": round(sum(s["cpu_time_s"] or 0 for s in stages), 3) if stages else None,
            "started_at": p.get("started_at"), "profiler_system": p.get("system"), "stages": stages,
            "notes": ["py_heap_* is tracemalloc (Python heap only: no torch/OpenCV buffers, no "
                      "subprocesses)", "cuda_alloc_delta_mb is CUDA only; MPS use is not recorded"]}


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
