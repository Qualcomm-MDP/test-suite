"""Machine description for cross-machine comparison. GPU *availability* comes from the host and
from torch in the pipeline's interpreter; which device the pipeline would pick is asked from
PSM's own segmentation.model.select_device. Actual GPU *use* is only known from the run
(the PSM profiler records CUDA allocation deltas; it has no MPS counter)."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: float = 15) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model() -> str:
    if sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"])) / 2**30, 1)
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 2**20, 1)
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("tot", "av", "tp", "ap", "tv", "av2", "ext")]
            m = MS(len=ctypes.sizeof(MS))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.tot / 2**30, 1)
    except (OSError, TypeError, ValueError, AttributeError):
        pass
    return None


def nvidia_gpus() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3:
            gpus.append({"vendor": "NVIDIA", "name": p[0], "memory_mb": float(p[1]), "driver": p[2]})
    return gpus


def is_laptop() -> bool | None:
    if sys.platform == "darwin":
        out = _run(["pmset", "-g", "batt"])
        return None if out is None else "InternalBattery" in out
    if sys.platform.startswith("linux"):
        return any(Path("/sys/class/power_supply").glob("BAT*"))
    return None


PROBE = r"""
import json, platform, sys
out = {"python": platform.python_version(), "python_implementation": platform.python_implementation()}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda_build"] = torch.version.cuda  # None for CPU/MPS builds
    out["cuda_available"] = torch.cuda.is_available()
    out["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    out["mps_available"] = bool(mps and mps.is_available())
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
try:
    sys.path.insert(0, sys.argv[1])
    from segmentation.model import select_device
    out["segmentation_device"] = str(select_device())
except Exception as e:
    out["segmentation_device_error"] = f"{type(e).__name__}: {e}"
print("PROBE" + json.dumps(out))
"""


def probe_pipeline_python(python: str | None, repo: Path) -> dict:
    if not python:
        return {"error": "no pipeline interpreter"}
    try:
        r = subprocess.run([python, "-c", PROBE, str(repo / "src")], capture_output=True, encoding="utf-8", errors="replace",
                           timeout=180, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                             "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    for line in r.stdout.splitlines():
        if line.startswith("PROBE"):
            return json.loads(line[5:])
    return {"error": (r.stderr or r.stdout).strip()[-500:]}


def collect(python: str | None, repo: Path, probe: bool = True) -> dict:
    cpu = cpu_model()
    nv = nvidia_gpus()
    gpus = list(nv)
    if sys.platform == "darwin" and platform.machine() == "arm64":
        gpus.append({"vendor": "Apple", "name": f"{cpu} integrated GPU"})
    torch = probe_pipeline_python(python, repo) if probe else {"skipped": True}
    usable = [b for b, k in (("CUDA", "cuda_available"), ("MPS", "mps_available")) if torch.get(k)]
    if usable:
        label = f"CPU + GPU ({'/'.join(usable)} usable by torch)"
    elif gpus:
        label = "CPU + GPU present (not usable by the pipeline's torch)"
    else:
        label = "CPU only"
    return {
        "label": os.environ.get("PSM_MACHINE_LABEL") or socket.gethostname().split(".")[0],
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu": cpu,
        "cpu_cores": os.cpu_count(),
        "ram_gb": ram_gb(),
        "laptop": is_laptop(),
        "gpus": gpus,
        "compute": label,
        "torch": torch,
        # Versions of PSM's interpreter (the one that runs the pipeline), not the suite's
        "pipeline_python": torch.get("python"),
        "torch_version": torch.get("torch"),
        "torch_cuda_build": torch.get("torch_cuda_build"),
        "suite_python": platform.python_version(),
        # What select_device() would pick; what was actually used is pipeline.ml_devices
        "expected_ml_device": torch.get("segmentation_device"),
    }
