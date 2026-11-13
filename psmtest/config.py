"""config.toml -> one settings dict, with CLI/env overrides, and the PSM CLI arguments it means.

Precedence everywhere: command line > environment variable > config file > automatic discovery.
Relative paths in the config file are resolved against the config file's folder."""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path

from . import pipeline

SUITE_DIR = Path(__file__).resolve().parents[1]
POSES = ("raw", "computed", "colmap")
TEXTURES = ("blend", "best")

DEFAULTS = {
    "psm": {"dir": "", "python": "", "colmap": ""},
    "region": {"name": "", "bbox": [], "location": "", "center": [], "size_km": 0.0,
               "max_area_km2": 2.0},
    # Each key maps to a real src/main.py option (PSM 767fbaa); see pipeline_args().
    "pipeline": {"terrain": True, "segmentation": True, "fill_obstructions": True, "naip": False,
                 "texture": "blend", "poses": "raw", "aligned_only": False, "alignment_dir": "",
                 "min_quality": None, "evaluate_texture": True, "extra_args": []},
    # network_test: latency to PSM's services + a ~25 MB speed test, before and after the run
    "test": {"results_dir": "results", "machine_label": "", "run_evaluations": True,
             "network_test": True},
    "evaluation": {"geometry_ref": "origin/feat/geometry-eval", "laz": [], "timeout_s": 3600},
}


def load(path: str | Path | None) -> tuple[dict, Path]:
    """Defaults merged with the file (if any). Returns (settings, folder relative paths use)."""
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    base = SUITE_DIR
    if path:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"config file not found: {path}")
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        base = path.parent
        for section, values in data.items():
            if section not in cfg:
                raise ValueError(f"{path.name}: unknown section [{section}]")
            for k, v in values.items():
                if k not in cfg[section]:
                    raise ValueError(f"{path.name}: unknown key {section}.{k}")
                cfg[section][k] = v
    validate(cfg)
    return cfg, base


LOCAL_CONFIG = SUITE_DIR / "config.toml"
BASELINE = SUITE_DIR / "configs" / "baseline.toml"
# Settings that describe the machine, not the test: a shared config (e.g. the baseline) leaves
# them empty and each machine's own config.toml fills them in.
MACHINE_KEYS = (("psm", "dir"), ("psm", "python"), ("psm", "colmap"), ("test", "machine_label"))


def layer_machine(cfg: dict, config_file: str | Path | None, local: Path | None = None) -> str | None:
    """Fill the machine settings that `cfg` leaves empty from the local config.toml, when `cfg`
    came from a different file. Relative paths are made absolute against the local file's folder,
    so they mean the same as in that file. Returns the local file used, or None."""
    local = Path(local or LOCAL_CONFIG).expanduser().resolve()
    if not config_file or not local.exists() or Path(config_file).expanduser().resolve() == local:
        return None
    mine, mine_base = load(local)
    used = False
    for section, key in MACHINE_KEYS:
        value = mine[section][key]
        if cfg[section][key] or not value:
            continue
        if section == "psm":
            value = str(_path(value, mine_base) if key == "dir" else _interpreter(value, mine_base))
        cfg[section][key] = value
        used = True
    return str(local) if used else None


def validate(cfg: dict) -> None:
    p, r = cfg["pipeline"], cfg["region"]
    if p["poses"] not in POSES:
        raise ValueError(f"pipeline.poses must be one of {POSES}, got {p['poses']!r}")
    if p["texture"] not in TEXTURES:
        raise ValueError(f"pipeline.texture must be one of {TEXTURES}, got {p['texture']!r}")
    if p["min_quality"] is not None and not 0 <= float(p["min_quality"]) <= 1:
        raise ValueError("pipeline.min_quality must be between 0 and 1")
    if (p["aligned_only"] or p["alignment_dir"]) and p["poses"] != "colmap":
        raise ValueError("pipeline.aligned_only / alignment_dir need poses = \"colmap\"")
    if not isinstance(p["evaluate_texture"], bool):
        raise ValueError("pipeline.evaluate_texture must be true or false")
    if not isinstance(p["extra_args"], list):
        raise ValueError("pipeline.extra_args must be a list of strings")
    if r["bbox"] and len(r["bbox"]) != 4:
        raise ValueError("region.bbox needs 4 numbers: [west, south, east, north]")
    if sum(bool(x) for x in (r["bbox"], r["location"], r["center"])) > 1:
        raise ValueError("set only one of region.bbox, region.location, region.center")


def pipeline_args(p: dict) -> list[str]:
    """Settings -> src/main.py arguments (only flags main.py defines)."""
    args = []
    if p["terrain"] or p["naip"]:
        args.append("--terrain")
    if p["naip"]:
        args.append("--naip")
    if not p["segmentation"]:
        args.append("--no-seg")
    elif not p["fill_obstructions"]:
        args.append("--no-fill")
    if p["texture"] != "blend":
        args += ["--texture", p["texture"]]
    if p["poses"] != "raw":
        args += ["--poses", p["poses"]]
    if p["aligned_only"]:
        args.append("--aligned-only")
    if p["alignment_dir"]:
        args += ["--alignment-dir", str(p["alignment_dir"])]
    if p["min_quality"] is not None:
        args += ["--min-quality", str(p["min_quality"])]
    if p["evaluate_texture"]:  # PSM >= 9f5dbb6; run_test drops it for older checkouts
        args.append("--evaluate-texture")
    return args + [str(a) for a in p["extra_args"]]


def _path(value: str, base: Path) -> Path:
    q = Path(os.path.expandvars(value)).expanduser()
    return (q if q.is_absolute() else base / q).resolve()


def resolve_psm(cli: str | None, cfg: dict, base: Path) -> tuple[Path, str]:
    """PSM checkout: --psm-dir > $PSM_REPO > psm.dir > next to the suite. Must hold src/main.py."""
    tried = []
    for source, value, rel in (("--psm-dir", cli, Path.cwd()), ("PSM_REPO", os.environ.get("PSM_REPO"), Path.cwd()),
                               ("config psm.dir", cfg["psm"]["dir"], base)):
        if value:
            repo = _path(value, rel)
            if not (repo / "src" / "main.py").exists():
                raise ValueError(f"{source} = {repo} is not a PSM checkout (no src/main.py)")
            return repo, source
    for cand in (SUITE_DIR.parent / "PSM", SUITE_DIR.parent / "qualcomm" / "PSM", SUITE_DIR / "PSM"):
        tried.append(str(cand))
        if (cand / "src" / "main.py").exists():
            return cand.resolve(), "auto-discovered"
    raise ValueError("PSM checkout not found. Set psm.dir in config.toml, $PSM_REPO or --psm-dir. "
                     f"Looked in: {', '.join(tried)}")


def _interpreter(value: str, base: Path) -> Path:
    """Like _path, but symlinks are NOT followed: a virtualenv's bin/python is a symlink to the
    base interpreter, and only the symlink's own path activates the environment's packages."""
    q = Path(os.path.expandvars(value)).expanduser()
    return Path(os.path.abspath(q if q.is_absolute() else base / q))


def resolve_colmap(cfg: dict, base: Path, repo: Path) -> tuple[str | None, str]:
    """COLMAP executable for --poses colmap: psm.colmap > $COLMAP_BIN > COLMAP_BIN in <PSM>/.env >
    `colmap` on PATH. Symlinks are kept (a conda env's bin/colmap finds its libraries itself)."""
    for source, value, rel in (("config psm.colmap", cfg["psm"]["colmap"], base),
                               ("COLMAP_BIN", os.environ.get("COLMAP_BIN"), Path.cwd()),
                               ("<PSM>/.env COLMAP_BIN", pipeline.env_file_value(repo, "COLMAP_BIN"), repo)):
        if value:
            exe = _interpreter(value, rel)
            if not exe.exists():
                raise ValueError(f"{source} = {exe} does not exist")
            return str(exe), source
    found = shutil.which("colmap")
    return (found, "PATH") if found else (None, "not found")


def resolve_python(cli: str | None, cfg: dict, base: Path, repo: Path) -> tuple[str | None, str]:
    """--python > $PSM_PYTHON > psm.python > <repo>/.venv or Poetry's environment."""
    for source, value, rel in (("--python", cli, Path.cwd()), ("PSM_PYTHON", os.environ.get("PSM_PYTHON"), Path.cwd()),
                               ("config psm.python", cfg["psm"]["python"], base)):
        if value:
            py = _interpreter(value, rel)
            if not py.exists():
                raise ValueError(f"{source} = {py} does not exist")
            return str(py), source
    py = pipeline.find_python(repo)
    return py, "auto-discovered" if py else "not found"
