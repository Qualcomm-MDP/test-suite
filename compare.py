#!/usr/bin/env python3
"""One table across runs (machines, areas, commits, settings).

  python3 compare.py results/*/summary.json [--md compare.md] [--csv compare.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path


UNRECORDED = "unrecorded"  # the field did not exist in that summary's schema


def _max(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return max(vals) if vals else None


def row(s: dict) -> dict:
    """One flat row from a psm-test/1 or psm-test/2 summary. Newer fields missing from an older
    summary come from the older equivalent where one exists, else are 'unrecorded'."""
    perf, m, p, loc = s.get("performance", {}), s["machine"], s["pipeline"], s["location"]
    geo = next((e for e in s.get("evaluations", []) if e["name"] == "geometry_vs_lidar"), {})
    g = geo.get("scores", {})
    st = perf.get("stages", [])
    stages = {x["name"]: x.get("wall_time_s") for x in st}
    total, fetch = perf.get("total_wall_time_s"), stages.get("fetch")
    dev, dl = p.get("ml_devices"), p.get("model_downloads")
    b = loc["bbox"]
    return {"run": s["run_id"], "schema": s.get("schema"), "machine": m["label"], "os": m.get("os"),
            "cpu": m["cpu"], "cores": m.get("cpu_cores"), "ram_gb": m.get("ram_gb"),
            "gpus": "; ".join(x.get("name", "") for x in m.get("gpus", [])) or None,
            "compute": m["compute"],
            "python": m.get("pipeline_python") or (perf.get("profiler_system") or {}).get("python"),
            "torch": m.get("torch_version") or (m.get("torch") or {}).get("torch"),
            "torch_cuda": m.get("torch_cuda_build") if "torch_cuda_build" in m else UNRECORDED,
            "ml_device_expected": m.get("expected_ml_device"),
            "ml_device_used": UNRECORDED if dev is None else ("/".join(dev.get("used") or []) or dev.get("status")),
            "weights": UNRECORDED if dl is None else dl.get("status"),
            "area": loc.get("name") or loc.get("query") or "bbox",
            "bbox": f"{b['west']},{b['south']},{b['east']},{b['north']}", "km2": loc["area_km2"],
            "psm": p["git"].get("short"), "suite": ((s.get("suite") or {}).get("commit") or "")[:7] or None,
            "options": " ".join(p["pipeline_args"]), "status": s["status"],
            "degraded": " ".join(p.get("degraded") or []) or None,
            "total_s": total,
            # fetch is network-bound; the rest is what the machine itself does
            "compute_s": round(total - fetch, 4) if total is not None and fetch is not None else None,
            "cpu_s": perf.get("total_cpu_time_s"), "min/km2": perf.get("min_per_km2"),
            **{f"{k}_s": stages.get(k) for k in ("fetch", "terrain", "build_mesh", "texturing", "export")},
            "max_rss_mb": p.get("max_process_rss_mb"),
            "py_heap_peak_mb": _max(x.get("py_heap_peak_mb") for x in st),
            "cuda_reserved_max_mb": _max((x.get("cuda_end") or {}).get("reserved_mb") for x in st),
            "geom": geo.get("status"),
            "M1_rmse_m": g.get("M1_height_err_rmse"), "M1_bias_m": g.get("M1_height_err_bias_mean"),
            "M2_rmse_m": g.get("M2_terrain_err_rmse"), "M2_base_rmse_m": g.get("M2_base_err_rmse"),
            "M3_rmse_m": g.get("M3_roof_err_rmse"), "height_within_3m": g.get("height_share_within_tol"),
            "terrain_QR06a": g.get("terrain_QR06a")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows = [row(json.loads(p.read_text(encoding="utf-8"))) for p in a.summaries]
    cols = list(rows[0])
    fmt = lambda v: "–" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    md = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                   + ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows])
    differ = [c for c in ("bbox", "psm", "suite", "options", "machine", "torch", "weights")
              if len({r[c] for r in rows}) > 1]
    if differ and len(rows) > 1:
        md += "\n\nThese differ between runs: " + ", ".join(differ)
    if any(r["weights"] == "downloaded" for r in rows):
        md += ("\n\nRuns with weights=downloaded include one-time model downloads in their timings "
               "(cold start); compare them with care.")
    if any(r[c] == UNRECORDED for r in rows for c in ("ml_device_used", "weights", "torch_cuda")):
        md += f"\n\n'{UNRECORDED}': the field did not exist in that run's schema (older summary)."
    try:
        print(md)
    except UnicodeEncodeError:  # a console that is not UTF-8 (Windows code page)
        enc = sys.stdout.encoding or "ascii"
        print(md.encode(enc, errors="replace").decode(enc))
    if a.md:
        a.md.write_text(md + "\n", encoding="utf-8")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
