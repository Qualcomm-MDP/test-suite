#!/usr/bin/env python3
"""One table across runs (machines, areas, commits, settings).

  python3 compare.py results/*/summary.json [--md compare.md] [--csv compare.csv]
"""

import argparse
import csv
import json
from pathlib import Path


def row(s: dict) -> dict:
    perf, m, p, loc = s["performance"], s["machine"], s["pipeline"], s["location"]
    geo = next((e for e in s["evaluations"] if e["name"] == "geometry_vs_lidar"), {})
    g = geo.get("scores", {})
    stages = {st["name"]: st["wall_time_s"] for st in perf.get("stages", [])}
    b = loc["bbox"]
    return {"run": s["run_id"], "machine": m["label"], "cpu": m["cpu"], "ram_gb": m.get("ram_gb"),
            "compute": m["compute"], "ml_device": m.get("expected_ml_device"),
            "area": loc.get("name") or loc.get("query") or "bbox",
            "bbox": f"{b['west']},{b['south']},{b['east']},{b['north']}", "km2": loc["area_km2"],
            "psm": p["git"].get("short"), "suite": (s.get("suite", {}).get("commit") or "")[:7] or None,
            "options": " ".join(p["pipeline_args"]), "status": s["status"],
            "total_s": perf.get("total_wall_time_s"), "min/km2": perf.get("min_per_km2"),
            **{f"{k}_s": stages.get(k) for k in ("fetch", "terrain", "build_mesh", "texturing", "export")},
            "max_rss_mb": p.get("max_process_rss_mb"), "geom": geo.get("status"),
            "M1_rmse_m": g.get("M1_height_err_rmse"), "M2_rmse_m": g.get("M2_terrain_err_rmse"),
            "M3_rmse_m": g.get("M3_roof_err_rmse"), "height_within_3m": g.get("height_share_within_tol"),
            "terrain_QR06a": g.get("terrain_QR06a")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows = [row(json.loads(p.read_text())) for p in a.summaries]
    cols = list(rows[0])
    fmt = lambda v: "–" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    md = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                   + ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows])
    differ = [c for c in ("bbox", "psm", "suite", "options", "machine") if len({r[c] for r in rows}) > 1]
    if differ and len(rows) > 1:
        md += "\n\nThese differ between runs: " + ", ".join(differ)
    print(md)
    if a.md:
        a.md.write_text(md + "\n")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
