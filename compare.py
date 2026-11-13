#!/usr/bin/env python3
"""One table across runs (machines, areas, commits, settings).

  python3 compare.py results/*/summary.json [--md compare.md] [--csv compare.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path


EVAL_STAGES = ("evaluate_geometry", "evaluate_texture")  # PSM >= 9f5dbb6 profiler stages
UNRECORDED = "unrecorded"  # the field did not exist in that summary's schema


def _max(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return max(vals) if vals else None


def _timing(perf, p, poses, total, fetch, stages):
    """(processing_s, download_s, photos, processing_has_photo_downloads). Summaries before the
    timing split: a raw-pose run's only separate download is fetch (PSM never caches photos and
    downloads them inside texturing), so it can be derived; a COLMAP run's alignment photo
    download was not recorded."""
    tm = perf.get("timing") or {}
    if tm.get("available"):
        ph = p.get("photos") or {}
        return (tm["processing_s"], tm["download_total_s"], ph.get("status"),
                "yes" if tm.get("processing_includes_photo_downloads") else "no")
    if total is None or fetch is None:
        return None, None, UNRECORDED, UNRECORDED
    if poses == "raw":
        scoring = sum(stages.get(k) or 0 for k in EVAL_STAGES)
        return round(total - fetch - scoring, 4), round(fetch, 4), "all_downloaded", "yes"
    return UNRECORDED, UNRECORDED, UNRECORDED, UNRECORDED


def row(s: dict) -> dict:
    """One flat row from a psm-test/1 or psm-test/2 summary. Newer fields missing from an older
    summary come from the older equivalent where one exists, else are 'unrecorded'."""
    perf, m, p, loc = s.get("performance", {}), s["machine"], s["pipeline"], s["location"]
    geo = next((e for e in s.get("evaluations", []) if e["name"] == "geometry_vs_lidar"), {})
    g = geo.get("scores", {})
    tex = next((e for e in s.get("evaluations", []) if e["name"] == "texture_quality_uv_fid"), {})
    tx = tex.get("scores", {})
    al = next((e for e in s.get("evaluations", []) if e["name"] == "pose_alignment_stats"), {})
    ax = al.get("scores", {})
    cm = p.get("colmap") or {}
    args = p["pipeline_args"]
    st = perf.get("stages", [])
    stages = {x["name"]: x.get("wall_time_s") for x in st}
    total, fetch = perf.get("total_wall_time_s"), stages.get("fetch")
    dev, dl = p.get("ml_devices"), p.get("model_downloads")
    b = loc["bbox"]
    poses = args[args.index("--poses") + 1] if "--poses" in args[:-1] else "raw"
    processing, download, photos, overlap = _timing(perf, p, poses, total, fetch, stages)
    net = s.get("network") or {}
    speed = lambda k: ((net.get(k) or {}).get("download") or {}).get("mbps") if net else UNRECORDED  # noqa: E731
    during = (net.get("during") or {}).get("run") or {}
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
            "poses": poses,
            "colmap": (f"{cm.get('version')}{'+cuda' if cm.get('cuda') else ''}" if cm.get("used") else None),
            "alignment_reused": p.get("alignment_reused") if "photos" in p else UNRECORDED,
            "degraded": " ".join(p.get("degraded") or []) or None,
            # the headline time: never includes the downloads PSM times or the scoring stages
            "processing_s": processing, "download_s": download, "wall_clock_s": total,
            "photos": photos, "processing_has_photo_downloads": overlap,
            "net_mbps_before": speed("before"), "net_mbps_after": speed("after"),
            "run_rx_mb": round(during["bytes_received"] / 1e6, 1) if during else (None if net else UNRECORDED),
            "cpu_s": perf.get("total_cpu_time_s"), "min/km2": perf.get("min_per_km2"),
            **{f"{k}_s": stages.get(k) for k in ("fetch", "align_poses", "terrain", "build_mesh", "texturing",
                                                  "export", "evaluate_texture")},
            "max_rss_mb": p.get("max_process_rss_mb"),
            "py_heap_peak_mb": _max(x.get("py_heap_peak_mb") for x in st),
            "cuda_reserved_max_mb": _max((x.get("cuda_end") or {}).get("reserved_mb") for x in st),
            "geom": geo.get("status"),
            "M1_rmse_m": g.get("M1_height_err_rmse"), "M1_bias_m": g.get("M1_height_err_bias_mean"),
            "M2_rmse_m": g.get("M2_terrain_err_rmse"), "M2_base_rmse_m": g.get("M2_base_err_rmse"),
            "M3_rmse_m": g.get("M3_roof_err_rmse"), "height_within_3m": g.get("height_share_within_tol"),
            "terrain_QR06a": g.get("terrain_QR06a"),
            "texture": tex.get("status"), "tex_confidence": tx.get("confidence"),
            "tex_partial": tx.get("partial"), "tex_O": tx.get("O_obstruction"),
            "tex_D": tx.get("D_distortion"), "tex_F": tx.get("F_realism"), "tex_fid": tx.get("fid"),
            "alignment": al.get("status"), "frames": ax.get("frames"),
            "aligned_share": ax.get("aligned_share"), "reliable_share": ax.get("reliable_share"),
            "wall_fit_before_m": ax.get("wall_fit_median_m_before"),
            "wall_fit_after_m": ax.get("wall_fit_median_m_after")}


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
    differ = [c for c in ("bbox", "psm", "suite", "options", "poses", "machine", "torch", "weights", "photos")
              if len({r[c] for r in rows}) > 1]
    if differ and len(rows) > 1:
        md += "\n\nThese differ between runs: " + ", ".join(differ)
    if len({r["poses"] for r in rows}) > 1:
        md += ("\n\nposes differ: with colmap, texturing reuses the alignment's photos and class maps, "
               "so that work moves from texturing_s into align_poses_s; compare processing_s, not stages.")
    md += ("\n\nprocessing_s leaves out downloads PSM times on their own (fetch: OSM + Mapillary metadata; "
           "the alignment's photo download) and scoring; wall_clock_s is everything.")
    if any(r["processing_has_photo_downloads"] == "yes" for r in rows):
        md += ("\n\nprocessing_has_photo_downloads=yes: photos were downloaded inside texturing, in parallel "
               "with its processing (raw poses), so processing_s still includes that download time; "
               "it depends on the network (see net_mbps_*).")
    if any(r["photos"] in ("all_cached", "partly_cached") for r in rows) or \
            any(r["alignment_reused"] not in (None, UNRECORDED) for r in rows):
        md += ("\n\nSome runs reused cached photos / an earlier alignment (photos, alignment_reused): they "
               "skip downloads and, with a reused alignment, segmentation and COLMAP. Don't compare their "
               "times with full runs.")
    if any(r["weights"] == "downloaded" for r in rows):
        md += ("\n\nRuns with weights=downloaded include one-time model downloads in their timings "
               "(cold start); compare them with care.")
    if any(r[c] == UNRECORDED for r in rows for c in ("ml_device_used", "weights", "torch_cuda", "photos",
                                                       "net_mbps_before", "processing_s")):
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
