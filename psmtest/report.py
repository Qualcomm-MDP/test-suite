"""summary.json -> report.md"""

from __future__ import annotations


def _f(v, fmt="{:.3f}"):
    if v is None:
        return "unavailable"
    if isinstance(v, float):
        return fmt.format(v)
    return str(v)


def _devices(d: dict | None) -> str:
    if not d:
        return "not recorded (older summary)"
    if d.get("status") != "confirmed":
        return f"not observed — {d.get('note') or d.get('status')}"
    parts = []
    for comp, o in d["observed"].items():
        fb = f" (fell back from {o['fell_back_from']})" if o.get("fell_back_from") else ""
        parts.append(f"{comp}: {o['device']}{fb}")
    return "confirmed from PSM's log — " + ", ".join(parts)


def _downloads(d: dict | None) -> str:
    if not d:
        return "not recorded (older summary)"
    what = ", ".join((d.get("new_cache_models") or []) + (d.get("log_evidence") or [])[:3])
    return f"{d['status']}{' (' + what + ')' if what else ''} — {d.get('note')}"


PHOTO_STATUS = {"all_downloaded": "all downloaded during this run",
                "all_cached": "all cached (read from disk, none downloaded)",
                "partly_cached": "partly cached, partly downloaded", "none": "no photos used",
                "unknown": "not determined"}


def _photos(ph: dict) -> str:
    if ph.get("total") is None:
        return f"{PHOTO_STATUS.get(ph.get('status'), ph.get('status'))} — {ph.get('note')}"
    head = (f"{ph['total']} used, {PHOTO_STATUS.get(ph['status'], ph['status'])} "
            f"({ph.get('cached', 0)} cached, {ph.get('downloaded_by_alignment', 0)} by the alignment, "
            f"{ph.get('downloaded_in_texturing', 0)} during texturing, {ph.get('failed', 0)} failed)")
    return head + (f". {ph['note']}." if ph.get("note") else ".")


def _network(net: dict) -> list[str]:
    if not net or not any(k in net for k in ("before", "after", "during")):
        return []
    L = ["## Network", ""]
    probes = [(k, net[k]) for k in ("before", "after") if net.get(k)]
    if probes:
        names = list(probes[0][1]["latency"])
        L += ["| | " + " | ".join(k for k, _ in probes) + " |", "|---|" + "--:|" * len(probes)]
        L.append("| download speed (Mbit/s, ~25 MB test) | " + " | ".join(
            _f(x["download"].get("mbps"), "{:.1f}") for _, x in probes) + " |")
        for n in names:
            L.append(f"| latency {n} (ms) | " + " | ".join(
                _f(x["latency"][n].get("ms"), "{:.0f}") if x["latency"][n].get("ok")
                else "unreachable" for _, x in probes) + " |")
        L += ["", f"Measured {', '.join(k + ' ' + x['measured_utc'] for k, x in probes)} (UTC).", ""]
    d = net.get("during") or {}
    if d.get("available"):
        r = d["run"]
        L += [f"During the run: {r['bytes_received'] / 1e6:.1f} MB received in {r['seconds']:.0f} s "
              f"(average {_f(r.get('mbps'), '{:.2f}')} Mbit/s).", ""]
        if d.get("stages"):
            L += ["| stage | MB received | Mbit/s |", "|---|--:|--:|"]
            L += [f"| {x['name']} | {_f(None if x['bytes_received'] is None else x['bytes_received'] / 1e6, '{:.1f}')} "
                  f"| {_f(x['mbps'], '{:.2f}')} |" for x in d["stages"]]
            L.append("")
        L += [f"- {d['note']}", ""]
    elif d:
        L += [f"Bytes received during the run: unavailable — {d.get('reason')}", ""]
    return L


def markdown(s: dict) -> str:
    m, loc, p, perf = s["machine"], s["location"], s["pipeline"], s["performance"]
    b = loc["bbox"]
    L = [f"# PSM test: {loc.get('name') or loc.get('query') or 'bbox'} on {m['label']}", "",
         f"**Overall: {s['status'].upper()}**  (run `{s['run_id']}`, {s['created_utc']})", "",
         "| | |", "|---|---|",
         f"| Area | {b['west']}, {b['south']}, {b['east']}, {b['north']} (W S E N), "
         f"{loc['area_km2']} km² — {loc['method']} |",
         f"| Machine | {m['cpu']}, {m['cpu_cores']} cores, {_f(m['ram_gb'], '{:.0f}')} GB RAM, "
         f"{m['os']} |",
         f"| Compute | {m['compute']}; ML device PSM would pick: {m.get('expected_ml_device') or 'unknown'} |",
         f"| ML device used | {_devices(p.get('ml_devices'))} |",
         f"| Model weights | {_downloads(p.get('model_downloads'))} |",
         f"| Software | PSM Python {m.get('pipeline_python') or 'unrecorded'}, torch "
         f"{m.get('torch_version') or (m.get('torch') or {}).get('torch') or 'unrecorded'}"
         f"{', CUDA build ' + m['torch_cuda_build'] if m.get('torch_cuda_build') else ''} |",
         f"| PSM | `{p['git'].get('short')}` ({p['git'].get('branch')}"
         f"{', dirty' if p['git'].get('dirty') else ''}"
         f"{', behind origin/main by ' + str(p['git']['behind_origin_main']) if p['git'].get('behind_origin_main') else ''}) |",
         f"| Options | `{' '.join(p['pipeline_args']) or '(PSM defaults)'}` |",
         f"| Pipeline | {p['status']} (exit {p['exit_code']}), wall {_f(p['wall_time_s'], '{:.1f}')} s"
         f"{', degraded: ' + ', '.join(p['degraded']) if p['degraded'] else ''} |",
         f"| Buildings left out of mesh | {p['flagged_buildings']} |",
         *([f"| Alignment | **reused** from `{p['alignment_reused']}` (no photo download, segmentation "
            f"or COLMAP this run: not comparable with a full run) |"] if p.get("alignment_reused") else []),
         ""]
    if p.get("error_lines"):
        L += ["**Errors**", "", "```", *p["error_lines"], "```", ""]
    L += ["## Performance (PSM profiler)", ""]
    if not perf.get("available"):
        L += [f"Unavailable: {perf.get('reason')}", ""]
    else:
        tm = perf.get("timing") or {}
        if tm.get("available"):
            d = tm["downloads_s"]
            L += [f"**Processing time {_f(tm['processing_s'], '{:.1f}')} s**"
                  + (f" ({perf['min_per_km2']:.1f} min/km²)" if perf.get("min_per_km2") else "")
                  + " — downloads and scoring left out"
                  + (" (except photo downloads overlapped with texturing, see Photos)"
                     if tm.get("processing_includes_photo_downloads") else "") + ".", "",
                  "| | s |", "|---|--:|",
                  f"| processing | {_f(tm['processing_s'], '{:.1f}')} |",
                  f"| downloads: OSM + Mapillary metadata (fetch) | {_f(d.get('fetch'), '{:.1f}')} |",
                  f"| downloads: alignment photos | {_f(d.get('alignment_photos'), '{:.1f}')} |",
                  f"| downloads: alignment OSM outlines | {_f(d.get('alignment_osm'), '{:.1f}')} |",
                  f"| scoring (PSM's evaluate_* stages) | {_f(tm['scoring_s'], '{:.1f}')} |",
                  f"| wall clock (everything) | {_f(tm['wall_clock_s'], '{:.1f}')} |", ""]
        else:
            L += [f"Wall clock {_f(perf['total_wall_time_s'], '{:.2f}')} s (downloads included; "
                  "no processing/download split recorded)", ""]
        ph = p.get("photos") or {}
        if ph:
            L += [f"**Photos:** {_photos(ph)}", ""]
        L += [f"Largest process RSS {_f(p.get('max_process_rss_mb'), '{:.0f}')} MB (OS).", "",
              "| stage | status | wall s | cpu s | py heap peak MB | CUDA Δ MB |", "|---|---|--:|--:|--:|--:|"]
        for st in perf["stages"]:
            L.append(f"| {st['name']} | {st['status']} | {_f(st['wall_time_s'], '{:.2f}')} | "
                     f"{_f(st['cpu_time_s'], '{:.2f}')} | {_f(st['py_heap_peak_mb'], '{:.1f}')} | "
                     f"{_f(st['cuda_alloc_delta_mb'], '{:.1f}')} |")
        L += ["", *[f"- {n}" for n in perf.get("notes", []) + (perf.get("timing") or {}).get("notes", [])], ""]
    L += _network(s.get("network") or {})
    L += ["## Evaluations", "", "| metric | status | source | result |", "|---|---|---|---|"]
    for e in s["evaluations"]:
        if e["status"] == "OK":
            key = {k: v for k, v in e["scores"].items() if v is not None and not isinstance(v, dict)}
            txt = ", ".join(f"{k}={_f(v)}" for k, v in list(key.items())[:8])
        else:
            txt = e.get("reason") or ""
        L.append(f"| {e['name']} | {e['status']} | `{e['source']}` | {txt} |")
    L.append("")
    geo = next((e for e in s["evaluations"] if e["name"] == "geometry_vs_lidar" and e["status"] == "OK"), None)
    if geo:
        g = geo["scores"]
        L += ["### Geometry vs. LiDAR (all evaluated buildings, metres, model − LiDAR)", "",
              "| metric | n | bias | median | RMSE | MAE | NMAD | ≤1 m | ≤3 m |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for k in ("M1_height_err", "M2_terrain_err", "M2_base_err", "M3_roof_err"):
            L.append("| " + " | ".join([k] + [_f(g.get(f"{k}_{st}")) for st in
                     ("n", "bias_mean", "median", "rmse", "mae", "nmad", "within_1m", "within_3m")]) + " |")
        L += ["", f"Building status: {g.get('building_status_counts')}", ""]
    cm = p.get("colmap") or {}
    if cm.get("used"):
        L += [f"COLMAP: {cm.get('path') or 'not found'} — version {cm.get('version') or 'unknown'}, "
              f"{'with' if cm.get('cuda') else 'without'} CUDA ({cm.get('source')})", ""]
    al = next((e for e in s["evaluations"] if e["name"] == "pose_alignment_stats" and e["status"] == "OK"), None)
    if al:
        x = al["scores"]
        L += ["### Pose alignment (COLMAP)", "",
              "| frames | aligned | reliable | aligned share | reliable share | legs aligned | pieces aligned |",
              "|--:|--:|--:|--:|--:|--:|--:|",
              f"| {_f(x.get('frames'))} | {_f(x.get('frames_aligned'))} | {_f(x.get('frames_reliable'))} | "
              f"{_f(x.get('aligned_share'))} | {_f(x.get('reliable_share'))} | "
              f"{_f(x.get('legs_aligned'))}/{_f(x.get('legs'))} | {_f(x.get('pieces_aligned'))}/{_f(x.get('pieces'))} |", ""]
        if x.get("wall_fit_median_m_after") is not None:
            L += [f"Points to OSM walls (median over pieces): {_f(x.get('wall_fit_median_m_before'), '{:.2f}')} m "
                  f"with Mapillary's poses -> {_f(x.get('wall_fit_median_m_after'), '{:.2f}')} m aligned; "
                  f"within 1 m: {_f(x.get('wall_within_1m_before'), '{:.2f}')} -> "
                  f"{_f(x.get('wall_within_1m_after'), '{:.2f}')}.", ""]
        steps = {k[6:]: v for k, v in x.items() if k.startswith("fit_s_")}
        if steps:
            L += ["Final fitting step (align.py) times, s: " + ", ".join(f"{k} {_f(v, '{:.1f}')}" for k, v in steps.items())
                  + ". The align_poses stage also includes the photo download, masking and COLMAP.", ""]
    tex = next((e for e in s["evaluations"] if e["name"] == "texture_quality_uv_fid" and e["status"] == "OK"), None)
    if tex:
        x = tex["scores"]
        L += ["### Texture quality (UV mapping; 0-1, lower O/D/F and higher confidence are better)", "",
              "| confidence | O obstruction | D distortion | F realism | FID | partial | device | seconds |",
              "|--:|--:|--:|--:|--:|---|---|--:|",
              "| " + " | ".join(_f(x.get(k)) for k in ("confidence", "O_obstruction", "D_distortion",
                                                      "F_realism", "fid", "partial", "device", "seconds")) + " |", ""]
        if x.get("partial"):
            L += [f"Partial score: F (realism) left out — {x.get('fid_reason') or 'no reason given'}.", ""]
    for e in s["evaluations"]:
        if e["status"] == "OK" and e.get("interpretation"):
            L += [f"- **{e['name']}** ({e['units']}): {e['interpretation']}"]
    return "\n".join(L) + "\n"
