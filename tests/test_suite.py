"""Offline checks of the suite itself, using a fake PSM checkout (no network, no pipeline env).

  python3 -m unittest discover -s tests -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_test  # noqa: E402
from psmtest import config, evals, hardware, location, pipeline  # noqa: E402

PROFILE = {"pipeline": "photorealistic-street-maps", "total_wall_time_s": 12.5,
           "system": {"gpu": {"available": False}},
           "stages": [{"name": "fetch", "status": "ok", "wall_time_s": 4.0, "cpu_time_s": 1.0,
                       "peak_memory_mb": 50.0, "memory_delta_mb": 2.0, "gpu_delta_allocated_mb": "N/A",
                       "throughput": {}, "error": None},
                      {"name": "terrain", "status": "skipped", "wall_time_s": 0.0, "cpu_time_s": 0.0,
                       "peak_memory_mb": 0.0, "memory_delta_mb": 0.0, "gpu_delta_allocated_mb": "N/A",
                       "throughput": {}, "error": None}]}

# Shape of PSM's data/alignment/summary.json (pose_alignment/align.py, PSM 9f5dbb6)
ALIGN_SUMMARY = {"pipeline": "Alignment_Pipeline",
                 "reconstruction": {"features": "aliked", "mapper": "global", "refine_intrinsics": True,
                                    "gravity_priors": True, "mask_moving": True},
                 "frames": 40, "leg_links": [], "frames_aligned": 30, "frames_reliable": 24,
                 "legs": [{"leg": 0, "folder": "seqA", "aligned": True, "registered": 30, "reliable": True},
                          {"leg": 1, "folder": "seqB", "aligned": False, "registered": 0, "reliable": False}],
                 "seconds": {"place_linked_legs": 1.25, "figures": 3.5, "total": 12.0}}
FAKE_COLMAP = "#!/bin/sh\necho 'COLMAP VERSION (Commit Unknown on Unknown CUDA_PART, std hash maps)'\necho 'Usage:'\n"


def fake_colmap(folder: Path, version="4.2.1", cuda=True) -> Path:
    exe = folder / "colmap"
    exe.write_text(FAKE_COLMAP.replace("VERSION", version).replace("CUDA_PART", "with CUDA" if cuda else "without CUDA"),
                   encoding="utf-8")
    exe.chmod(0o755)
    return exe


FAKE_MAIN = textwrap.dedent("""
    import argparse, json, os, shutil, sys
    ALIGN_SUMMARY_JSON = ALIGN_REPR
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4); ap.add_argument("--output"); ap.add_argument("--profile")
    ap.add_argument("--dump-mapillary"); ap.add_argument("--terrain", action="store_true")
    ap.add_argument("--poses", default="raw")
    a = ap.parse_args()
    if a.poses == "colmap":  # PSM's align_poses: needs COLMAP_BIN (or colmap on PATH)
        exe = os.environ.get("COLMAP_BIN")
        if exe and os.path.exists(exe):
            root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "outputs", "alignment", "mapillary_roi_1", "data", "alignment")
            os.makedirs(os.path.join(root, "leg0_seqA"))
            json.dump(ALIGN, open(os.path.join(root, "summary.json"), "w", encoding="utf-8"))
            json.dump({"aligned": True, "registered": 30}, open(os.path.join(root, "leg0_seqA", "stats.json"),
                                                                 "w", encoding="utf-8"))
        else:
            print("align_poses: alignment failed (COLMAP not found: install it, or set COLMAP_BIN in "
                  ".env to its executable); every image keeps Mapillary's computed pose")
    print("fetch ...")
    print("\u2713 Completed: fetch (1/7)")  # PSM common/ProgressMonitor.py prints this mark
    print("Downloading images:  50%|#####     | 1/2 [00:01<00:01,  1.0it/s]")  # Mapillary, not weights
    if os.environ.get("FAKE_FAIL"):
        print("RuntimeError: No images found."); sys.exit(1)
    print("Terrain failed: offline")
    if os.environ.get("FAKE_SEG"):
        print("Loading segmentation model (facebook/fake-model) on cpu …")
    if os.environ.get("FAKE_DOWNLOAD"):  # a new weight file appears in torch.hub's cache
        d = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "fake-lama.pt"), "wb").write(b"w")
    open(a.output, "wb").write(b"glTF")
    if getattr(a, "evaluate_texture", False):  # only when the test adds the option (newer PSM)
        mode = os.environ.get("FAKE_TEX", "ok")
        if mode == "ok":  # PSM writes texture_eval/ next to the .glb
            d = os.path.join(os.path.dirname(a.output), "texture_eval")
            os.makedirs(d)
            shutil.copy(os.environ["FAKE_TEX_SUMMARY"], os.path.join(d, "summary.json"))
            open(os.path.join(d, "per_group.csv"), "w", encoding="utf-8").write("group")
        elif mode == "unavailable":
            print("Texture evaluation not available: Missing package: torchvision. The model is kept.")
        elif mode == "failed":
            print("Texture evaluation failed: RuntimeError: CUDA out of memory. The model is kept.")
    json.dump(PROFILE, open(a.profile, "w", encoding="utf-8"))
""").replace("PROFILE", repr(PROFILE)).replace("ALIGN_REPR", repr(ALIGN_SUMMARY)).replace("json.dump(ALIGN,", "json.dump(ALIGN_SUMMARY_JSON,")
TEXTURE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "texture_summary.json"  # real PSM output


class Location(unittest.TestCase):
    def test_bbox_and_area(self):
        loc = location.resolve(bbox=[-83.742, 42.277, -83.730, 42.286])
        self.assertEqual(loc["bbox"]["west"], -83.742)
        self.assertAlmostEqual(loc["area_km2"], 0.99 * 1.0, delta=0.05)

    def test_bad_bbox(self):
        with self.assertRaises(ValueError):
            location.parse_bbox("-83.7 42.3 -83.8 42.2")
        with self.assertRaises(ValueError):
            location.parse_bbox("1 2 3")

    def test_geocode_refuses_large_place_and_boxes_centre(self):
        hit = [{"lat": "42.28", "lon": "-83.74", "display_name": "Ann Arbor", "osm_type": "relation",
                "osm_id": 1, "boundingbox": ["42.22", "42.32", "-83.80", "-83.67"]}]
        fetch = lambda url: hit  # noqa: E731
        with self.assertRaises(ValueError):  # city box is ~120 km²: never silently shrunk
            location.resolve("Ann Arbor, Michigan", fetch=fetch)
        loc = location.resolve("Ann Arbor, Michigan", size_km=1, fetch=fetch)
        self.assertAlmostEqual(loc["area_km2"], 1.0, delta=0.01)
        self.assertIn("geocoded centre", loc["method"])
        self.assertEqual(location.slug("Ann Arbor, Michigan"), "ann-arbor")


class Hardware(unittest.TestCase):
    def test_collect_without_probe(self):
        m = hardware.collect(None, Path("."), probe=False)
        for k in ("os", "arch", "cpu", "cpu_cores", "ram_gb", "gpus", "compute"):
            self.assertIn(k, m)
        self.assertTrue(m["cpu"])

    def test_labels(self):
        with mock.patch.object(hardware, "probe_pipeline_python", return_value={"mps_available": True}):
            self.assertIn("MPS usable", hardware.collect("py", Path("."))["compute"])
        with mock.patch.object(hardware, "probe_pipeline_python", return_value={}), \
                mock.patch.object(hardware, "nvidia_gpus", return_value=[]), \
                mock.patch.object(hardware.platform, "machine", return_value="x86_64"):
            self.assertEqual(hardware.collect("py", Path("."))["compute"], "CPU only")


    def test_probe_reports_pipeline_python(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "src").mkdir()
            probe = hardware.probe_pipeline_python(sys.executable, Path(d))
        import platform
        self.assertEqual(probe["python"], platform.python_version())
        self.assertTrue("torch" in probe or "torch_error" in probe)  # either is recorded, never guessed
        with mock.patch.object(hardware, "probe_pipeline_python",
                               return_value={"python": "3.13.0", "torch": "2.14.1", "torch_cuda_build": None}):
            m = hardware.collect("py", Path("."))
        self.assertEqual((m["pipeline_python"], m["torch_version"], m["torch_cuda_build"]),
                         ("3.13.0", "2.14.1", None))


class Paths(unittest.TestCase):
    def test_rel_and_index(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            (run / "artifacts").mkdir(parents=True)
            (run / "artifacts" / "profile.json").write_text("{}", encoding="utf-8")
            self.assertEqual(run_test.rel(run / "artifacts" / "profile.json", run), "artifacts/profile.json")
            self.assertEqual(run_test.rel("/elsewhere/model.glb", run), "/elsewhere/model.glb")  # --glb
            self.assertIsNone(run_test.rel(None, run))
            idx = run_test.file_index(run)
        self.assertEqual(idx, {"summary": "summary.json", "report": "report.md",
                               "profile": "artifacts/profile.json"})


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ann-arbor-full" / "summary.json"


class CompareSchemas(unittest.TestCase):
    """compare.py and report.py on the published psm-test/1 example and on a psm-test/2 summary."""

    def setUp(self):
        if not EXAMPLE.exists():
            self.skipTest("example not present")
        import compare
        self.compare = compare
        self.old = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.new = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.new["schema"] = "psm-test/2"
        self.new["machine"].update(pipeline_python="3.13.0", torch_version="2.14.1", torch_cuda_build=None)
        self.new["pipeline"]["ml_devices"] = pipeline.ml_devices(REAL_LOG, "mps")
        self.new["pipeline"]["model_downloads"] = pipeline.model_downloads(REAL_LOG, None, None)

    def test_old_schema(self):
        r = self.compare.row(self.old)
        self.assertEqual(r["schema"], "psm-test/1")
        self.assertEqual(r["python"], "3.13.0")       # from the profiler block
        self.assertEqual(r["torch"], "2.14.1")        # from machine.torch
        self.assertEqual((r["ml_device_used"], r["weights"], r["torch_cuda"]), ("unrecorded",) * 3)
        self.assertEqual(r["ml_device_expected"], "mps")
        self.assertAlmostEqual(r["compute_s"], r["total_s"] - r["fetch_s"], places=6)
        self.assertEqual(r["M1_rmse_m"], self.old["evaluations"][0]["scores"]["M1_height_err_rmse"])
        self.assertIsNone(r["M1_bias_m"])            # not copied into psm-test/1 summaries
        self.assertEqual((r["poses"], r["colmap"], r["alignment"], r["aligned_share"]),
                         ("raw", None, "SKIPPED", None))

    def test_new_schema_and_mixed_table(self):
        r = self.compare.row(self.new)
        self.assertEqual((r["ml_device_used"], r["weights"], r["torch_cuda"]), ("mps", "downloaded", None))
        self.assertEqual(list(r), list(self.compare.row(self.old)))  # same columns for both schemas
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, s in (("old", self.old), ("new", self.new)):
                paths.append(Path(d) / f"{name}.json")
                paths[-1].write_text(json.dumps(s), encoding="utf-8")
            out = Path(d) / "c.csv"
            with mock.patch.object(sys, "argv", ["compare.py", *map(str, paths), "--csv", str(out)]), \
                    mock.patch("sys.stdout") as so:
                self.compare.main()
            printed = "".join(c.args[0] for c in so.write.call_args_list)
            import csv
            with out.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertIn("weights", printed.split("differ between runs:")[1])
        self.assertIn("cold start", printed)
        self.assertIn("did not exist in that run's schema", printed)

    def test_texture_columns_and_eval_stage_time(self):
        raw = json.loads(TEXTURE_FIXTURE.read_text(encoding="utf-8"))
        self.new["evaluations"] = [e for e in self.new["evaluations"] if e["name"] != "texture_quality_uv_fid"]
        self.new["evaluations"].append(evals.result("texture_quality_uv_fid", "OK", "x",
                                                    scores=evals.texture_scores(raw)))
        stages = self.new["performance"]["stages"]
        stages.append({"name": "evaluate_texture", "status": "ok", "wall_time_s": 30.0})
        self.new["performance"]["total_wall_time_s"] += 30.0
        r, old = self.compare.row(self.new), self.compare.row(self.old)
        self.assertEqual((r["texture"], r["tex_confidence"], r["tex_partial"], r["tex_O"], r["tex_F"]),
                         ("OK", raw["confidence"], True, raw["terms"]["O"], None))
        self.assertEqual(r["evaluate_texture_s"], 30.0)
        self.assertAlmostEqual(r["compute_s"], old["compute_s"], places=6)  # scoring time left out
        self.assertEqual(r["total_s"], old["total_s"] + 30.0)
        self.assertEqual((old["texture"], old["tex_confidence"]), ("SKIPPED", None))  # psm-test/1 example

    def test_colmap_columns(self):
        self.new["pipeline"]["pipeline_args"] = ["--terrain", "--poses", "colmap"]
        self.new["pipeline"]["colmap"] = {"used": True, "version": "4.2.1", "cuda": True}
        self.new["evaluations"] = [e for e in self.new["evaluations"] if e["name"] != "pose_alignment_stats"]
        self.new["evaluations"].append(evals.result("pose_alignment_stats", "OK", "x",
                                                    scores=evals.alignment_scores(ALIGN_SUMMARY)))
        r = self.compare.row(self.new)
        self.assertEqual((r["poses"], r["colmap"], r["alignment"], r["frames"], r["aligned_share"]),
                         ("colmap", "4.2.1+cuda", "OK", 40, 0.75))
        self.assertIn("align_poses_s", r)
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, s in (("old", self.old), ("new", self.new)):
                paths.append(Path(d) / f"{name}.json")
                paths[-1].write_text(json.dumps(s), encoding="utf-8")
            with mock.patch.object(sys, "argv", ["compare.py", *map(str, paths)]), mock.patch("sys.stdout") as so:
                self.compare.main()
            printed = "".join(c.args[0] for c in so.write.call_args_list)
        self.assertIn("poses differ", printed)

    def test_report_renders_both(self):
        from psmtest import report
        old = report.markdown(self.old)
        self.assertIn("not recorded (older summary)", old)
        new = report.markdown(self.new)
        self.assertIn("confirmed from PSM's log — segmentation: mps, lama: mps", new)
        self.assertIn("downloaded (https://github.com/x/releases/download/v0.1.0/big-lama.pt)", new)


class Profile(unittest.TestCase):
    def test_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profile.json"
            self.assertFalse(pipeline.parse_profile(p)["available"])
            p.write_text(json.dumps(PROFILE), encoding="utf-8")
            perf = pipeline.parse_profile(p)
        self.assertEqual(perf["total_wall_time_s"], 12.5)
        self.assertEqual(perf["stages"][0]["cpu_time_s"], 1.0)
        self.assertIsNone(perf["stages"][0]["cuda_alloc_delta_mb"])  # "N/A" -> unavailable
        self.assertEqual(perf["stages"][0]["cuda_start"],  # absent -> unavailable, not 0
                         {"allocated_mb": None, "reserved_mb": None, "free_mb": None})

    def test_cuda_snapshots_kept(self):
        prof = json.loads(json.dumps(PROFILE))
        prof["stages"][0].update(gpu_start={"allocated_mb": 10.0, "reserved_mb": 20.0, "free_mb": 7000.0},
                                 gpu_end={"allocated_mb": 900.5, "reserved_mb": 1024.0, "free_mb": "N/A"},
                                 gpu_delta_allocated_mb=890.5)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profile.json"
            p.write_text(json.dumps(prof), encoding="utf-8")
            st = pipeline.parse_profile(p)["stages"][0]
        self.assertEqual(st["cuda_alloc_delta_mb"], 890.5)
        self.assertEqual(st["cuda_start"]["free_mb"], 7000.0)
        self.assertEqual(st["cuda_end"], {"allocated_mb": 900.5, "reserved_mb": 1024.0, "free_mb": None})

    def test_saved_profiles_round_trip(self):
        """Every stage value parsed from the real profile.json equals the raw value."""
        ex = Path(__file__).resolve().parents[1] / "examples" / "ann-arbor-full" / "artifacts" / "profile.json"
        if not ex.exists():
            self.skipTest("example not present")
        raw = json.loads(ex.read_text(encoding="utf-8"))
        perf = pipeline.parse_profile(ex)
        self.assertEqual(perf["total_wall_time_s"], raw["total_wall_time_s"])
        num = lambda v: v if isinstance(v, (int, float)) else None  # noqa: E731
        for r, s in zip(raw["stages"], perf["stages"], strict=True):
            self.assertEqual((s["wall_time_s"], s["cpu_time_s"], s["py_heap_peak_mb"]),
                             (r["wall_time_s"], r["cpu_time_s"], r["peak_memory_mb"]))
            self.assertEqual(s["cuda_end"]["reserved_mb"], num(r["gpu_end"]["reserved_mb"]))

    def test_classify_and_log(self):
        scan = pipeline.scan_log("x\nTerrain failed: boom\n3 building(s) were left out of the mesh:\n")
        self.assertEqual(scan["degraded"], ["terrain_failed"])
        self.assertEqual(scan["flagged_buildings"], 3)
        self.assertEqual(pipeline.classify(0, True, {}, []), "completed")
        self.assertEqual(pipeline.classify(0, True, {}, ["terrain_failed"]), "completed_degraded")
        self.assertEqual(pipeline.classify(1, True, {}, []), "partial")
        self.assertEqual(pipeline.classify(1, False, {"available": False}, []), "failed")


REAL_LOG = """\
Downloading images:   0%|          | 0/349 [00:00<?, ?it/s]
Downloading images:   0%|          | 0/349 [01:01<?, ?it/s]Downloading: "https://github.com/x/releases/download/v0.1.0/big-lama.pt" to /Users/u/.cache/torch/hub/checkpoints/big-lama.pt
Loading segmentation model (facebook/mask2former-swin-large-mapillary-vistas-semantic) on mps …
Loading LaMa inpainting model on mps …
  Downloaded 2 Terrarium elevation tile(s) at zoom 15
"""  # shapes copied from run 2's logs/pipeline.log


class DevicesAndDownloads(unittest.TestCase):
    def test_devices_confirmed_vs_expected(self):
        d = pipeline.ml_devices(REAL_LOG, expected="mps")
        self.assertEqual(d["status"], "confirmed")
        self.assertEqual(d["expected"], "mps")
        self.assertEqual(d["used"], ["mps"])
        self.assertEqual(d["observed"]["segmentation"]["model"],
                         "facebook/mask2former-swin-large-mapillary-vistas-semantic")
        self.assertEqual(d["observed"]["lama"]["device"], "mps")

    def test_device_not_observed_keeps_expected(self):
        d = pipeline.ml_devices("fetch ...\nTerrain ok\n", expected="mps")  # e.g. --no-seg
        self.assertEqual((d["status"], d["used"], d["expected"]), ("not_observed", [], "mps"))

    def test_lama_fallback_is_cpu(self):
        d = pipeline.ml_devices("Loading LaMa inpainting model on mps …\n"
                                "LaMa on mps failed (out of memory); using the CPU\n", expected="mps")
        self.assertEqual(d["observed"]["lama"]["device"], "cpu")
        self.assertEqual(d["observed"]["lama"]["fell_back_from"], "mps")
        self.assertEqual(pipeline.ml_devices("Model nvidia/segformer on cuda:0\n", None)["used"], ["cuda:0"])

    def test_download_states(self):
        snap = lambda *f: {"dirs": {}, "files": set(f)}  # noqa: E731
        d = pipeline.model_downloads(REAL_LOG, None, None)  # log evidence alone confirms it
        self.assertEqual(d["status"], "downloaded")
        self.assertEqual(d["log_evidence"], ["https://github.com/x/releases/download/v0.1.0/big-lama.pt"])
        d = pipeline.model_downloads("", snap("a"), snap("a", "huggingface/models--fb--m2f/abc"))
        self.assertEqual((d["status"], d["new_cache_models"]), ("downloaded", ["models--fb--m2f"]))
        self.assertEqual(pipeline.model_downloads("", snap("a"), snap("a"))["status"], "not_downloaded")
        self.assertEqual(pipeline.model_downloads("", None, snap("a"))["status"], "unknown")
        self.assertEqual(pipeline.model_downloads("", None, None)["status"], "unknown")
        # Mapillary image and terrain-tile downloads are not model weights
        only_data = "Downloading images:  10%|#   | 1/9\n  Downloaded 2 Terrarium elevation tile(s)\n"
        self.assertEqual(pipeline.model_downloads(only_data, snap(), snap())["status"], "not_downloaded")
        hf = "model.safetensors:  40%|####      | 350M/880M [00:10<00:15, 35MB/s]\n"
        self.assertEqual(pipeline.model_downloads(hf, None, None)["log_evidence"], ["model.safetensors"])

    def test_cache_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            env = {"HF_HUB_CACHE": f"{d}/hf", "TORCH_HOME": f"{d}/torch"}
            self.assertEqual(pipeline.cache_snapshot(env)["files"], set())  # no cache yet = empty
            blobs = Path(d, "hf", "models--fb--m2f", "blobs")
            blobs.mkdir(parents=True)
            (blobs / "abc").write_bytes(b"x")
            (blobs / "def.incomplete").write_bytes(b"x")  # unfinished download: ignored
            ck = Path(d, "torch", "hub", "checkpoints")
            ck.mkdir(parents=True)
            (ck / "big-lama.pt").write_bytes(b"x")
            self.assertEqual(pipeline.cache_snapshot(env)["files"],
                             {"huggingface/models--fb--m2f/abc", "torch_hub/big-lama.pt"})
            self.assertEqual(pipeline.model_cache_dirs({"HF_HOME": "/h"})["huggingface"], Path("/h/hub"))


@unittest.skipIf(os.name == "nt", "fake colmap is a shell script")
class Colmap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.repo = self.t / "PSM"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "src" / "main.py").write_text("", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_info_from_banner(self):
        new = pipeline.colmap_info(str(fake_colmap(self.t)))
        self.assertEqual((new["version"], new["cuda"], new["version_ok"]), ("4.2.1", True, True))
        (self.t / "old").mkdir()
        old = pipeline.colmap_info(str(fake_colmap(self.t / "old", "3.9.1", cuda=False)))
        self.assertEqual((old["version"], old["cuda"], old["version_ok"]), ("3.9.1", False, False))
        self.assertFalse(pipeline.colmap_info(None)["version_ok"])
        self.assertIn("error", pipeline.colmap_info(str(self.t / "missing")))

    def test_env_file_value(self):
        (self.repo / ".env").write_text('MAPILLARY_ACCESS_TOKEN="secret"\nexport COLMAP_BIN="/opt/c/colmap"\n',
                                        encoding="utf-8")
        self.assertEqual(pipeline.env_file_value(self.repo, "COLMAP_BIN"), "/opt/c/colmap")
        self.assertIsNone(pipeline.env_file_value(self.repo, "NOPE"))
        self.assertIsNone(pipeline.env_file_value(self.t, "COLMAP_BIN"))  # no .env

    def test_resolve_precedence(self):
        cfg, base = config.load(None)
        a, b, c = (fake_colmap(self.t / d) for d in ("a", "b", "c") if (self.t / d).mkdir() is None)
        (self.repo / ".env").write_text(f"COLMAP_BIN={c}\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"COLMAP_BIN": str(b)}):
            self.assertEqual(config.resolve_colmap(dict(cfg, psm={**cfg["psm"], "colmap": str(a)}), base, self.repo),
                             (str(a), "config psm.colmap"))
            self.assertEqual(config.resolve_colmap(cfg, base, self.repo), (str(b), "COLMAP_BIN"))
        with mock.patch.dict(os.environ, {"COLMAP_BIN": ""}):
            self.assertEqual(config.resolve_colmap(cfg, base, self.repo), (str(c), "<PSM>/.env COLMAP_BIN"))
            (self.repo / ".env").unlink()
            with mock.patch.object(config.shutil, "which", return_value=None):
                self.assertEqual(config.resolve_colmap(cfg, base, self.repo), (None, "not found"))
            with self.assertRaises(ValueError):
                config.resolve_colmap(dict(cfg, psm={**cfg["psm"], "colmap": str(self.t / "nope")}), base, self.repo)

    def test_plan(self):
        cfg, base = config.load(None)
        self.assertFalse(run_test.colmap_plan(cfg, base, self.repo)["used"])  # poses = raw
        cfg["pipeline"]["poses"] = "colmap"
        with mock.patch.dict(os.environ, {"COLMAP_BIN": str(fake_colmap(self.t))}):
            plan = run_test.colmap_plan(cfg, base, self.repo)
        self.assertEqual((plan["used"], plan["version"], plan["cuda"]), (True, "4.2.1", True))
        self.assertNotIn("warning", plan)
        self.assertIn("COLMAP 4.2.1, with CUDA", plan["check"])
        with mock.patch.dict(os.environ, {"COLMAP_BIN": ""}), mock.patch.object(config.shutil, "which", return_value=None):
            plan = run_test.colmap_plan(cfg, base, self.repo)
        self.assertEqual(plan["check"], "NOT FOUND")
        self.assertIn("keep Mapillary's poses", plan["warning"])

    def test_alignment_scores(self):
        x = evals.alignment_scores(ALIGN_SUMMARY, "mapillary_roi_3")
        self.assertEqual((x["frames"], x["frames_aligned"], x["frames_reliable"]), (40, 30, 24))
        self.assertEqual((x["aligned_share"], x["reliable_share"]), (0.75, 0.6))
        self.assertEqual((x["legs"], x["legs_aligned"], x["features"], x["mapper"]), (2, 1, "aliked", "global"))
        self.assertEqual((x["pieces"], x["pieces_aligned"]), (2, 1))
        self.assertEqual((x["fit_s_total"], x["roi_folder"]), (12.0, "mapillary_roi_3"))
        # per-leg timings ("fit leg01") are left out, aggregates kept
        x = evals.alignment_scores({**ALIGN_SUMMARY, "seconds": {"fit leg01": 4.9, "fit_legs": 78.4}})
        self.assertEqual([k for k in x if k.startswith("fit_s_")], ["fit_s_fit_legs"])
        # two capture sequences on one leg: 1 leg, 2 pieces
        two = {**ALIGN_SUMMARY, "legs": [{"leg": 4, "aligned": True}, {"leg": 4, "aligned": True}]}
        y = evals.alignment_scores(two)
        self.assertEqual((y["legs"], y["legs_aligned"], y["pieces"], y["pieces_aligned"]), (1, 1, 2, 2))

    def test_wall_fit(self):
        def piece(before, after, w_before, w_after, aligned=True):
            return {"leg": "leg01", "aligned": aligned, "points_to_footprint": {
                "before": {"median_m": before, "within_1m": w_before},
                "after": {"median_m": after, "within_1m": w_after}}}
        pieces = [piece(10.42, 0.80, 0.02, 0.53), piece(4.33, 1.56, 0.13, 0.41), piece(1.41, 1.50, 0.41, 0.40),
                  piece(99.0, 99.0, 0.0, 0.0, aligned=False)]  # not aligned: left out
        x = evals.alignment_scores({**ALIGN_SUMMARY, "legs": pieces})
        self.assertEqual((x["wall_fit_median_m_before"], x["wall_fit_median_m_after"]), (4.33, 1.5))
        self.assertEqual((x["wall_within_1m_before"], x["wall_within_1m_after"]), (0.13, 0.41))
        self.assertIsNone(evals.alignment_scores(ALIGN_SUMMARY)["wall_fit_median_m_after"])  # no data
        self.assertIsNone(evals.alignment_scores({})["aligned_share"])

    def test_poses_cli(self):
        cfg, _ = run_test.settings(run_test.parse_args(["--poses", "colmap"]))
        self.assertEqual(cfg["pipeline"]["poses"], "colmap")
        self.assertIn("colmap", config.pipeline_args(cfg["pipeline"]))


class Evaluations(unittest.TestCase):
    def test_geometry_scores(self):
        s = {"requirements": {"terrain": {"result": "PASS", "share": 0.97, "ci95": [0.9, 0.99], "n": 40},
                              "height": {"share": 0.4, "n": 40}},
             "groups": {"all": {"M1_height_err": {"n": 40, "median": 1.2, "rmse": 3.4, "mae": 2.0,
                                                  "bias_mean": -0.5, "nmad": 1.1, "within_1m": 0.3,
                                                  "within_3m": 0.6}}},
             "status_counts": {"ok": 40}}
        g = evals.geometry_scores(s)
        self.assertEqual(g["terrain_QR06a"], "PASS")
        self.assertEqual(g["M1_height_err_rmse"], 3.4)
        self.assertEqual((g["M1_height_err_bias_mean"], g["M1_height_err_nmad"],
                          g["M1_height_err_within_1m"], g["M1_height_err_within_3m"]), (-0.5, 1.1, 0.3, 0.6))
        self.assertIsNone(g["M2_terrain_err_rmse"])
        self.assertIsNone(g["M2_terrain_err_bias_mean"])

    def test_scores_match_saved_run(self):
        """The real evaluator output in the published example -> every copied value matches."""
        ex = Path(__file__).resolve().parents[1] / "examples" / "ann-arbor-full" / "eval" / "geometry" / "summary.json"
        if not ex.exists():
            self.skipTest("example not present")
        raw = json.loads(ex.read_text(encoding="utf-8"))
        g = evals.geometry_scores(raw)
        for m in evals.GEOMETRY_METRICS:
            for k in evals.GEOMETRY_STATS:
                self.assertEqual(g[f"{m}_{k}"], raw["groups"]["all"][m][k], f"{m}_{k}")

    def test_broken_evaluation_is_isolated(self):
        def boom(ctx):
            raise RuntimeError("bad")
        with mock.patch.object(evals, "EVALUATIONS", [boom, evals.texture_uv_fid]):
            r = evals.run_all({})
        self.assertEqual([x["status"] for x in r], ["FAILED", "SKIPPED"])
        self.assertEqual(r[1]["name"], "texture_quality_uv_fid")


class EndToEndFake(unittest.TestCase):
    """run_test.main() against a fake src/main.py run with this interpreter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.repo = t / "PSM"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "src" / "main.py").write_text(FAKE_MAIN, encoding="utf-8")
        (self.repo / "scripts" / "eval").mkdir(parents=True)
        (self.repo / "scripts/eval/eval_geometry_lidar.py").write_text("import sys; print('no LiDAR'); sys.exit(3)", encoding="utf-8")
        self.results = t / "results"
        (t / "empty.toml").write_text("", encoding="utf-8")
        self.base = ["--config", str(t / "empty.toml"), "--bbox", "-83.742", "42.277", "-83.730", "42.286", "--name", "aa",
                     "--psm-repo", str(self.repo), "--python", sys.executable, "--results", str(self.results)]

    def tearDown(self):
        self.tmp.cleanup()

    def _with_texture_flag(self):
        """Make the fake main.py look like PSM >= 9f5dbb6 (defines --evaluate-texture)."""
        main = self.repo / "src" / "main.py"
        old = 'ap.add_argument("--terrain", action="store_true")'
        main.write_text(main.read_text(encoding="utf-8").replace(
            old, old + '; ap.add_argument("--evaluate-texture", action="store_true")'), encoding="utf-8")

    def _tex(self, s):
        return next(e for e in s["evaluations"] if e["name"] == "texture_quality_uv_fid")

    def test_texture_skipped_on_old_psm(self):
        _, s, run = self._run()  # the plain fake has no --evaluate-texture, like PSM 767fbaa
        self.assertNotIn("--evaluate-texture", s["pipeline"]["pipeline_args"])  # would crash old PSM
        tex = self._tex(s)
        self.assertEqual(tex["status"], "SKIPPED")
        self.assertIn("9f5dbb6", tex["reason"])

    def test_texture_scored(self):
        self._with_texture_flag()
        _, s, run = self._run({"FAKE_TEX_SUMMARY": str(TEXTURE_FIXTURE)})
        self.assertIn("--evaluate-texture", s["pipeline"]["pipeline_args"])
        tex = self._tex(s)
        self.assertEqual(tex["status"], "OK")
        raw = json.loads(TEXTURE_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual((tex["scores"]["confidence"], tex["scores"]["partial"]), (raw["confidence"], True))
        self.assertEqual((tex["scores"]["O_obstruction"], tex["scores"]["D_distortion"], tex["scores"]["F_realism"]),
                         (raw["terms"]["O"], raw["terms"]["D"], None))
        self.assertEqual(tex["scores"]["faces"], raw["counts"]["faces"])
        self.assertEqual(tex["artifacts"], ["artifacts/texture_eval/summary.json", "artifacts/texture_eval/per_group.csv"])
        self.assertEqual(s["files"]["texture_summary"], "artifacts/texture_eval/summary.json")
        rep = (run / "report.md").read_text(encoding="utf-8")
        self.assertIn("### Texture quality", rep)
        self.assertIn("Partial score: F (realism) left out — turned off (--no-fid)", rep)

    def test_texture_not_run_reported(self):
        self._with_texture_flag()
        for mode, status, text in (("unavailable", "SKIPPED", "not available: Missing package: torchvision"),
                                   ("failed", "FAILED", "failed: RuntimeError: CUDA out of memory"),
                                   ("silent", "FAILED", "summary.json was not written")):
            for d in list(self.results.iterdir()) if self.results.exists() else []:
                shutil.rmtree(d)  # temp results of the previous mode
            rc, s, _ = self._run({"FAKE_TEX": mode})
            tex = self._tex(s)
            self.assertEqual((rc, tex["status"]), (0, status), mode)  # never fails the run
            self.assertIn(text, tex["reason"], mode)
            self.assertNotIn("The model is kept", tex["reason"])

    def test_texture_turned_off(self):
        self._with_texture_flag()
        self.base.append("--no-texture-eval")
        _, s, _ = self._run()
        self.assertNotIn("--evaluate-texture", s["pipeline"]["pipeline_args"])
        self.assertIn("turned off", self._tex(s)["reason"])

    @unittest.skipIf(os.name == "nt", "fake colmap is a shell script")
    def test_colmap_alignment(self):
        exe = fake_colmap(Path(self.tmp.name))
        self.base += ["--poses", "colmap"]
        _, s, run = self._run({"COLMAP_BIN": str(exe)})
        self.assertIn("--poses", s["pipeline"]["pipeline_args"])
        cm = s["pipeline"]["colmap"]
        self.assertEqual((cm["used"], cm["path"], cm["version"], cm["cuda"], cm["source"]),
                         (True, str(exe), "4.2.1", True, "COLMAP_BIN"))
        al = next(e for e in s["evaluations"] if e["name"] == "pose_alignment_stats")
        self.assertEqual(al["status"], "OK", al["reason"])
        self.assertEqual((al["scores"]["frames_aligned"], al["scores"]["aligned_share"]), (30, 0.75))
        self.assertEqual(al["artifacts"], ["eval/alignment/summary.json", "eval/alignment/leg0_seqA/stats.json"])
        self.assertEqual(s["status"], "completed_degraded")  # only the fake's "Terrain failed"
        self.assertNotIn("alignment_failed", s["pipeline"]["degraded"])
        rep = (run / "report.md").read_text(encoding="utf-8")
        self.assertIn("### Pose alignment (COLMAP)", rep)
        self.assertIn("COLMAP: " + str(exe) + " — version 4.2.1, with CUDA", rep)

    def test_colmap_missing(self):
        self.base += ["--poses", "colmap"]
        with mock.patch.object(config.shutil, "which", return_value=None):
            rc, s, _ = self._run({"COLMAP_BIN": ""})
        self.assertEqual(rc, 0)  # PSM keeps Mapillary's poses and finishes
        self.assertEqual(s["pipeline"]["colmap"]["path"], None)
        self.assertIn("alignment_failed", s["pipeline"]["degraded"])
        self.assertIn("colmap_missing", s["pipeline"]["degraded"])
        al = next(e for e in s["evaluations"] if e["name"] == "pose_alignment_stats")
        self.assertEqual(al["status"], "FAILED")
        self.assertIn("COLMAP not found", al["reason"])

    def _run(self, extra_env=None, deps_missing=True):
        t = Path(self.tmp.name)  # model caches in the temp folder: the real caches are never read
        env = {"MAPILLARY_ACCESS_TOKEN": "x", "PSM_MACHINE_LABEL": "test-box",
               "HF_HUB_CACHE": str(t / "hf"), "TORCH_HOME": str(t / "torch"), **(extra_env or {})}
        miss = ["laspy"] if deps_missing else []
        with mock.patch.dict(os.environ, env), mock.patch.object(evals, "missing_modules", return_value=miss), \
                mock.patch("sys.stdout"):
            rc = run_test.main(self.base)
        run = next(self.results.iterdir())
        return rc, json.loads((run / "summary.json").read_text(encoding="utf-8")), run

    def test_success_missing_metric_deps(self):
        rc, s, run = self._run()
        self.assertEqual(rc, 0)
        self.assertEqual(s["status"], "completed_degraded")  # fake log says "Terrain failed"
        self.assertEqual(s["performance"]["stages"][0]["name"], "fetch")
        self.assertEqual(s["pipeline"]["pipeline_args"], ["--terrain"])
        geo = s["evaluations"][0]
        self.assertEqual((geo["name"], geo["status"]), ("geometry_vs_lidar", "SKIPPED"))
        self.assertIn("laspy", geo["reason"])
        self.assertTrue((run / "report.md").read_text(encoding="utf-8").startswith("# PSM test"))
        self.assertTrue((run / "logs" / "pipeline.log").exists())
        # schema 2: paths relative to the run folder, and each one resolves from there
        self.assertEqual(s["schema"], "psm-test/2")
        self.assertEqual(s["pipeline"]["glb"], "artifacts/model.glb")
        self.assertEqual(s["files"]["profile"], "artifacts/profile.json")
        for path in s["files"].values():
            self.assertTrue((run / path).exists(), path)
        self.assertNotIn(str(run), json.dumps(s["evaluations"]))
        # no model line, no new cache files: not observed / warm
        self.assertEqual(s["pipeline"]["ml_devices"]["status"], "not_observed")
        self.assertEqual(s["pipeline"]["model_downloads"]["status"], "not_downloaded")
        self.assertIn("pipeline_python", s["machine"])

    def test_device_and_cold_start_recorded(self):
        _, s, run = self._run({"FAKE_SEG": "1", "FAKE_DOWNLOAD": "1"})
        dev, dl = s["pipeline"]["ml_devices"], s["pipeline"]["model_downloads"]
        self.assertEqual((dev["status"], dev["used"]), ("confirmed", ["cpu"]))
        self.assertEqual((dl["status"], dl["new_cache_models"]), ("downloaded", ["fake-lama.pt"]))
        rep = (run / "report.md").read_text(encoding="utf-8")
        self.assertIn("confirmed from PSM's log — segmentation: cpu", rep)
        self.assertIn("downloaded (fake-lama.pt)", rep)

    def test_metric_artifacts_relative(self):
        script = ("import sys, json, pathlib\nout = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
                  "out.mkdir(parents=True)\n(out / 'per_building.csv').write_text('osm_id\\n1\\n', encoding='utf-8')\n"
                  "(out / 'summary.json').write_text(json.dumps({'groups': {'all': {}}}), encoding='utf-8')\n")
        (self.repo / "scripts/eval/eval_geometry_lidar.py").write_text(script, encoding="utf-8")
        _, s, run = self._run(deps_missing=False)
        geo = s["evaluations"][0]
        self.assertEqual(geo["status"], "OK")
        self.assertEqual(geo["artifacts"], ["eval/geometry/summary.json", "eval/geometry/per_building.csv"])
        self.assertEqual(geo["log"], "logs/eval_geometry.log")
        self.assertEqual(s["files"]["geometry_per_building"], "eval/geometry/per_building.csv")

    def test_metric_failure_recorded(self):
        rc, s, _ = self._run(deps_missing=False)
        geo = s["evaluations"][0]
        self.assertEqual(geo["status"], "FAILED")
        self.assertIn("no LiDAR", geo["reason"])
        self.assertEqual(rc, 0)  # a failed metric does not fail the run

    def test_non_utf8_console_encoding(self):
        """Windows regression: a piped child Python writes cp1252 by default, and PSM's
        ProgressMonitor prints '\u2713'. The suite must make PSM write UTF-8 and keep the mark."""
        t = Path(self.tmp.name)
        # 1. the bug reproduces without the suite's environment ...
        raw = subprocess.run([sys.executable, str(self.repo / "src" / "main.py"), "--bbox", "0", "0", "1", "1",
                              "--output", str(t / "m.glb"), "--profile", str(t / "p.json")],
                             capture_output=True, env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
        self.assertNotEqual(raw.returncode, 0)
        self.assertIn(b"UnicodeEncodeError", raw.stderr)
        # 2. ... and not through run_test.py, with the same cp1252 setting inherited
        rc, s, run = self._run({"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
        self.assertEqual(rc, 0)
        self.assertEqual(s["status"], "completed_degraded")  # all stages ran ("Terrain failed" is faked)
        self.assertEqual(s["performance"]["total_wall_time_s"], 12.5)
        self.assertIn("\u2713 Completed: fetch (1/7)", (run / "logs" / "pipeline.log").read_text(encoding="utf-8"))
        self.assertIn("CUDA \u0394 MB", (run / "report.md").read_text(encoding="utf-8"))

    def test_failed_pipeline(self):
        rc, s, run = self._run({"FAKE_FAIL": "1"})
        self.assertEqual(rc, 1)
        self.assertEqual(s["status"], "failed")
        self.assertFalse(s["performance"]["available"])
        self.assertTrue(any("No images found" in l for l in s["pipeline"]["error_lines"]))
        self.assertTrue(all(e["status"] == "SKIPPED" for e in s["evaluations"]))
        self.assertEqual([e["name"] for e in s["evaluations"]],  # same names as in a successful run
                         ["geometry_vs_lidar", "texture_quality_uv_fid", "texture_confidence",
                          "pose_alignment_stats", "render_vs_photo_metrics"])
        self.assertIn("Unavailable", (run / "report.md").read_text(encoding="utf-8"))


class Config(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        for name in ("cfgPSM", "envPSM", "cliPSM"):
            (self.t / name / "src").mkdir(parents=True)
            (self.t / name / "src" / "main.py").write_text("", encoding="utf-8")
        self.cfg_file = self.t / "c" / "config.toml"
        self.cfg_file.parent.mkdir()
        self.cfg_file.write_text(textwrap.dedent("""
            [psm]
            dir = "../cfgPSM"
            [region]
            name = "Test"
            bbox = [-83.75, 42.28, -83.747, 42.2825]
            [pipeline]
            segmentation = false
            min_quality = 0.6
            extra_args = ["--texture", "best"]
            [test]
            machine_label = "cfg-box"
        """), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_and_args(self):
        cfg, _ = config.load(None)
        self.assertEqual(config.pipeline_args(cfg["pipeline"]), ["--terrain", "--evaluate-texture"])
        cfg, base = config.load(self.cfg_file)
        self.assertEqual(base, self.cfg_file.parent.resolve())
        self.assertEqual(config.pipeline_args(cfg["pipeline"]),
                         ["--terrain", "--no-seg", "--min-quality", "0.6", "--evaluate-texture",
                          "--texture", "best"])
        p = dict(cfg["pipeline"], segmentation=True, fill_obstructions=False, naip=True, poses="colmap",
                 aligned_only=True, min_quality=None, evaluate_texture=False, extra_args=[])
        self.assertEqual(config.pipeline_args(p),
                         ["--terrain", "--naip", "--no-fill", "--poses", "colmap", "--aligned-only"])

    def test_example_config_is_valid(self):
        cfg, _ = config.load(Path(__file__).resolve().parents[1] / "config.example.toml")
        self.assertEqual(cfg["region"]["bbox"], [-83.74998, 42.28025, -83.74695, 42.2825])
        self.assertEqual(config.pipeline_args(cfg["pipeline"]), ["--terrain", "--evaluate-texture"])

    def test_invalid(self):
        for text in ('[pipeline]\nposes = "gps"', '[pipeline]\nmin_quality = 2', '[nope]\nx = 1',
                     '[pipeline]\nbogus = 1', '[region]\nbbox = [1, 2]',
                     '[region]\nbbox = [1,2,3,4]\nlocation = "x"', '[pipeline]\naligned_only = true',
                     '[pipeline]\nevaluate_texture = "yes"'):
            self.cfg_file.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError, msg=text):
                config.load(self.cfg_file)

    def test_psm_precedence(self):
        cfg, base = config.load(self.cfg_file)
        with mock.patch.dict(os.environ, {"PSM_REPO": ""}):
            repo, src = config.resolve_psm(None, cfg, base)
            self.assertEqual((repo.name, src), ("cfgPSM", "config psm.dir"))  # relative to config file
        with mock.patch.dict(os.environ, {"PSM_REPO": str(self.t / "envPSM")}):
            self.assertEqual(config.resolve_psm(None, cfg, base)[0].name, "envPSM")
            self.assertEqual(config.resolve_psm(str(self.t / "cliPSM"), cfg, base)[0].name, "cliPSM")
            with self.assertRaises(ValueError):  # not a PSM checkout
                config.resolve_psm(str(self.t / "c"), cfg, base)

    def test_venv_python_symlink_kept(self):
        """A virtualenv's bin/python is a symlink to the base Python; following it would run
        PSM without its packages (seen on Great Lakes: /sw/pkgs/arc/python/3.12.1/bin/python3.12)."""
        base_py = self.t / "base" / "bin" / "python3.12"
        base_py.parent.mkdir(parents=True)
        base_py.write_text("", encoding="utf-8")
        venv_py = self.t / "venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.symlink_to(base_py)
        cfg, base = config.load(self.cfg_file)
        cfg["psm"]["python"] = "../venv/bin/python"  # relative to the config file
        venv_end = os.path.join("venv", "bin", "python")
        with mock.patch.dict(os.environ, {"PSM_PYTHON": ""}):
            got, src = config.resolve_python(None, cfg, base, self.t)
            self.assertTrue(got.endswith(venv_end), got)  # not .../base/bin/python3.12
            self.assertEqual(src, "config psm.python")
            self.assertEqual(config.resolve_python(str(venv_py), cfg, base, self.t), (str(venv_py), "--python"))
        with mock.patch.dict(os.environ, {"PSM_PYTHON": str(venv_py)}):
            self.assertEqual(config.resolve_python(None, cfg, base, self.t), (str(venv_py), "PSM_PYTHON"))

    def test_cli_overrides_config(self):
        a = run_test.parse_args(["--config", str(self.cfg_file), "--bbox", "1", "2", "3", "4",
                                 "--no-terrain", "--", "--naip"])
        cfg, _ = run_test.settings(a)
        self.assertEqual(cfg["region"]["bbox"], [1, 2, 3, 4])
        self.assertEqual(cfg["region"]["name"], "")  # config's name belonged to its own bbox
        self.assertEqual(config.pipeline_args(cfg["pipeline"]),
                         ["--no-seg", "--min-quality", "0.6", "--evaluate-texture", "--texture", "best", "--naip"])
        for flag in ("--no-texture-eval", "--no-evals"):
            a = run_test.parse_args(["--config", str(self.cfg_file), flag])
            cfg, _ = run_test.settings(a)
            self.assertNotIn("--evaluate-texture", config.pipeline_args(cfg["pipeline"]), flag)

    def test_texture_eval_plan(self):
        repo = self.t / "cfgPSM"
        (repo / "src" / "main.py").write_text('ap.add_argument("--terrain")\n', encoding="utf-8")
        old = run_test.texture_eval_plan(["--terrain", "--evaluate-texture"], repo)
        self.assertEqual((old["requested"], old["dropped"]), (False, True))
        self.assertIn("9f5dbb6", old["reason"])
        (repo / "src" / "main.py").write_text('ap.add_argument(\n    "--evaluate-texture",\n)\n', encoding="utf-8")
        new = run_test.texture_eval_plan(["--terrain", "--evaluate-texture"], repo)
        self.assertEqual((new["requested"], new["dropped"], new["reason"]), (True, False, None))
        off = run_test.texture_eval_plan(["--terrain"], repo)
        self.assertEqual((off["requested"], off["dropped"]), (False, False))
        self.assertIn("turned off", off["reason"])

    def test_run_dirs_never_reused(self):
        a = run_test.new_run_dir(self.t / "r", "x")
        b = run_test.new_run_dir(self.t / "r", "x")
        self.assertNotEqual(a, b)
        self.assertEqual(b.name, "x_2")


if __name__ == "__main__":
    unittest.main()
