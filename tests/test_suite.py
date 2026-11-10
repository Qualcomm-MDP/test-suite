"""Offline checks of the suite itself, using a fake PSM checkout (no network, no pipeline env).

  python3 -m unittest discover -s tests -v
"""

import json
import os
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

FAKE_MAIN = textwrap.dedent("""
    import argparse, json, os, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4); ap.add_argument("--output"); ap.add_argument("--profile")
    ap.add_argument("--dump-mapillary"); ap.add_argument("--terrain", action="store_true")
    a = ap.parse_args()
    print("fetch ...")
    if os.environ.get("FAKE_FAIL"):
        print("RuntimeError: No images found."); sys.exit(1)
    print("Terrain failed: offline")
    open(a.output, "wb").write(b"glTF")
    json.dump(PROFILE, open(a.profile, "w"))
""").replace("PROFILE", repr(PROFILE))


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


class Profile(unittest.TestCase):
    def test_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "profile.json"
            self.assertFalse(pipeline.parse_profile(p)["available"])
            p.write_text(json.dumps(PROFILE))
            perf = pipeline.parse_profile(p)
        self.assertEqual(perf["total_wall_time_s"], 12.5)
        self.assertEqual(perf["stages"][0]["cpu_time_s"], 1.0)
        self.assertIsNone(perf["stages"][0]["cuda_alloc_delta_mb"])  # "N/A" -> unavailable

    def test_classify_and_log(self):
        scan = pipeline.scan_log("x\nTerrain failed: boom\n3 building(s) were left out of the mesh:\n")
        self.assertEqual(scan["degraded"], ["terrain_failed"])
        self.assertEqual(scan["flagged_buildings"], 3)
        self.assertEqual(pipeline.classify(0, True, {}, []), "completed")
        self.assertEqual(pipeline.classify(0, True, {}, ["terrain_failed"]), "completed_degraded")
        self.assertEqual(pipeline.classify(1, True, {}, []), "partial")
        self.assertEqual(pipeline.classify(1, False, {"available": False}, []), "failed")


class Evaluations(unittest.TestCase):
    def test_geometry_scores(self):
        s = {"requirements": {"terrain": {"result": "PASS", "share": 0.97, "ci95": [0.9, 0.99], "n": 40},
                              "height": {"share": 0.4, "n": 40}},
             "groups": {"all": {"M1_height_err": {"n": 40, "median": 1.2, "rmse": 3.4, "mae": 2.0}}},
             "status_counts": {"ok": 40}}
        g = evals.geometry_scores(s)
        self.assertEqual(g["terrain_QR06a"], "PASS")
        self.assertEqual(g["M1_height_err_rmse"], 3.4)
        self.assertIsNone(g["M2_terrain_err_rmse"])

    def test_broken_evaluation_is_isolated(self):
        def boom(ctx):
            raise RuntimeError("bad")
        with mock.patch.object(evals, "EVALUATIONS", [boom, evals.texture_uv_fid]):
            r = evals.run_all({})
        self.assertEqual([x["status"] for x in r], ["FAILED", "SKIPPED"])


class EndToEndFake(unittest.TestCase):
    """run_test.main() against a fake src/main.py run with this interpreter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.repo = t / "PSM"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / "src" / "main.py").write_text(FAKE_MAIN)
        (self.repo / "scripts" / "eval").mkdir(parents=True)
        (self.repo / "scripts/eval/eval_geometry_lidar.py").write_text("import sys; print('no LiDAR'); sys.exit(3)")
        self.results = t / "results"
        (t / "empty.toml").write_text("")
        self.base = ["--config", str(t / "empty.toml"), "--bbox", "-83.742", "42.277", "-83.730", "42.286", "--name", "aa",
                     "--psm-repo", str(self.repo), "--python", sys.executable, "--results", str(self.results)]

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, extra_env=None, deps_missing=True):
        env = {"MAPILLARY_ACCESS_TOKEN": "x", "PSM_MACHINE_LABEL": "test-box", **(extra_env or {})}
        miss = ["laspy"] if deps_missing else []
        with mock.patch.dict(os.environ, env), mock.patch.object(evals, "missing_modules", return_value=miss), \
                mock.patch("sys.stdout"):
            rc = run_test.main(self.base)
        run = next(self.results.iterdir())
        return rc, json.loads((run / "summary.json").read_text()), run

    def test_success_missing_metric_deps(self):
        rc, s, run = self._run()
        self.assertEqual(rc, 0)
        self.assertEqual(s["status"], "completed_degraded")  # fake log says "Terrain failed"
        self.assertEqual(s["performance"]["stages"][0]["name"], "fetch")
        self.assertEqual(s["pipeline"]["pipeline_args"], ["--terrain"])
        geo = s["evaluations"][0]
        self.assertEqual((geo["name"], geo["status"]), ("geometry_vs_lidar", "SKIPPED"))
        self.assertIn("laspy", geo["reason"])
        self.assertTrue((run / "report.md").read_text().startswith("# PSM test"))
        self.assertTrue((run / "logs" / "pipeline.log").exists())

    def test_metric_failure_recorded(self):
        rc, s, _ = self._run(deps_missing=False)
        geo = s["evaluations"][0]
        self.assertEqual(geo["status"], "FAILED")
        self.assertIn("no LiDAR", geo["reason"])
        self.assertEqual(rc, 0)  # a failed metric does not fail the run

    def test_failed_pipeline(self):
        rc, s, run = self._run({"FAKE_FAIL": "1"})
        self.assertEqual(rc, 1)
        self.assertEqual(s["status"], "failed")
        self.assertFalse(s["performance"]["available"])
        self.assertTrue(any("No images found" in l for l in s["pipeline"]["error_lines"]))
        self.assertTrue(all(e["status"] == "SKIPPED" for e in s["evaluations"]))
        self.assertIn("Unavailable", (run / "report.md").read_text())


class Config(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        for name in ("cfgPSM", "envPSM", "cliPSM"):
            (self.t / name / "src").mkdir(parents=True)
            (self.t / name / "src" / "main.py").write_text("")
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
        """))

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_and_args(self):
        cfg, _ = config.load(None)
        self.assertEqual(config.pipeline_args(cfg["pipeline"]), ["--terrain"])
        cfg, base = config.load(self.cfg_file)
        self.assertEqual(base, self.cfg_file.parent.resolve())
        self.assertEqual(config.pipeline_args(cfg["pipeline"]),
                         ["--terrain", "--no-seg", "--min-quality", "0.6", "--texture", "best"])
        p = dict(cfg["pipeline"], segmentation=True, fill_obstructions=False, naip=True, poses="colmap",
                 aligned_only=True, min_quality=None, extra_args=[])
        self.assertEqual(config.pipeline_args(p),
                         ["--terrain", "--naip", "--no-fill", "--poses", "colmap", "--aligned-only"])

    def test_example_config_is_valid(self):
        cfg, _ = config.load(Path(__file__).resolve().parents[1] / "config.example.toml")
        self.assertEqual(cfg["region"]["bbox"], [-83.74998, 42.28025, -83.74695, 42.2825])
        self.assertEqual(config.pipeline_args(cfg["pipeline"]), ["--terrain"])

    def test_invalid(self):
        for text in ('[pipeline]\nposes = "gps"', '[pipeline]\nmin_quality = 2', '[nope]\nx = 1',
                     '[pipeline]\nbogus = 1', '[region]\nbbox = [1, 2]',
                     '[region]\nbbox = [1,2,3,4]\nlocation = "x"', '[pipeline]\naligned_only = true'):
            self.cfg_file.write_text(text)
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

    def test_cli_overrides_config(self):
        a = run_test.parse_args(["--config", str(self.cfg_file), "--bbox", "1", "2", "3", "4",
                                 "--no-terrain", "--", "--naip"])
        cfg, _ = run_test.settings(a)
        self.assertEqual(cfg["region"]["bbox"], [1, 2, 3, 4])
        self.assertEqual(cfg["region"]["name"], "")  # config's name belonged to its own bbox
        self.assertEqual(config.pipeline_args(cfg["pipeline"]),
                         ["--no-seg", "--min-quality", "0.6", "--texture", "best", "--naip"])

    def test_run_dirs_never_reused(self):
        a = run_test.new_run_dir(self.t / "r", "x")
        b = run_test.new_run_dir(self.t / "r", "x")
        self.assertNotEqual(a, b)
        self.assertEqual(b.name, "x_2")


if __name__ == "__main__":
    unittest.main()
