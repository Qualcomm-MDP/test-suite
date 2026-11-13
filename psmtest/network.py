"""Network conditions of a run: how fast the services PSM downloads from answered, how fast a
plain download went, and how many bytes the machine received during each pipeline stage.

Standard library only. Nothing here sends credentials: the Mapillary request carries no token
(it is answered with an error, which still measures the round trip)."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "psm-test-suite/2 (benchmark network check)"
# The hosts PSM downloads from (PSM src/common/constants.py, terrain/terrain.py, naip/naip.py,
# common/providers/osm_api.py); small, cheap requests to each.
SERVICES = {
    "mapillary_api": "https://graph.mapillary.com/images",
    "overpass": "https://overpass-api.de/api/status",
    "osm_api": "https://api.openstreetmap.org/api/capabilities",
    "elevation_tiles": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/0/0/0.png",
    "naip": "https://imagery.nationalmap.gov/arcgis/rest/services?f=json",
}
# A fixed-size download from Cloudflare's public speed-test endpoint (no account, no cookies).
SPEED_URL = "https://speed.cloudflare.com/__down?bytes={n}"
SPEED_BYTES = 25_000_000
NOTE_SHARED = ("bytes_received is the whole machine's traffic, not PSM's alone: on a shared node "
               "(e.g. a cluster node running other jobs) it includes other users' downloads")


def _request(url: str, timeout: float):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                  timeout=timeout)


def latency(url: str, timeout: float = 10.0) -> dict:
    """Time from sending a request to its first byte (DNS + connect + TLS + server). An HTTP
    error status still counts: the server answered."""
    t0 = time.perf_counter()
    try:
        with _request(url, timeout) as r:
            r.read(1)
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"url": url, "ok": False, "ms": None, "error": str(getattr(e, "reason", e))[:200]}
    return {"url": url, "ok": True, "ms": round((time.perf_counter() - t0) * 1000, 1), "http_status": code}


def speed_test(nbytes: int = SPEED_BYTES, timeout: float = 30.0) -> dict:
    """Download nbytes from the speed-test endpoint; stops after `timeout` s and reports the rate
    over what arrived (a slow link is measured, not waited out)."""
    url = SPEED_URL.format(n=nbytes)
    got, t0 = 0, time.perf_counter()
    try:
        with _request(url, timeout) as r:
            while got < nbytes and time.perf_counter() - t0 < timeout:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                got += len(chunk)
    except (urllib.error.URLError, OSError, ValueError) as e:
        if not got:
            return {"url": url, "ok": False, "mbps": None, "error": str(getattr(e, "reason", e))[:200]}
    secs = time.perf_counter() - t0
    return {"url": url, "ok": got > 0, "bytes": got, "seconds": round(secs, 2),
            "complete": got >= nbytes, "mbps": round(got * 8 / secs / 1e6, 1) if got and secs else None}


def probe(speed: bool = True) -> dict:
    """Latency to every service PSM uses and, optionally, a download speed test."""
    out = {"measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "latency": {name: latency(url) for name, url in SERVICES.items()}}
    out["download"] = speed_test() if speed else {"ok": False, "skipped": True}
    return out


# ------------------------------------------------------------------ bytes received during the run

def rx_bytes() -> int | None:
    """Bytes this machine has received on all non-loopback interfaces since boot, or None."""
    try:
        if sys.platform.startswith("linux"):
            total = 0
            for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
                name, data = line.split(":", 1)
                if name.strip() != "lo":
                    total += int(data.split()[0])
            return total
        if sys.platform == "darwin":
            out = subprocess.run(["netstat", "-ibn"], capture_output=True, text=True, timeout=10).stdout
            return parse_netstat_ibn(out)
        if sys.platform == "win32":
            out = subprocess.run(["netstat", "-e"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"^Bytes\s+(\d+)", out, re.M)
            return int(m.group(1)) if m else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def parse_netstat_ibn(out: str) -> int | None:
    """macOS `netstat -ibn`: one <Link#n> row per interface; Ibytes is 5th from the right
    (the Address column can be empty, so count from the right)."""
    total, seen = 0, False
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) < 7 or not f[2].startswith("<Link#") or f[0].startswith("lo"):
            continue
        try:
            total += int(f[-5])
            seen = True
        except ValueError:
            continue
    return total if seen else None


class Sampler:
    """Samples rx_bytes() every `interval` s in a background thread while the pipeline runs."""

    def __init__(self, interval: float = 1.0, read=rx_bytes):
        self.interval, self.read, self.samples = interval, read, []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while True:
            v = self.read()
            if v is not None:
                self.samples.append((time.time(), v))
            if self._stop.wait(self.interval):
                break

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)
        v = self.read()
        if v is not None:
            self.samples.append((time.time(), v))


def _at(samples, t: float) -> float | None:
    """Received-bytes counter at time t, interpolated between samples."""
    if not samples or t < samples[0][0] or t > samples[-1][0]:
        return None
    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return float(samples[-1][1])


def stage_traffic(samples, started_at: str | None, stages: list[dict]) -> dict:
    """Bytes received and the average rate during each profiler stage. Stage windows come from
    the profile's started_at plus the stages' wall times in order (PSM runs them one by one)."""
    if len(samples) < 2:
        return {"available": False, "reason": "network counters not readable on this platform"}
    run = {"bytes_received": samples[-1][1] - samples[0][1],
           "seconds": round(samples[-1][0] - samples[0][0], 1)}
    run["mbps"] = round(run["bytes_received"] * 8 / run["seconds"] / 1e6, 2) if run["seconds"] else None
    out = {"available": True, "source": "machine-wide received bytes, sampled every second",
           "run": run, "stages": [], "note": NOTE_SHARED}
    try:
        t = datetime.fromisoformat(started_at).timestamp() if started_at else None
    except ValueError:
        t = None
    if t is None:
        out["stages_reason"] = "profile has no started_at"
        return out
    for st in stages:
        wall = st.get("wall_time_s") or 0.0
        a, b = _at(samples, t), _at(samples, t + wall)
        row = {"name": st.get("name"), "seconds": wall, "bytes_received": None, "mbps": None}
        if a is not None and b is not None:
            row["bytes_received"] = int(b - a)
            row["mbps"] = round((b - a) * 8 / wall / 1e6, 2) if wall >= 0.5 else None
        out["stages"].append(row)
        t += wall
    return out
