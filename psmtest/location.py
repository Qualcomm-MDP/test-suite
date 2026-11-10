"""Turn a place name, a bounding box or a centre point into the WEST SOUTH EAST NORTH box
that PSM's `src/main.py --bbox` takes. Only place names need the network (OSM Nominatim)."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request

NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "psm-test-suite/2.0"
EARTH_R_KM = 6371.0088
KM_PER_DEG_LAT = 111.32


def area_km2(w: float, s: float, e: float, n: float) -> float:
    """Approximate area of a lon/lat box (equirectangular; fine for boxes of a few km)."""
    width = math.radians(e - w) * EARTH_R_KM * math.cos(math.radians((s + n) / 2))
    height = math.radians(n - s) * EARTH_R_KM
    return abs(width * height)


def check_bbox(w: float, s: float, e: float, n: float) -> tuple[float, float, float, float]:
    if not (-180 <= w < e <= 180):
        raise ValueError(f"need -180 <= WEST < EAST <= 180, got WEST={w} EAST={e}")
    if not (-90 <= s < n <= 90):
        raise ValueError(f"need -90 <= SOUTH < NORTH <= 90, got SOUTH={s} NORTH={n}")
    return w, s, e, n


def parse_bbox(value) -> tuple[float, float, float, float]:
    """'W,S,E,N', 'W S E N' or a list of 4 numbers."""
    parts = value.replace(",", " ").split() if isinstance(value, str) else list(value)
    if len(parts) != 4:
        raise ValueError(f"a bounding box needs 4 numbers (WEST SOUTH EAST NORTH), got {value!r}")
    return check_bbox(*(float(p) for p in parts))


def box_around(lat: float, lon: float, size_km: float) -> tuple[float, float, float, float]:
    """Square box of side size_km centred on (lat, lon)."""
    half_lat = size_km / 2 / KM_PER_DEG_LAT
    half_lon = size_km / 2 / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))
    return check_bbox(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def _http_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def geocode(query: str, fetch=_http_json) -> dict:
    url = NOMINATIM + "?" + urllib.parse.urlencode({"q": query, "format": "json", "limit": 1})
    hits = fetch(url)
    if not hits:
        raise ValueError(f"Nominatim found no place called {query!r}; pass --bbox instead")
    h = hits[0]
    s, n, w, e = (float(x) for x in h["boundingbox"])  # Nominatim order: S, N, W, E
    return {"display_name": h.get("display_name"), "lat": float(h["lat"]), "lon": float(h["lon"]),
            "osm": f"{h.get('osm_type')} {h.get('osm_id')}", "place_bbox": [w, s, e, n]}


def resolve(location: str | None = None, bbox=None, center=None, size_km: float | None = None,
            max_area_km2: float = 2.0, fetch=_http_json) -> dict:
    """Return {query, method, bbox: {west,south,east,north}, area_km2, geocoder}.
    Never shrinks or moves an area on its own: a place whose own box is larger than
    max_area_km2 is refused unless size_km asks for a box around its centre."""
    geo = None
    if bbox is not None:
        w, s, e, n = parse_bbox(bbox)
        method, query = "bbox", None
    elif center is not None:
        if not size_km:
            raise ValueError("--center needs --size-km")
        lat, lon = (float(x) for x in center)
        w, s, e, n = box_around(lat, lon, size_km)
        method, query = f"center + {size_km:g} km square", f"{lat},{lon}"
    elif location:
        geo = geocode(location, fetch)
        query = location
        if size_km:
            w, s, e, n = box_around(geo["lat"], geo["lon"], size_km)
            method = f"geocoded centre + {size_km:g} km square"
        else:
            w, s, e, n = check_bbox(*geo["place_bbox"])
            method = "geocoded place bbox"
    else:
        raise ValueError("give --location, --bbox or --center")
    area = area_km2(w, s, e, n)
    if area > max_area_km2:
        raise ValueError(
            f"{method} for {query or 'the box'} covers {area:.2f} km² (> --max-area-km2 "
            f"{max_area_km2:g}). PSM runtime grows with area; pass --size-km (e.g. 1) for a box "
            f"around the centre, an explicit --bbox, or raise --max-area-km2.")
    return {"query": query, "method": method,
            "bbox": {"west": round(w, 6), "south": round(s, 6), "east": round(e, 6), "north": round(n, 6)},
            "area_km2": round(area, 4), "geocoder": geo}


def slug(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in text.split(",")[0])
    return "-".join(p for p in out.split("-") if p)[:40] or "area"
