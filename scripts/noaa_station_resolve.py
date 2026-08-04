"""noaa_station_resolve.py — Resolve T3 trial locations to coordinates via NOAA GHCN-D.

T3's /locations returns no latitude/longitude, but some locations carry a
NOAA GHCN-D station id in additionalInfo.noaa_station_id (e.g.
"GHCND:US1SDBK0019").  This script resolves coordinates for every location in
a trials catalog using two fallbacks:

  1. exact match on the NOAA station id from /locations (if present)
  2. fuzzy match of GHCN-D station NAME/STATE against the location name
     (e.g. "Aurora, SD" -> state SD, name starting "AURORA...")

It emits an enriched environment table (the trials catalog joined with
latitude/longitude/elevation/station) that feeds weather re-extraction.

Station metadata source: NOAA NCEI GHCN-D stations file
https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
(fixed-width: ID, LATITUDE, LONGITUDE, ELEVATION, STATE, NAME, ...).

Usage:
    python scripts/noaa_station_resolve.py \
        --catalog data/t3/trials_catalog_combined.parquet \
        --out data/t3/envs_with_coords.parquet
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from t3_brapi_export import BrApiClient, _get  # noqa: E402

STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
DEFAULT_STATIONS_CACHE = Path("data/t3/ghcnd-stations.txt")
DEFAULT_TOKEN_FILE = Path("data/t3/.t3_token")

# fixed-width column layout of ghcnd-stations.txt (1-indexed inclusive)
STATION_COLUMNS = [
    ("station_id", 1, 11),
    ("latitude", 13, 20),
    ("longitude", 22, 30),
    ("elevation", 32, 37),
    ("state", 39, 40),
    ("name", 42, 71),
    ("gsn_flag", 73, 75),
    ("hcn_crn_flag", 77, 79),
    ("wmo_id", 81, 85),
]

CITY_STATE_RE = re.compile(r"^\s*([A-Za-z .\-]+?)\s*,\s*([A-Z]{2})\s*$")


# ---------------------------------------------------------------- stations

def _stations_file_complete(path: Path, min_lines: int = 80_000) -> bool:
    """Heuristic completeness check for ghcnd-stations.txt.

    The full file has ~100k+ station lines including USW (airport) and USC
    (cooperative) prefixes.  A truncated download has far fewer lines and is
    missing USW/USC entries (it may contain only early US1 lines).
    """
    if not path.exists() or path.stat().st_size < 4_000_000:
        return False
    n_lines = 0
    seen_usw_usc = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            if line.startswith(("USW", "USC", "USR", "USS")):
                seen_usw_usc = True
            if n_lines > min_lines and seen_usw_usc:
                break
    return n_lines >= min_lines and seen_usw_usc


def download_stations_file(cache_path: Path, max_retries: int = 6) -> Path:
    """Download the GHCN-D stations file, caching to `cache_path`.

    The connection to NOAA NCEI is flaky (same as T3 on this host), so the
    download retries with backoff and resumes from the partial file via HTTP
    Range when the server supports it.  A stale/truncated cache is re-downloaded.
    """
    import urllib.request

    if _stations_file_complete(cache_path):
        print(f"[noaa_resolve] using cached stations file: {cache_path}")
        return cache_path
    if cache_path.exists():
        print(f"[noaa_resolve] cache incomplete; re-downloading ({cache_path})", file=sys.stderr)
        cache_path.unlink()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial = cache_path.with_suffix(cache_path.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"[noaa_resolve] downloading GHCN-D stations from NOAA NCEI ...")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            _download_resumable(partial, cache_path)
            if not _stations_file_complete(cache_path):
                raise RuntimeError("download incomplete after transfer")
            print(f"[noaa_resolve] stations file -> {cache_path} ({cache_path.stat().st_size} bytes)")
            return cache_path
        except (urllib.error.URLError, ConnectionError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            wait = 2.0 * (2 ** (attempt - 1))
            print(f"[noaa_resolve] download retry {attempt}/{max_retries} "
                  f"({type(exc).__name__}, wait {wait:.0f}s)", file=sys.stderr)
            import time
            time.sleep(wait)
    raise RuntimeError(f"Stations download failed after {max_retries} retries: {last_error}") from last_error


def _download_resumable(partial: Path, cache_path: Path) -> None:
    """Download with HTTP Range resume into `partial`, then atomically rename."""
    import time
    import urllib.request

    headers = {"User-Agent": "EnvIndex-census/0.1"}
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(STATIONS_URL, headers=headers)
    mode = "ab" if offset else "wb"
    with urllib.request.urlopen(request, timeout=60) as response, open(partial, mode) as out:
        while True:
            block = response.read(1 << 16)
            if not block:
                break
            out.write(block)
    if partial.stat().st_size == 0:
        raise RuntimeError("empty download")
    partial.replace(cache_path)


def parse_stations_file(path: Path) -> pd.DataFrame:
    """Parse the fixed-width ghcnd-stations.txt into a DataFrame."""
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(line.rstrip("\n")) < 40:
                continue
            record = {}
            for col, start, end in STATION_COLUMNS:
                record[col] = line[start - 1 : end].strip()
            try:
                record["latitude"] = float(record["latitude"])
                record["longitude"] = float(record["longitude"])
                record["elevation"] = float(record["elevation"]) if record["elevation"] else None
            except ValueError:
                continue
            rows.append(record)
    stations = pd.DataFrame(rows)
    stations["name"] = stations["name"].str.strip()
    return stations


def match_station_by_id(stations: pd.DataFrame, station_id: str) -> pd.Series | None:
    """Exact match on GHCN station id (strip a leading 'GHCND:' if present)."""
    sid = station_id.strip()
    if sid.lower().startswith("ghcnd:"):
        sid = sid[6:].strip()
    hit = stations[stations["station_id"] == sid]
    if hit.empty:
        return None
    return hit.iloc[0]


def _normalise_city(city: str) -> str:
    return re.sub(r"[^a-z]", "", city.lower())


def match_station_by_city(stations: pd.DataFrame, state: str, city: str) -> pd.Series | None:
    """Fuzzy match: same state + GHCN station NAME starting with the city.

    Prefers a station whose name is exactly the city over one with a
    directional/quadrant suffix (e.g. "BROOKINGS 4 N").
    """
    sub = stations[stations["state"] == state]
    if sub.empty:
        return None
    city_norm = _normalise_city(city)
    if not city_norm:
        return None

    def name_norm(name: str) -> str:
        return _normalise_city(name)

    sub = sub.copy()
    sub["_city_norm"] = sub["name"].map(name_norm)
    matched = sub[sub["_city_norm"].str.startswith(city_norm)]
    if matched.empty:
        return None
    # Prefer exact city match, then shortest name (fewest directional suffixes).
    matched = matched.sort_values(
        by="_city_norm",
        key=lambda s: s.str.len(),
    )
    return matched.iloc[0]


def resolve_location(
    stations: pd.DataFrame,
    station_id: str | None,
    location_name: str,
) -> tuple[str | None, str, float, float, float]:
    """Resolve a location to (station_id, match_method, lat, lon, elev).

    Returns match_method in {"exact_station", "city_state", "unmatched"}.
    """
    if station_id:
        hit = match_station_by_id(stations, station_id)
        if hit is not None:
            return hit["station_id"], "exact_station", hit["latitude"], hit["longitude"], hit["elevation"]

    m = CITY_STATE_RE.match(location_name or "")
    if m:
        city, state = m.group(1).strip(), m.group(2)
        hit = match_station_by_city(stations, state, city)
        if hit is not None:
            return hit["station_id"], "city_state", hit["latitude"], hit["longitude"], hit["elevation"]

    return None, "unmatched", None, None, None


# ---------------------------------------------------------------- main

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", type=Path, required=True, help="Trials catalog parquet (must have location_db_id)")
    parser.add_argument("--out", type=Path, default=Path("data/t3/envs_with_coords.parquet"), help="Enriched output parquet")
    parser.add_argument("--stations-file", type=Path, default=DEFAULT_STATIONS_CACHE, help="GHCN stations cache path")
    parser.add_argument("--base-url", default="https://wheat.triticeaetoolbox.org/brapi/v2/", help="T3 BrAPI base URL")
    parser.add_argument("--token", default=None, help="T3 bearer token (or T3_TOKEN / token file)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel /locations fetch workers")
    return parser.parse_args(argv)


def _load_token(args_token: str | None) -> str | None:
    if args_token:
        return args_token
    token = os.environ.get("T3_TOKEN")
    if token:
        return token
    if DEFAULT_TOKEN_FILE.exists():
        return DEFAULT_TOKEN_FILE.read_text(encoding="utf-8").strip()
    return None


def main(argv: list[str] | None = None) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    args = _parse_args(argv)
    catalog = pd.read_parquet(args.catalog)
    if "location_db_id" not in catalog.columns:
        print(f"[noaa_resolve] ERROR: catalog has no location_db_id column", file=sys.stderr)
        return 2

    client = BrApiClient(base_url=args.base_url, token=_load_token(args.token), sleep_seconds=0.1)
    locations = catalog[["location_db_id", "location_name"]].drop_duplicates()
    print(f"[noaa_resolve] {len(locations)} unique locations to resolve")

    def fetch_loc(loc_db_id: str) -> tuple[str, str | None, str]:
        try:
            loc = client._call_with_retry(
                client.base_url + f"locations/{urllib.parse.quote(loc_db_id)}", None
            )
            result = loc.get("result", {})
            additional = _get(result, "additionalInfo", default={})
            station = additional.get("noaa_station_id") if isinstance(additional, dict) else None
            name = result.get("locationName", "")
            return loc_db_id, (station or None), (name or "")
        except Exception as exc:
            print(f"[noaa_resolve] WARN location {loc_db_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return loc_db_id, None, ""

    loc_info: dict[str, tuple[str | None, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_loc, str(db_id)): str(db_id) for db_id in locations["location_db_id"]}
        for future in as_completed(futures):
            db_id, station, name = future.result()
            loc_info[db_id] = (station, name)

    stations_file = download_stations_file(args.stations_file)
    stations = parse_stations_file(stations_file)
    print(f"[noaa_resolve] parsed {len(stations)} GHCN-D stations")

    resolved_rows = []
    n_unmatched = 0
    for db_id, (station_id, api_name) in loc_info.items():
        # Prefer the API-provided name; fall back to the catalog name.
        catalog_name = locations.loc[locations["location_db_id"] == db_id, "location_name"].iloc[0]
        name = api_name or catalog_name
        sid, method, lat, lon, elev = resolve_location(stations, station_id, name)
        if method == "unmatched":
            n_unmatched += 1
        resolved_rows.append(
            {
                "location_db_id": db_id,
                "location_name": name,
                "noaa_station_id_t3": station_id,
                "ghcn_station_id": sid,
                "match_method": method,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev,
            }
        )
    resolved = pd.DataFrame(resolved_rows)

    # catalog and resolved both carry location_name -> keep the canonical
    # /locations name from `resolved`, drop the study-derived one.
    enriched = catalog.drop(columns=["location_name"]).merge(resolved, on="location_db_id", how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.out, index=False)
    print(f"[noaa_resolve] enriched catalog ({len(enriched)} rows) -> {args.out}")
    print(f"[noaa_resolve] match summary: exact_station={int((resolved['match_method']=='exact_station').sum())}, "
          f"city_state={int((resolved['match_method']=='city_state').sum())}, "
          f"unmatched={n_unmatched}")

    if n_unmatched:
        print("[noaa_resolve] unmatched locations:")
        for _, row in resolved[resolved["match_method"] == "unmatched"].iterrows():
            print(f"  - {row['location_db_id']}: {row['location_name']} (station={row['noaa_station_id_t3']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
