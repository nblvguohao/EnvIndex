"""nasa_power_extract.py — re-extract daily weather from NASA POWER for
CIMMYT IWIN environments (no data-use agreement needed; alternative to the
restricted AgERA5 file, amendments/2026-08-05_agera5-access-constraint.md).

For each environment it queries NASA POWER daily agroclimatology
(T2M_MAX, T2M_MIN, T2M, PRECTOTCORR, ALLSKY_SFC_SW_DWN, RH2M) over the
growing season (sowing day-of-year -> + days-to-maturity) and writes a
protocol §3.3 weather_daily.parquet (environment_id, date,
day_after_planting, tmax, tmin, tmean, precipitation, solar_radiation,
relative_humidity, vpd, gdd).

The unified R1 extractor (envindex.r1_unified.build_r1_from_daily) can then
compute consistent stage-summary features from this table.

Usage:
    python scripts/nasa_power_extract.py --out data/cimmyt/weather_daily_power.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/cimmyt"

ESWYT = DATA / "ESWYT_Obs_Sim_Yld_Phe_Climate_All.tab"
LOCATIONS = DATA / "IWIN_Locations_AgERA5_20210211.txt"

POWER_URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?"
             "parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M"
             "&community=AG&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eswyt", type=Path, default=ESWYT, help="ESWYT tab file")
    parser.add_argument("--locations", type=Path, default=LOCATIONS, help="IWIN locations table")
    parser.add_argument("--n-envs", type=int, default=0, help="Limit number of environments (0 = all)")
    parser.add_argument("--out", type=Path, default=DATA / "weather_daily_power.parquet")
    parser.add_argument("--gdd-base", type=float, default=0.0, help="Wheat GDD base temperature (C)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls (0 when parallel)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel API workers (NASA POWER ~50 req/min anonymous)")
    return parser.parse_args(argv)


def build_env_table(eswyt: Path, locations: Path) -> pd.DataFrame:
    """Per-environment growing-season window + coordinates from ESWYT + IWIN locations."""
    df = pd.read_csv(eswyt, sep="\t")
    df["env_id"] = df["loc"].astype(str) + "_" + df["year"].astype(str)
    env = df.groupby("env_id").agg(
        loc=("loc", "first"),
        year=("year", "first"),
        sow_doy=("sow", "median"),
        matu_days=("matu", "median"),
    ).reset_index()
    locs = pd.read_csv(locations)
    env = env.merge(locs.rename(columns={"LocNo": "loc", "Lat": "latitude", "Long": "longitude"}), on="loc", how="left")
    env = env.dropna(subset=["latitude", "longitude", "sow_doy", "matu_days"])
    env["planting_date"] = env.apply(
        lambda r: date(int(r["year"]), 1, 1) + timedelta(days=int(r["sow_doy"]) - 1), axis=1
    )
    return env


def fetch_power_daily(lat: float, lon: float, start: date, end: date, retries: int = 5) -> dict:
    url = POWER_URL.format(lat=lat, lon=lon, start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            import json
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())["properties"]["parameter"]
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(5.0 * (2 ** attempt))
    raise RuntimeError(f"NASA POWER failed after {retries}: {last}")


def env_weather(env: pd.Series, gdd_base: float) -> pd.DataFrame:
    """Query POWER for one env and build a protocol-schema weather frame."""
    plant = env["planting_date"]
    end = plant + timedelta(days=int(env["matu_days"]))
    param = fetch_power_daily(env["latitude"], env["longitude"], plant, end)

    def val(key, d):
        v = param.get(key, {}).get(d)
        return None if (v is None or v == -999.0) else v

    days = sorted(param.get("T2M", {}))
    rows = []
    for i, d in enumerate(days):
        tmean = val("T2M", d)
        tmax = val("T2M_MAX", d)
        tmin = val("T2M_MIN", d)
        precip = val("PRECTOTCORR", d)
        solar = val("ALLSKY_SFC_SW_DWN", d)
        rh = val("RH2M", d)
        # VPD from T2M + RH (kPa)
        es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3)) if tmean is not None else None
        vpd = (es - es * rh / 100.0) if (es is not None and rh is not None) else None
        gdd = max(0.0, tmean - gdd_base) if tmean is not None else None
        rows.append({
            "environment_id": env["env_id"],
            "date": pd.to_datetime(d),
            "day_after_planting": i,
            "tmax": tmax, "tmin": tmin, "tmean": tmean,
            "precipitation": precip, "solar_radiation": solar,
            "relative_humidity": rh, "vpd": vpd, "gdd": gdd,
        })
    return pd.DataFrame(rows)


def _extract_one(env: pd.Series, gdd_base: float) -> tuple[str, pd.DataFrame | None, str]:
    try:
        return env["env_id"], env_weather(env, gdd_base), ""
    except Exception as exc:
        return env["env_id"], None, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    args = _parse_args(argv)
    envs = build_env_table(args.eswyt, args.locations)
    if args.n_envs > 0:
        envs = envs.head(args.n_envs)
    print(f"[nasa_power] {len(envs)} environments to extract with {args.workers} workers")

    frames = []
    n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract_one, row, args.gdd_base): env_id for env_id, row in envs.iterrows()}
        done = 0
        for fut in as_completed(futures):
            env_id, w, err = fut.result()
            done += 1
            if w is not None and len(w):
                frames.append(w)
                print(f"[nasa_power] {done}/{len(envs)} {env_id} ({len(w)} days)", flush=True)
            else:
                n_fail += 1
                print(f"[nasa_power] FAIL {env_id}: {err}", file=sys.stderr)
            time.sleep(args.sleep)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.out, index=False)
        print(f"[nasa_power] weather -> {args.out} ({len(out)} rows, {out['environment_id'].nunique()} envs, {n_fail} failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
