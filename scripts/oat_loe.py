"""oat_loe.py — lightweight cross-species LOEO for oat (H2 direction test).

Builds oat environments from the T3-exported yield+heading observations,
resolves coordinates via GHCN station city/state matching, re-extracts NASA
POWER daily weather over a spring-oat season window, computes oat-profile
stage features, and runs a lightweight LOEO (learned embedding) to get the
per-environment Delta for the Δ-dist direction test.

This is the cross-species confirmation for the wheat result
(reports/discriminator_final_2026-08-06.md): does Delta INCREASE with dist
for a second species?

Usage:
    python scripts/oat_loe.py --items data/t3/oat_items_100.parquet \
        --embed-mode learned --epochs 40 --plot-cap 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "nc" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", type=Path, default=ROOT / "data/t3/oat_items_100.parquet")
    parser.add_argument("--out-results", type=Path, default=ROOT / "data/t3/loe_oat_learned.parquet")
    parser.add_argument("--embed-mode", choices=["learned", "pca"], default="learned")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--plot-cap", type=int, default=100)
    parser.add_argument("--fold-workers", type=int, default=8)
    parser.add_argument("--geno-offset", choices=["none", "empirical"], default="none",
                        help="empirical = M3 fairness diagnostic (shared genotype main effect)")
    parser.add_argument("--n-gpus", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def resolve_coords(location_name: str, stations: pd.DataFrame) -> tuple[float, float] | None:
    """Resolve 'City, ST' to (lat, lon) via GHCN station city match."""
    import re

    m = re.match(r"^\s*([A-Za-z .\-]+?)\s*,\s*([A-Z]{2})\s*$", location_name)
    if not m:
        return None
    city, state = m.group(1).strip(), m.group(2)
    sub = stations[stations["state"] == state]
    if sub.empty:
        return None
    cnorm = re.sub(r"[^a-z]", "", city.lower())
    match = sub[sub["name"].str.lower().str.replace(r"[^a-z]", "", regex=True).str.startswith(cnorm).fillna(False)]
    if match.empty:
        return None
    row = match.sort_values("name").iloc[0]
    return float(row["latitude"]), float(row["longitude"])


def fetch_power(lat, lon, start, end):
    """NASA POWER daily weather; returns {date: (tmax, tmin, tmean, precip, solar, rh)}."""
    import urllib.request
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,T2M_MIN,T2M,"
           "PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M&community=AG&longitude={lon}&latitude={lat}"
           "&start={start}&end={end}&format=JSON").format(lon=lon, lat=lat, start=start, end=end)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                import json
                return json.loads(r.read())["properties"]["parameter"]
        except Exception:
            time.sleep(3.0 * (attempt + 1))
    return None


def build_oat_env_features(items_path: Path, cache_path: Path | None = None) -> dict[str, np.ndarray]:
    """Resolve coords + fetch NASA POWER + compute oat stage features per env_id.

    Returns {env_id: x_matrix}.  Cached to `cache_path` (pickle) since the NASA
    POWER fetch is slow/rate-limited and this env-level feature set is reused
    by both the LOEO run and the Δ-dist curve (dist(e) computation).
    """
    import pickle

    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"[oat_features] loaded {len(cached)} cached env features -> {cache_path}")
        return cached

    from gxe_budget.data.crop_profiles import get_crop_profile
    from envindex.r1_unified import stage_summaries

    obs = pd.read_parquet(items_path)
    yield_obs = obs[obs["trait"].str.contains("yield", case=False)].copy()
    yield_obs["value"] = pd.to_numeric(yield_obs["value"], errors="coerce")
    yield_obs = yield_obs.dropna(subset=["value"])
    print(f"[oat_features] {len(yield_obs)} yield obs, {yield_obs['env_id'].nunique()} envs")

    stations = pd.read_csv(ROOT / "data/t3/ghcnd-stations.txt", sep=r"\s+",
                           names=["id", "latitude", "longitude", "elev", "state", "name"],
                           usecols=[0, 1, 2, 3, 4, 5], header=None)
    coords = {}
    for loc in yield_obs["location_name"].unique():
        c = resolve_coords(loc, stations)
        if c:
            coords[loc] = c
    print(f"[oat_features] resolved coords for {len(coords)}/{yield_obs['location_name'].nunique()} locations")
    yield_obs = yield_obs[yield_obs["location_name"].isin(coords)]

    profile = get_crop_profile("oat")
    env_x: dict[str, np.ndarray] = {}
    for env_id, g in yield_obs.groupby("env_id"):
        loc = g["location_name"].iloc[0]
        year = int(g["year"].iloc[0])
        lat, lon = coords[loc]
        # spring-oat season window (US oat nursery): Apr 1 - Aug 31
        start = f"{year}0401"
        end = f"{year}0831"
        param = fetch_power(lat, lon, start, end)
        if not param:
            continue
        days = sorted(param.get("T2M", {}))
        rows = []
        for i, d in enumerate(days):
            def v(k):
                x = param.get(k, {}).get(d)
                return None if (x is None or x == -999.0) else float(x)
            tmean, tmax, tmin = v("T2M"), v("T2M_MAX"), v("T2M_MIN")
            rows.append({"day_after_planting": i, "tmax": tmax, "tmin": tmin, "tmean": tmean,
                         "precipitation": v("PRECTOTCORR"), "solar_radiation": v("ALLSKY_SFC_SW_DWN"),
                         "relative_humidity": v("RH2M"), "vpd": None, "gdd": None})
        if not rows:
            continue
        wdf = pd.DataFrame(rows)
        es = 0.6108 * np.exp(17.27 * wdf["tmean"] / (wdf["tmean"] + 237.3))
        wdf["vpd"] = es - es * wdf["relative_humidity"] / 100.0
        wdf["gdd"] = np.maximum(0, wdf["tmean"] - profile.gdd_base_temp)
        env_x[env_id] = stage_summaries(wdf, profile.stage_windows, profile.heat_day_tmax_threshold)
    print(f"[oat_features] built features for {len(env_x)} envs")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(env_x, f)
        print(f"[oat_features] cached -> {cache_path}")
    return env_x


def build_oat_items(items_path: Path, cache_path: Path | None = None) -> list[dict]:
    """Per-(env,geno) LOEO items, reusing the cached/rebuilt per-env feature matrix."""
    env_x = build_oat_env_features(items_path, cache_path)
    obs = pd.read_parquet(items_path)
    yield_obs = obs[obs["trait"].str.contains("yield", case=False)].copy()
    yield_obs["value"] = pd.to_numeric(yield_obs["value"], errors="coerce")
    yield_obs = yield_obs.dropna(subset=["value"])
    yield_obs = yield_obs[yield_obs["env_id"].isin(env_x)]
    items = []
    for env_id, g in yield_obs.groupby("env_id"):
        year = int(g["year"].iloc[0])
        for _, r in g.iterrows():
            items.append({"env_id": env_id, "geno": str(r["geno"]), "y": float(r["value"]),
                          "x": env_x[env_id], "env_label": year})
    return items


def main(argv=None):
    sys.path.insert(0, str(ROOT / "scripts"))
    from loe_pilot import run_loe

    args = _parse_args(argv)
    cache_path = ROOT / "data/t3/oat_env_features.pkl"
    items = build_oat_items(args.items, cache_path)
    print(f"[oat_loe] built {len(items)} items, {len(set(i['env_id'] for i in items))} envs")

    # plot cap
    if args.plot_cap > 0:
        df_items = pd.DataFrame(items)
        df_items = df_items.groupby("env_id").head(args.plot_cap)
        items = df_items.to_dict("records")

    # lightweight LOEO
    res = run_loe(items, d_embed=32, d_geno=32, rank=4, epochs=args.epochs, device=args.device,
                  seed=0, batch_size=512, fold_workers=args.fold_workers, n_gpus=args.n_gpus,
                  embed_mode=args.embed_mode, geno_offset=args.geno_offset)
    args.out_results.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([{"crop": "oat", "env_id": e, **r} for e, r in res.items()])
    out.to_parquet(args.out_results, index=False)
    print(f"[oat_loe] results ({len(out)} envs) -> {args.out_results}")
    print(f"[oat_loe] mean Delta: {out['delta'].mean():+.4f} | Gz {out['pcc_gz'].mean():+.3f} | G+E {out['pcc_ge'].mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
