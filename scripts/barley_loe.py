"""barley_loe.py — lightweight cross-species LOEO for barley (H2 direction test,
4th crop / tie-breaker alongside wheat, corn, oat).

Builds barley environments from the T3-exported yield observations (Western
Regional Barley Nursery — spring barley, has real per-trial planting/harvest
dates unlike the winter malting nurseries), resolves coordinates via GHCN
station city/state matching, re-extracts NASA POWER daily weather over each
trial's actual planting->harvest window (falling back to a generic Apr1-Aug31
spring window when dates are missing), computes barley-profile stage
features, and runs LOEO (learned + PCA-control embedding) to get the
per-environment Delta for the Delta-dist direction test.

Mirrors oat_loe.py (see reports/discriminator_final_2026-08-06.md for the
wheat result this is a cross-species check against); the two diverge only in
crop profile and in using real per-trial season dates instead of a fixed
window, since WRBN's catalog actually has them.

Usage:
    python scripts/barley_loe.py --items data/t3/barley_items_wrbn.parquet \
        --embed-mode learned --epochs 60 --plot-cap 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "nc" / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from oat_loe import resolve_coords, fetch_power  # noqa: E402  (shared T3/GHCN/POWER helpers)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", type=Path, default=ROOT / "data/t3/barley_items_wrbn.parquet")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/t3/trials_catalog_barley.parquet",
                        help="Trial catalog with planting_date/harvest_date, keyed by study_name == env_id")
    parser.add_argument("--out-results", type=Path, default=ROOT / "data/t3/loe_barley_learned.parquet")
    parser.add_argument("--embed-mode", choices=["learned", "pca"], default="learned")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--plot-cap", type=int, default=100)
    parser.add_argument("--fold-workers", type=int, default=8)
    parser.add_argument("--n-gpus", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def build_barley_env_features(items_path: Path, catalog_path: Path, cache_path: Path | None = None) -> dict[str, np.ndarray]:
    """Resolve coords + fetch NASA POWER (per-trial planting->harvest window,
    falling back to a generic Apr1-Aug31 spring window) + compute barley
    stage features per env_id.  Cached to `cache_path` (pickle) — same
    rationale as oat_loe.build_oat_env_features (slow/rate-limited fetch,
    reused by both the LOEO run and the Delta-dist curve)."""
    import pickle

    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"[barley_features] loaded {len(cached)} cached env features -> {cache_path}")
        return cached

    from gxe_budget.data.crop_profiles import get_crop_profile
    from envindex.r1_unified import stage_summaries

    obs = pd.read_parquet(items_path)
    yield_obs = obs[obs["trait"].str.contains("yield", case=False)].copy()
    yield_obs["value"] = pd.to_numeric(yield_obs["value"], errors="coerce")
    yield_obs = yield_obs.dropna(subset=["value"])
    print(f"[barley_features] {len(yield_obs)} yield obs, {yield_obs['env_id'].nunique()} envs")

    cat = pd.read_parquet(catalog_path).set_index("study_name")
    stations = pd.read_csv(ROOT / "data/t3/ghcnd-stations.txt", sep=r"\s+",
                           names=["id", "latitude", "longitude", "elev", "state", "name"],
                           usecols=[0, 1, 2, 3, 4, 5], header=None)
    coords = {}
    for loc in yield_obs["location_name"].unique():
        c = resolve_coords(loc, stations)
        if c:
            coords[loc] = c
    print(f"[barley_features] resolved coords for {len(coords)}/{yield_obs['location_name'].nunique()} locations")
    yield_obs = yield_obs[yield_obs["location_name"].isin(coords)]

    profile = get_crop_profile("barley")
    env_x: dict[str, np.ndarray] = {}
    n_real_dates = 0
    for env_id, g in yield_obs.groupby("env_id"):
        loc = g["location_name"].iloc[0]
        year = int(g["year"].iloc[0])
        lat, lon = coords[loc]

        start = f"{year}0401"
        end = f"{year}0831"
        if env_id in cat.index:
            row = cat.loc[env_id]
            pdate, hdate = row.get("planting_date"), row.get("harvest_date")
            if pd.notna(pdate) and pd.notna(hdate):
                start = pd.Timestamp(pdate).strftime("%Y%m%d")
                end = pd.Timestamp(hdate).strftime("%Y%m%d")
                n_real_dates += 1

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
    print(f"[barley_features] built features for {len(env_x)} envs ({n_real_dates} with real trial dates, "
          f"{len(env_x) - n_real_dates} fell back to generic Apr1-Aug31 window)")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(env_x, f)
        print(f"[barley_features] cached -> {cache_path}")
    return env_x


def build_barley_items(items_path: Path, catalog_path: Path, cache_path: Path | None = None) -> list[dict]:
    """Per-(env,geno) LOEO items, reusing the cached/rebuilt per-env feature matrix."""
    env_x = build_barley_env_features(items_path, catalog_path, cache_path)
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
    from loe_pilot import run_loe

    args = _parse_args(argv)
    cache_path = ROOT / "data/t3/barley_env_features.pkl"
    items = build_barley_items(args.items, args.catalog, cache_path)
    print(f"[barley_loe] built {len(items)} items, {len(set(i['env_id'] for i in items))} envs")

    if args.plot_cap > 0:
        df_items = pd.DataFrame(items)
        df_items = df_items.groupby("env_id").head(args.plot_cap)
        items = df_items.to_dict("records")

    res = run_loe(items, d_embed=32, d_geno=32, rank=4, epochs=args.epochs, device=args.device,
                  seed=0, batch_size=512, fold_workers=args.fold_workers, n_gpus=args.n_gpus,
                  embed_mode=args.embed_mode)
    args.out_results.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([{"crop": "barley", "env_id": e, **r} for e, r in res.items()])
    out.to_parquet(args.out_results, index=False)
    print(f"[barley_loe] results ({len(out)} envs) -> {args.out_results}")
    print(f"[barley_loe] mean Delta: {out['delta'].mean():+.4f} | Gz {out['pcc_gz'].mean():+.3f} | G+E {out['pcc_ge'].mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
