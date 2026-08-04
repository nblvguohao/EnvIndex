"""cimmyt_coords.py — Resolve CIMMYT IWIN environment coordinates + build env catalog.

Merges the IWIN cleaned trial files with the AgERA5 IWIN locations table
(LocNo -> Lat/Long, from hdl:11529/10548548) and emits an environment-level
catalog: environment_id, nursery, loc, year, sow/head/matu coverage, yield
coverage, latitude, longitude.

The location table (IWIN_Locations_AgERA5_20210211.txt) is small and fetched
from data.cimmyt.org when missing.

Usage:
    python scripts/cimmyt_coords.py --out data/cimmyt/cimmyt_envs.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/cimmyt")
LOCATIONS_URL = "https://data.cimmyt.org/api/access/datafile/15266"
LOCATIONS_FILENAME = "IWIN_Locations_AgERA5_20210211.txt"

NURSERIES = ["ESWYT", "HTWYT", "IDYN", "IWWYT_IRR", "IWWYT_SA"]
PHENO_COLS = ["sow", "head", "matu"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="CIMMYT data directory")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "cimmyt_envs.parquet", help="Output parquet")
    return parser.parse_args(argv)


def _ensure_locations(data_dir: Path) -> Path:
    path = data_dir / LOCATIONS_FILENAME
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"[cimmyt_coords] downloading IWIN locations table")
    data_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(LOCATIONS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as out:
        out.write(r.read())
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    loc_path = _ensure_locations(args.data_dir)
    locs = pd.read_csv(loc_path)
    locs["LocNo"] = locs["LocNo"].astype(str)

    env_rows = []
    for nursery in NURSERIES:
        fname = f"{nursery}_Obs_Sim_Yld_Phe_Climate_All.tab"
        path = args.data_dir / fname
        if not path.exists():
            print(f"[cimmyt_coords] WARN missing {fname}", file=sys.stderr)
            continue
        df = pd.read_csv(path, sep="\t", usecols=lambda c: c in ["loc", "year", "gen"] + PHENO_COLS + ["yld"])
        df["loc"] = df["loc"].astype(str)
        for (loc, year), g in df.groupby(["loc", "year"]):
            env_rows.append(
                {
                    "nursery": nursery,
                    "loc": loc,
                    "year": int(year),
                    "n_plots": len(g),
                    "n_genotypes": g["gen"].nunique(),
                    "sow_cov": int(100 * g["sow"].notna().mean()),
                    "head_cov": int(100 * g["head"].notna().mean()),
                    "matu_cov": int(100 * g["matu"].notna().mean()),
                    "yld_cov": int(100 * g["yld"].notna().mean()),
                    "mean_yld": float(g["yld"].mean()) if g["yld"].notna().any() else None,
                }
            )
    envs = pd.DataFrame(env_rows)

    envs = envs.merge(
        locs.rename(columns={"LocNo": "loc", "Lat": "latitude", "Long": "longitude"}),
        on="loc",
        how="left",
    )
    envs["has_coords"] = envs["latitude"].notna()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    envs.to_parquet(args.out, index=False)

    n = len(envs)
    n_coord = int(envs["has_coords"].sum())
    print(f"[cimmyt_coords] environments: {n} | with coords: {n_coord} ({100 * n_coord / max(n,1):.0f}%)")
    for nursery, g in envs.groupby("nursery"):
        print(f"  {nursery:10s}: {len(g):4d} envs, {int(g['has_coords'].sum()):4d} with coords, "
              f"head_cov={int(g['head_cov'].mean()):3d}%")
    print(f"[cimmyt_coords] catalog -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
