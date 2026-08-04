"""soynam_census.py — Census SoyNAM environments and phenology coverage.

Reads the public SoyNAM R package dataset (CRAN SoyNAM_1.6.2.tar.gz) and
reports which environments have flowering (R1/GDD) phenology records.

SoyNAM dataset objects:
  data.line  (main):  18 environments (9 locations x 2011-2013) with `flower`
                      (flowering date), `planting`, `maturity`, `R8`, `yield`
  data.line.in (Purdue): 3 environments (IN 2013-2015) with `R1` (days to
                      flowering), `GDD_R1`, `GDD_R8`

Outputs an environment catalog parquet + summary.

Usage:
    python scripts/soynam_census.py --out data/t3/soynam_envs.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

CRAN_TARBALL = "https://cran.r-project.org/src/contrib/SoyNAM_1.6.2.tar.gz"
DEFAULT_CACHE = Path("data/t3/SoyNAM_1.6.2.tar.gz")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tarball", type=Path, default=DEFAULT_CACHE, help="SoyNAM tarball cache path")
    parser.add_argument("--out", type=Path, default=Path("data/t3/soynam_envs.parquet"), help="Output parquet")
    return parser.parse_args(argv)


def _ensure_tarball(path: Path) -> Path:
    import urllib.request

    if path.exists() and path.stat().st_size > 0:
        print(f"[soynam_census] using cached tarball: {path}")
        return path
    print(f"[soynam_census] downloading {CRAN_TARBALL}")
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(CRAN_TARBALL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as out:
        out.write(r.read())
    return path


def _extract_and_convert(tarball: Path, rdata_name: str) -> dict:
    import rdata

    with tarfile.open(tarball, "r:gz") as tar:
        member = tar.getmember(f"SoyNAM/data/{rdata_name}")
        f = tar.extractfile(member)
        tmp = Path(tempfile.mkstemp(suffix=".RData")[1])
        tmp.write_bytes(f.read())
        f.close()
    try:
        parsed = rdata.parser.parse_file(str(tmp))
        return rdata.conversion.convert(parsed)
    finally:
        try:
            tmp.unlink()
        except PermissionError:
            pass  # Windows keeps the file locked; the temp dir is cleaned by the OS


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    args = _parse_args(argv)
    tarball = _ensure_tarball(args.tarball)

    main = _extract_and_convert(tarball, "soynam.RData")
    purdue = _extract_and_convert(tarball, "soyin.RData")
    dl = main["data.line"]
    dli = purdue["data.line.in"]

    # Main dataset: environments with flowering-date coverage.
    main_envs = dl.groupby("environ").agg(
        year=("year", "first"),
        location=("location", "first"),
        n_plots=("strain", "count"),
        flower_coverage=("flower", lambda s: int(s.notna().mean() * 100)),
        planting_coverage=("planting", lambda s: int(s.notna().mean() * 100)),
        yield_coverage=("yield", lambda s: int(s.notna().mean() * 100)),
    ).reset_index()
    main_envs["dataset"] = "data.line"

    # Purdue subset: R1 / GDD records (no `location` column; state is the
    # environ prefix, e.g. "IN_2013").
    pur_envs = dli.groupby("environ").agg(
        year=("year", "first"),
        n_plots=("strain", "count"),
        r1_coverage=("R1", lambda s: int(s.notna().mean() * 100)),
        gdd_r1_coverage=("GDD_R1", lambda s: int(s.notna().mean() * 100)),
        gdd_r8_coverage=("GDD_R8", lambda s: int(s.notna().mean() * 100)),
    ).reset_index()
    pur_envs["location"] = pur_envs["environ"].str.split("_").str[0]
    pur_envs["dataset"] = "data.line.in"

    catalog = pd.concat([main_envs, pur_envs], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(args.out, index=False)

    print(f"[soynam_census] environments: main={len(main_envs)} (all flower-date), "
          f"purdue={len(pur_envs)} (all R1/GDD)")
    print(f"[soynam_census] catalog -> {args.out}")
    print(f"[soynam_census] main env flower coverage: {int(dl['flower'].notna().sum())}/{len(dl)} "
          f"| purdue R1: {int(dli['R1'].notna().sum())}/{len(dli)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
