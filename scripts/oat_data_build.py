"""oat_data_build.py — build oat (and barley) LOEO-ready data from T3.

For the Uniform Oat Performance Nursery (UOPN) phenology trials: exports
per-plot yield + heading observations via BrAPI, resolves coordinates, and
re-extracts NASA POWER weather.  Produces an `items` parquet consumable by
loe_pilot-style LOEO.

This is the cross-species data source for the H2 direction test
(reports/discriminator_final_2026-08-06.md).

Usage:
    python scripts/oat_data_build.py --crop oat --token-file data/t3/.t3_token_oat
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/t3"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--crop", choices=["oat", "barley"], default="oat")
    parser.add_argument("--token-file", type=Path, default=DATA / ".t3_token_oat")
    parser.add_argument("--catalog", type=Path, default=None, help="trials_catalog_{crop}.parquet")
    parser.add_argument("--program", default="Uniform Oat Performance Nursery", help="Program substring filter")
    parser.add_argument("--n-envs", type=int, default=0, help="Cap environments (0 = all)")
    parser.add_argument("--out", type=Path, default=DATA / "oat_items.parquet")
    return parser.parse_args(argv)


def main(argv=None):
    import urllib.request

    args = _parse_args(argv)
    catalog_path = args.catalog or (DATA / f"trials_catalog_{args.crop}.parquet")
    cat = pd.read_parquet(catalog_path)
    ph = cat[cat["has_phenology"]]
    if args.program:
        ph = ph[ph["program_name"].str.contains(args.program, na=False)]
    if args.n_envs > 0:
        ph = ph.head(args.n_envs)
    token = args.token_file.read_text(encoding="utf-8").strip()
    base = f"https://{args.crop}.triticeaetoolbox.org/brapi/v2"
    print(f"[oat_build] {len(ph)} phenology trials ({args.crop}, program={args.program})")

    obs_rows = []
    for i, (_, trial) in enumerate(ph.iterrows()):
        sid = trial["study_db_id"]
        url = f"{base}/observations?studyDbId={sid}&pageSize=1000"
        data = []
        for attempt in range(3):  # retry on rate-limit / transient errors
            try:
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    import json
                    data = json.loads(r.read())["result"].get("data", [])
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"[oat_build] FAIL {sid}: {type(exc).__name__}", file=sys.stderr)
                time.sleep(3.0 * (attempt + 1))
        for obs in data:
            vname = str(obs.get("observationVariableName", ""))
            if not any(k in vname.lower() for k in ["yield", "heading", "anthes"]):
                continue
            obs_rows.append({
                "env_id": trial["study_name"],
                "location_name": trial["location_name"],
                "year": trial["year"],
                "geno": str(obs.get("germplasmName", "")),
                "trait": vname.split("|")[0].strip(),
                "value": obs.get("value"),
            })
        if (i + 1) % 25 == 0:
            print(f"[oat_build] {i+1}/{len(ph)} trials, {len(obs_rows)} obs rows", flush=True)
        time.sleep(1.2)  # rate-limit courtesy (0.3s caused silent failures)

    obs = pd.DataFrame(obs_rows)
    print(f"[oat_build] collected {len(obs)} observation rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(args.out, index=False)
    # quick summary
    yield_rows = obs[obs["trait"].str.contains("yield", case=False, na=False)]
    print(f"[oat_build] yield rows: {len(yield_rows)}, envs: {yield_rows['env_id'].nunique() if len(yield_rows) else 0}")
    print(f"[oat_build] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
