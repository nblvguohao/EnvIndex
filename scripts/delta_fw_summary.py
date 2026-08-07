"""delta_fw_summary.py — four-crop summary of the M3 main metric Delta_FW
(amendment 2026-08-06 §7 选定方案 3, post-D1/D2-fix rerun).

Per crop (learned arm): mean PCC of the interaction model (pcc_gz), the
Finlay-Wilkinson reaction-norm baseline (pcc_fw), the genotype-mean baseline
(pcc_gm), and Delta_FW(e) = pcc_gz - pcc_fw with an environment-level
bootstrap CI and two-sided p.  BH across crops (protocol §6).

Usage:  python scripts/delta_fw_summary.py [--dir data/t3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CROP_FILES = {
    "wheat": "loe_fix_learned_wheat.parquet",
    "corn": "loe_fix_learned_corn.parquet",
    "oat": "loe_oat_fix_learned.parquet",
    "barley": "loe_barley_fix_learned.parquet",
}


def boot_mean(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "data/t3")
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args(argv)

    rows, pvals = [], {}
    for crop, fname in CROP_FILES.items():
        path = args.dir / fname
        if not path.exists():
            print(f"[skip] {crop}: missing {fname}")
            continue
        df = pd.read_parquet(path)
        d = df["delta_fw"].dropna().to_numpy()
        lo, hi, p = boot_mean(d, args.n_boot, args.seed)
        pvals[crop] = p
        rows.append({
            "crop": crop, "n_env": len(d),
            "pcc_gz": df["pcc_gz"].mean(), "pcc_fw": df["pcc_fw"].mean(),
            "pcc_gm": df["pcc_gm"].mean(),
            "delta_fw": d.mean(), "ci_lo": lo, "ci_hi": hi, "p": p,
        })
    df = pd.DataFrame(rows)

    # BH step-up
    m = len(pvals)
    order = sorted(pvals, key=pvals.get)
    passing = [c for r, c in enumerate(order) if pvals[c] <= 0.05 * (r + 1) / m]
    cutoff = max((0.05 * (order.index(c) + 1) / m for c in passing), default=0.0)
    df["bh_sig"] = df["crop"].map({c: pvals[c] <= cutoff for c in pvals})

    print(df.round(4).to_string(index=False))
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"[saved] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
