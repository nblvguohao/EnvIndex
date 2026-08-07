"""paired_bootstrap_discriminator.py — four-crop z_e vs PCA paired bootstrap
(amendment 2026-08-06 §5, rerun after D1/D2 fix).

For each crop, align the learned and PCA LOEO arms by env_id and form the
per-environment paired difference of the discriminator statistic.  Because
pcc_ge / pcc_gm are encoder-independent, delta = pcc_gz - pcc_ge differs
between arms only through pcc_gz, so the paired diff of delta equals the
paired diff of pcc_gz; we compute it on delta for fidelity with amendment §5
("z_e 与 PCA 在同一环境上的 Δ 之差") and report pcc_gz means alongside.

Inference: environment-level bootstrap (resample envs with replacement,
recompute mean diff), percentile CI, two-sided p; Benjamini-Hochberg across
the four crops (protocol §6).  A secondary table repeats the comparison on
delta_fw (PCC vs Finlay-Wilkinson baseline, the M3 main metric).

Usage:
    python scripts/paired_bootstrap_discriminator.py [--dir data/t3] [--n-boot 3000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# per-crop (learned, pca) result files, relative to --dir
CROP_FILES = {
    "wheat": ("loe_fix_learned_wheat.parquet", "loe_fix_pca_wheat.parquet"),
    "corn": ("loe_fix_learned_corn.parquet", "loe_fix_pca_corn.parquet"),
    "oat": ("loe_oat_fix_learned.parquet", "loe_oat_fix_pca.parquet"),
    "barley": ("loe_barley_fix_learned.parquet", "loe_barley_fix_pca.parquet"),
}


def paired_diff(learned: pd.DataFrame, pca: pd.DataFrame, stat: str) -> pd.Series:
    """Per-environment paired difference (learned - pca) of `stat`."""
    l = learned.set_index("env_id")[stat]
    p = pca.set_index("env_id")[stat]
    common = l.index.intersection(p.index)
    d = (l.loc[common] - p.loc[common]).dropna()
    return d


def boot_p(d: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    """Mean diff, percentile CI, two-sided bootstrap p."""
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def bh_adjust(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Benjamini-Hochberg survival flags at FDR=alpha."""
    m = len(pvals)
    order = sorted(pvals, key=pvals.get)
    thresh = {c: alpha * (r + 1) / m for r, c in enumerate(order)}
    # step-up: reject all hypotheses with p <= max passing threshold
    passing = [c for c in order if pvals[c] <= thresh[c]]
    cutoff = max((thresh[c] for c in passing), default=0.0)
    return {c: pvals[c] <= cutoff for c in pvals}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "data/t3")
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Optional CSV of the primary table")
    args = ap.parse_args(argv)

    for stat, label in (("delta", "PRIMARY: paired diff of Delta(e) = PCC(Gz) - PCC(G+E)"),
                        ("delta_fw", "SECONDARY: paired diff of Delta_FW(e) = PCC(Gz) - PCC(FW)")):
        rows, pvals = [], {}
        for crop, (f_l, f_p) in CROP_FILES.items():
            p_l, p_p = args.dir / f_l, args.dir / f_p
            if not (p_l.exists() and p_p.exists()):
                print(f"[skip] {crop}: missing {p_l.name if not p_l.exists() else p_p.name}")
                continue
            learned, pca = pd.read_parquet(p_l), pd.read_parquet(p_p)
            d = paired_diff(learned, pca, stat)
            mean, lo, hi, p = boot_p(d.to_numpy(), args.n_boot, args.seed)
            pvals[crop] = p
            rows.append({
                "crop": crop, "n_env": len(d),
                "gz_learned": learned["pcc_gz"].mean(), "gz_pca": pca["pcc_gz"].mean(),
                "paired_diff": mean, "ci_lo": lo, "ci_hi": hi, "p": p,
            })
        if not rows:
            continue
        df = pd.DataFrame(rows)
        sig = bh_adjust(pvals)
        df["bh_sig"] = df["crop"].map(sig)
        print(f"\n=== {label} ===")
        print(df.round(4).to_string(index=False))
        if stat == "delta" and args.out_csv:
            df.to_csv(args.out_csv, index=False)
            print(f"[saved] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
