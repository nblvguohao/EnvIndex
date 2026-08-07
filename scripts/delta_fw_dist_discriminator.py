"""delta_fw_dist_discriminator.py — H2 discriminator under the M3 main metric
(amendment 2026-08-06 §7.2 follow-up, post-D1/D2-fix rerun).

Recomputes the Δ–dist relationship with Δ_FW(e) = PCC(G∘z,e) − PCC(FW,e)
on the fixed four-crop LOEO results, using the same dist(e) definition as
delta_dist_curve.py (standardized Euclidean 5-NN in the R1 feature space,
per crop).

Statistics per crop (discriminator preregistration 2026-08-05 §2.2, as
corrected by amendment M2 — point estimates WITH environment-level
bootstrap CIs):
  1. per-environment Pearson and Spearman corr(Δ_FW, dist) with p-values
  2. half-split contrast: mean Δ_FW in high-dist half − low-dist half
  3. endpoint-bin contrast: top octile-bin mean − bottom octile-bin mean
BH correction across crops (m=4) within each family.  Preregistered MDEs:
wheat 0.047, corn 0.024 (half-split); oat/barley have no preregistered MDE.

Usage:  python scripts/delta_fw_dist_discriminator.py [--n-boot 3000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "nc" / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from delta_dist_curve import _items_for_result_envs  # noqa: E402

CROP_FILES = {
    "wheat": "loe_fix_learned_wheat.parquet",
    "corn": "loe_fix_learned_corn.parquet",
    "oat": "loe_oat_fix_learned.parquet",
    "barley": "loe_barley_fix_learned.parquet",
}
MDE_HALF = {"wheat": 0.047, "corn": 0.024}  # prereg §2.2; oat/barley: none


def env_dist(items: list[dict]) -> dict[str, float]:
    """Standardized Euclidean 5-NN distance in the per-crop feature space
    (identical to delta_dist_curve.main)."""
    env_mats = {}
    for it in items:
        env_mats.setdefault(it["env_id"], it["x"])
    env_ids = sorted(env_mats)
    flat = np.stack([np.nan_to_num(env_mats[e]).flatten() for e in env_ids])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    zf = (flat - mu) / sd
    d = np.linalg.norm(zf[:, None, :] - zf[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    knn = np.sort(d, axis=1)[:, :5].mean(1)
    return dict(zip(env_ids, knn))


def boot_stat(d: np.ndarray, dist: np.ndarray, fn, n_boot: int, seed: int):
    """Bootstrap a contrast over environments; return (point, lo, hi, p)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(fn(d[idx], dist[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(fn(d, dist)), float(lo), float(hi), float(min(p, 1.0))


def half_split(d, dist):
    hi = dist >= np.median(dist)
    return d[hi].mean() - d[~hi].mean()


def endpoint_bin(d, dist, n_bins=8):
    edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    first = d[dist <= edges[1]]
    last = d[dist >= edges[-2]]
    return last.mean() - first.mean()


def bh_flags(pvals: dict[str, float], alpha=0.05) -> dict[str, bool]:
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    passing = [c for r, c in enumerate(order) if pvals[c] <= alpha * (r + 1) / m]
    cutoff = max((alpha * (order.index(c) + 1) / m for c in passing), default=0.0)
    return {c: pvals[c] <= cutoff for c in pvals}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "data/t3")
    ap.add_argument("--stat", default="delta_fw")
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", type=Path, default=ROOT / "data/t3/delta_fw_dist_discriminator.csv")
    args = ap.parse_args(argv)

    # merge all crops into one results frame (delta_dist_curve expects a `crop` column)
    frames = []
    for crop, fname in CROP_FILES.items():
        path = args.dir / fname
        if not path.exists():
            print(f"[skip] {crop}: missing {fname}")
            continue
        frames.append(pd.read_parquet(path))
    res = pd.concat(frames, ignore_index=True)

    items_by_crop = _items_for_result_envs(res, argparse.Namespace())

    rows = []
    p_families: dict[str, dict[str, float]] = {"pearson": {}, "half": {}, "endpoint": {}}
    for crop, items in items_by_crop.items():
        dist = env_dist(items)
        g = res[res["crop"] == crop].copy()
        g["dist"] = g["env_id"].map(dist)
        g = g.dropna(subset=["dist", args.stat])
        d, x = g[args.stat].to_numpy(), g["dist"].to_numpy()

        pr, pp = sstats.pearsonr(d, x)
        sr, sp = sstats.spearmanr(d, x)
        h_pt, h_lo, h_hi, h_p = boot_stat(d, x, half_split, args.n_boot, args.seed)
        e_pt, e_lo, e_hi, e_p = boot_stat(d, x, endpoint_bin, args.n_boot, args.seed)
        p_families["pearson"][crop] = pp
        p_families["half"][crop] = h_p
        p_families["endpoint"][crop] = e_p
        rows.append({
            "crop": crop, "n_env": len(g),
            "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp,
            "half_diff": h_pt, "half_ci_lo": h_lo, "half_ci_hi": h_hi, "half_p": h_p,
            "endpoint_diff": e_pt, "endpoint_ci_lo": e_lo, "endpoint_ci_hi": e_hi, "endpoint_p": e_p,
            "mde_half": MDE_HALF.get(crop),
        })

    df = pd.DataFrame(rows)
    for fam, col in (("pearson", "pearson_p"), ("half", "half_p"), ("endpoint", "endpoint_p")):
        flags = bh_flags(p_families[fam])
        df[f"{fam}_bh_sig"] = df["crop"].map(flags)

    pd.set_option("display.width", 220)
    print(f"\n=== Δ–dist discriminator under {args.stat} (n_boot={args.n_boot}) ===")
    print(df.round(4).to_string(index=False))
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
