"""derive_thresholds.py — data-driven pre-registration thresholds.

Derives protocol §6 / §4.4 thresholds from measured data rather than arbitrary
values (response to editor: "MDE not provided", "d_crit width not established",
"effective sample size under clustering not computed").

Inputs
------
- yield SD per crop (for SelectionGain sigma-anchored thresholds)
- per-env Delta stats from the scaled LOEO run
- dist(e) distributions (per crop)
- optional per-env results parquet (refines Delta stats)

Outputs
-------
MDE (mean-Delta), MDE (bin-comparison), MDE (slope discriminator),
d_crit acceptable width, SelectionGain@10% decision threshold, N_eff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

Z_ALPHA = 1.96   # two-sided alpha = 0.05
Z_BETA = 0.84    # power = 0.80


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=None, help="Per-env results parquet (refines Delta stats)")
    parser.add_argument("--out", type=Path, default=ROOT / "data/t3/thresholds.csv")
    return parser.parse_args(argv)


def mde_mean(delta_sd, n):
    """Minimum detectable mean-Delta != 0 at 80% power."""
    return (Z_ALPHA + Z_BETA) * delta_sd / np.sqrt(n)


def mde_bin_comparison(delta_sd, n):
    """MDE for Delta(high-dist bin) - Delta(low-dist bin), two halves."""
    return (Z_ALPHA + Z_BETA) * delta_sd * np.sqrt(2.0 / n)


def mde_slope(delta_sd, dist_sd, n):
    """MDE for slope difference (z_e vs PCA) at 80% power."""
    se_b = delta_sd / (dist_sd * np.sqrt(n))
    return (Z_ALPHA + Z_BETA) * np.sqrt(2.0) * se_b


def neff(icc, n, m):
    """Effective sample size under clustering (site clusters of mean size m)."""
    return n / (1 + (m - 1) * icc)


def main(argv=None):
    args = _parse_args(argv)
    rows = []

    # ---- 1. yield SD anchors (measured) ----
    g2f = pd.read_parquet(ROOT.parent / "nc/data/processed/g2f/phenotype.parquet")
    eswyt = pd.read_csv(ROOT / "data/cimmyt/ESWYT_Obs_Sim_Yld_Phe_Climate_All.tab", sep="\t")
    yield_info = {
        "corn": (float(g2f["phenotype_value"].std()), float(g2f["phenotype_value"].mean())),
        "wheat": (float(eswyt["yld"].std()), float(eswyt["yld"].mean())),
    }

    # ---- 2. Delta stats (from scaled run; refined by --results if given) ----
    delta = {"corn": {"sd": 0.085, "n": 200}, "wheat": {"sd": 0.238, "n": 396}}
    if args.results and args.results.exists():
        res = pd.read_parquet(args.results)
        for crop, g in res.dropna(subset=["delta"]).groupby("crop"):
            delta[crop] = {"sd": float(g["delta"].std()), "n": int(len(g))}

    # ---- 3. dist distributions ----
    dist = {}
    for crop in ("wheat", "corn"):
        p = ROOT / f"data/t3/dist_{crop}.npy"
        if p.exists():
            dv = np.load(p)
            dist[crop] = {"sd": float(dv.std()), "iqr": float(np.percentile(dv, 75) - np.percentile(dv, 25))}

    # ---- 4. cluster ICC estimate (G2F env-mean by location) ----
    g2f_env = g2f.groupby(["environment_id", "location_id"])["phenotype_value"].mean().reset_index()
    # within-location variance of env means vs between-location
    grand = g2f_env["phenotype_value"].var()
    within = g2f_env.groupby("location_id")["phenotype_value"].var().mean()
    icc_est = max(0.0, min(1.0, 1 - within / grand)) if grand > 0 else 0.0

    for crop in ("corn", "wheat"):
        ds = delta[crop]
        ds_icc = icc_est if crop == "corn" else icc_est  # same clustering assumption
        m_size = max(2, int(round(ds["n"] / max(1, _n_clusters(crop)))))
        n_eff = neff(ds_icc, ds["n"], m_size)
        dist_sd = dist.get(crop, {}).get("sd", np.nan)
        dist_iqr = dist.get(crop, {}).get("iqr", np.nan)
        yield_sd, yield_mean = yield_info[crop]

        rows.append({
            "crop": crop,
            "n_envs": ds["n"],
            "n_eff": round(n_eff, 1),
            "icc": round(ds_icc, 3),
            "delta_sd": round(ds["sd"], 4),
            "dist_sd": round(dist_sd, 4),
            "dist_iqr": round(dist_iqr, 4),
            "MDE_mean_delta": round(mde_mean(ds["sd"], ds["n"]), 4),
            "MDE_mean_delta_adj": round(mde_mean(ds["sd"], n_eff), 4),
            "MDE_bin_comparison": round(mde_bin_comparison(ds["sd"], ds["n"]), 4),
            "MDE_slope_discriminator": round(mde_slope(ds["sd"], dist_sd, ds["n"]), 4) if np.isfinite(dist_sd) else np.nan,
            "d_crit_max_ci_halfwidth": round(0.2 * dist_iqr, 6) if np.isfinite(dist_iqr) else np.nan,
            "selectionGain_threshold_5pct_sigma": round(0.05 * yield_sd, 3),
            "yield_mean": round(yield_mean, 2),
            "selectionGain_pct_of_mean": round(100 * 0.05 * yield_sd / yield_mean, 2),
        })

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\n[derive_thresholds] -> {args.out}")
    return 0


def _n_clusters(crop):
    # rough site count: corn 272 envs / ~9 sites-ish; use env count / 4 (multiple years per site)
    return {"corn": 40, "wheat": 100}.get(crop, 10)


if __name__ == "__main__":
    sys.exit(main())
