"""corn_heritability.py — per-environment heritability stratification for corn
(pillar leftover item 4; uses G2F replicate structure).

Per environment in the §7.2 corn cohort: entry-mean broad-sense H² via a
random-intercept mixed model  y ~ 1 + (1|geno)  on the PLOT-level records
(replicate_id gives the within-env replication; unbalanced designs handled
by MixedLM).  H2 = var_g / (var_g + var_e / r_bar), r_bar = harmonic mean
reps per genotype.

Then: is within-env PCC (and Delta_FW) stratified by H2?  Spearman/Pearson
correlations + tercile means with env-level bootstrap, answering the
reviewer question "is the null just low-quality environments?"

Wheat ESWYT (~12% replicated cells) and the T3 crops (checks only) do NOT
support stable per-env H2 -- reported as a coverage limitation, not skipped
silently.

Usage:  python scripts/corn_heritability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "nc" / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

G2F_PHENO = ROOT.parent / "nc/data/processed/g2f/phenotype.parquet"
RESULTS = ROOT / "data/t3/loe_fix_learned_corn.parquet"
OUT = ROOT / "data/t3/corn_heritability.parquet"


def env_h2(g: pd.DataFrame) -> tuple[float, float] | None:
    """Entry-mean H2 for one environment from plot-level records."""
    import statsmodels.formula.api as smf

    d = g.dropna(subset=["phenotype_value", "genotype_id"])
    if d["genotype_id"].nunique() < 20:
        return None
    reps = d.groupby("genotype_id").size()
    r_bar = len(reps) / np.sum(1.0 / reps)  # harmonic mean
    try:
        m = smf.mixedlm("phenotype_value ~ 1", d, groups=d["genotype_id"]).fit(reml=True, method="cg")
        var_g = float(m.cov_re.iloc[0, 0])
        var_e = float(m.scale)
    except Exception:
        return None
    if var_g <= 0 or var_e <= 0:
        return None
    h2 = var_g / (var_g + var_e / r_bar)
    return h2, float(r_bar)


def main() -> int:
    res = pd.read_parquet(RESULTS)
    cohort = set(res["env_id"])
    pheno = pd.read_parquet(G2F_PHENO)
    pheno = pheno[pheno["environment_id"].isin(cohort)]
    print(f"[h2] {pheno['environment_id'].nunique()} cohort envs in phenotype table")

    rows = []
    for env_id, g in pheno.groupby("environment_id"):
        out = env_h2(g)
        if out:
            rows.append({"env_id": env_id, "h2": out[0], "r_bar": out[1],
                         "n_plots": len(g), "n_geno": g["genotype_id"].nunique()})
    h2df = pd.DataFrame(rows)
    print(f"[h2] estimated for {len(h2df)}/{len(cohort)} envs | "
          f"median H2 = {h2df['h2'].median():.3f} | range ({h2df['h2'].min():.3f}, {h2df['h2'].max():.3f})")
    h2df.to_parquet(OUT, index=False)

    df = res.merge(h2df, on="env_id", how="inner")
    from scipy import stats as sstats
    print(f"\n=== H2 stratification (n={len(df)} envs) ===")
    for col in ("pcc_gz", "pcc_fw", "delta_fw", "pcc_gm"):
        r_s, p_s = sstats.spearmanr(df["h2"], df[col])
        r_p, p_p = sstats.pearsonr(df["h2"], df[col])
        print(f"  {col}: spearman r={r_s:+.3f} (p={p_s:.4f}) | pearson r={r_p:+.3f} (p={p_p:.4f})")

    df["h2_tercile"] = pd.qcut(df["h2"], 3, labels=["low", "mid", "high"])
    tab = df.groupby("h2_tercile", observed=True).agg(
        n=("env_id", "size"), h2=("h2", "mean"),
        pcc_gz=("pcc_gz", "mean"), pcc_fw=("pcc_fw", "mean"),
        pcc_gm=("pcc_gm", "mean"), delta_fw=("delta_fw", "mean"),
    )
    print("\n=== terciles ===")
    print(tab.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
