"""corn_gblup.py — marker-based GBLUP baseline for the corn LOEO cohort
(protocol §5 baseline wall, item 1; strengthens the negative-conclusion
baseline set after the D1/D2-fix rerun).

Design (leakage-safe, mirrors the LOEO folds exactly):
  * cohort: identical to amendment §7.2 rerun (load_corn(400, seed=0,
    plot_cap=100) -> 269 envs)
  * markers: G2F genotype.parquet (long: genotype_id, marker_id,
    allele_dosage) pivoted to a dosage matrix; VanRaden G = ZZ'/2sum(p(1-p))
  * per fold: training envs only.  Adjust y by TRAINING-fold env means, then
    solve the MME  [Z'Z + theta*G^-1] g = Z'y_adj  for genomic values.
    Held-out env prediction: y_hat = mu_train + g_geno  (environment effect
    unobservable -- same handicap as the genotype-mean baseline, so the
    comparison isolates marker information).
  * theta = (1-h2)/h2 with h2 fixed at 0.5; sensitivity h2 in {0.3, 0.7}
    reported alongside (variance-component estimation per fold would be
    leakage-safe too but adds little here -- noted as a limitation).

Output: per-env PCC(GBLUP) parquet + paired bootstrap vs the genotype-mean
baseline (pcc_gm from the fixed rerun), and vs PCC(Gz) for context.

Usage:  python scripts/corn_gblup.py [--h2 0.5] [--out ...]
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

from loe_pilot import load_corn, _pcc  # noqa: E402

G2F_GENO = ROOT.parent / "nc/data/processed/g2f/genotype.parquet"


def build_grm(geno_ids: list[str]) -> np.ndarray:
    """VanRaden genomic relationship matrix for the given genotypes."""
    g = pd.read_parquet(G2F_GENO)
    g = g[g["genotype_id"].isin(geno_ids)]
    wide = g.pivot_table(index="genotype_id", columns="marker_id",
                         values="allele_dosage", aggfunc="mean")
    wide = wide.reindex(geno_ids)
    X = wide.to_numpy(dtype=np.float64)  # (n_geno, n_marker)
    # mean-impute missing dosages per marker (rare in this panel)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    p = X.mean(axis=0) / 2.0
    denom = 2.0 * np.sum(p * (1.0 - p))
    Z = X - 2.0 * p
    G = (Z @ Z.T) / max(denom, 1e-12)
    return G


def gblup_fold(y_adj: np.ndarray, geno_pos: np.ndarray, G_inv: np.ndarray,
               theta: float) -> np.ndarray:
    """Solve [Z'Z + theta*G^-1] g = Z'y_adj (genotype-level incidence)."""
    n = G_inv.shape[0]
    # Z'Z is diagonal (one genotype per record): counts on the diagonal
    lhs = np.diag(np.bincount(geno_pos, minlength=n).astype(float))
    rhs = np.zeros(n)
    np.add.at(rhs, geno_pos, y_adj)
    lhs += theta * G_inv
    return np.linalg.solve(lhs, rhs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--h2-sensitivity", default="0.3,0.7")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/t3/loe_corn_gblup.parquet")
    ap.add_argument("--ref-results", type=Path, default=ROOT / "data/t3/loe_fix_learned_corn.parquet")
    ap.add_argument("--n-boot", type=int, default=3000)
    args = ap.parse_args(argv)

    items, envs = load_corn(400, args.seed, plot_cap=100)
    print(f"[gblup] corn cohort: {len(items)} items, {len(envs)} envs", flush=True)
    genos = sorted(set(i["geno"] for i in items))
    gpos = {g: i for i, g in enumerate(genos)}
    print(f"[gblup] {len(genos)} genotypes; building GRM ...", flush=True)
    G = build_grm(genos)
    # ridge-regularised inverse for numerical safety
    G_inv = np.linalg.inv(G + 1e-4 * np.eye(len(genos)))
    print("[gblup] GRM done", flush=True)

    env_of = np.array([i["env_id"] for i in items])
    geno_of = np.array([gpos[i["geno"]] for i in items])
    y = np.array([i["y"] for i in items], dtype=float)

    def run_loeo(h2: float) -> dict[str, float]:
        theta = (1.0 - h2) / h2
        out = {}
        for e in envs:
            tr = env_of != e
            te = ~tr
            if te.sum() < 3:
                continue
            env_mean = {}
            for env_t in pd.unique(env_of[tr]):
                m = tr & (env_of == env_t)
                env_mean[env_t] = y[m].mean()
            mu = y[tr].mean()
            y_adj = y[tr] - np.array([env_mean[v] for v in env_of[tr]])
            g_hat = gblup_fold(y_adj, geno_of[tr], G_inv, theta)
            pred = mu + g_hat[geno_of[te]]
            out[e] = _pcc(y[te], pred)
        return out

    pcc_by_h2 = {h2: run_loeo(h2) for h2 in [args.h2] + [float(v) for v in args.h2_sensitivity.split(",")]}
    main_pcc = pcc_by_h2[args.h2]
    for h2, d in pcc_by_h2.items():
        print(f"[gblup] h2={h2}: mean within-env PCC = {np.nanmean(list(d.values())):+.4f} (n={len(d)})",
              flush=True)

    df = pd.DataFrame([{"crop": "corn", "env_id": e, "pcc_gblup": v} for e, v in main_pcc.items()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[gblup] -> {args.out}")

    # paired bootstrap vs genotype-mean baseline and vs learned Gz
    if args.ref_results.exists():
        ref = pd.read_parquet(args.ref_results).set_index("env_id")
        common = df.set_index("env_id").join(ref[["pcc_gm", "pcc_gz"]], how="inner")
        rng = np.random.default_rng(0)
        for col in ("pcc_gm", "pcc_gz"):
            d = (common["pcc_gblup"] - common[col]).dropna().to_numpy()
            boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(args.n_boot)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
            print(f"[gblup] paired diff GBLUP - {col}: {d.mean():+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}] p={min(p,1):.4f} (n={len(d)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
