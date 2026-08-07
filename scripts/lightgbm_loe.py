"""lightgbm_loe.py — gradient-boosting baseline wall item (protocol §5),
LOEO-consistent, all four crops.

Per fold (held-out environment e):
  * features: flattened, TRAINING-fold-standardised environment stage
    features + genotype train-mean (leakage-safe target encoding, computed
    within the fold) + [corn only] top-32 marker PCs from the G2F dosage
    matrix (real-marker variant of the wall item)
  * LightGBM regressor, fixed 300 rounds, predicts the held-out env
  * metric: within-environment PCC on the raw scale -> directly comparable
    to pcc_gz / pcc_fw in the amendment §7.2 parquets

Output per crop: data/t3/loe_lgbm_{crop}.parquet + paired bootstrap vs the
learned model (pcc_gz) and FW (pcc_fw), BH across crops.

Usage:  python scripts/lightgbm_loe.py --crops barley,oat,corn,wheat --n-jobs 8
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

from loe_pilot import _pcc  # noqa: E402

RESULT_FILES = {
    "wheat": "loe_fix_learned_wheat.parquet",
    "corn": "loe_fix_learned_corn.parquet",
    "oat": "loe_oat_fix_learned.parquet",
    "barley": "loe_barley_fix_learned.parquet",
}


def load_items(crop: str, seed: int) -> list[dict]:
    if crop == "barley":
        from barley_loe import build_barley_items
        items = build_barley_items(ROOT / "data/t3/barley_items_wrbn.parquet",
                                   ROOT / "data/t3/trials_catalog_barley.parquet",
                                   ROOT / "data/t3/barley_env_features.pkl")
    elif crop == "oat":
        from oat_loe import build_oat_items
        items = build_oat_items(ROOT / "data/t3/oat_items_100.parquet",
                                ROOT / "data/t3/oat_env_features.pkl")
    elif crop == "wheat":
        from loe_pilot import load_wheat
        items, _ = load_wheat(800, seed, plot_cap=100)
    elif crop == "corn":
        from loe_pilot import load_corn
        items, _ = load_corn(400, seed, plot_cap=100)
    else:
        raise ValueError(crop)
    if crop in ("barley", "oat"):
        items = pd.DataFrame(items).groupby("env_id").head(100).to_dict("records")
    return items


def corn_marker_pcs(geno_ids: list[str], n_pc: int = 32) -> dict[str, np.ndarray]:
    """Top marker PCs from the G2F dosage matrix (corn only)."""
    g = pd.read_parquet(ROOT.parent / "nc/data/processed/g2f/genotype.parquet")
    g = g[g["genotype_id"].isin(geno_ids)]
    wide = g.pivot_table(index="genotype_id", columns="marker_id",
                         values="allele_dosage", aggfunc="mean").reindex(geno_ids)
    X = wide.to_numpy(dtype=np.float64)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    X = X - X.mean(0)
    sd = X.std(0) + 1e-8
    X = X / sd
    # PCA via Gram eigh (n_geno << n_markers)
    gram = X @ X.T / X.shape[1]
    evals, evecs = np.linalg.eigh(gram)
    pcs = evecs[:, ::-1][:, :n_pc] * np.sqrt(evals[::-1][:n_pc])
    return {g_: pcs[i] for i, g_ in enumerate(geno_ids)}


def run_crop(crop: str, args) -> pd.DataFrame:
    import lightgbm as lgb

    items = load_items(crop, args.seed)
    envs = sorted(set(i["env_id"] for i in items))
    genos = sorted(set(i["geno"] for i in items))
    print(f"[lgbm] {crop}: {len(items)} items, {len(envs)} envs, {len(genos)} genos", flush=True)

    marker_pc = corn_marker_pcs(genos) if crop == "corn" else None

    env_ids = np.array([i["env_id"] for i in items])
    geno_ids = np.array([i["geno"] for i in items])
    y = np.array([i["y"] for i in items], dtype=float)
    Xenv = np.stack([np.nan_to_num(i["x"]).reshape(-1) for i in items])
    if marker_pc is not None:
        Xpc = np.stack([marker_pc[g] for g in geno_ids])
        Xenv = np.hstack([Xenv, Xpc])

    rows = []
    for e in envs:
        tr, te = env_ids != e, env_ids == e
        if te.sum() < 3:
            continue
        mu, sd = Xenv[tr].mean(0), Xenv[tr].std(0) + 1e-6
        Xtr = (Xenv[tr] - mu) / sd
        Xte = (Xenv[te] - mu) / sd
        # leakage-safe genotype target encoding (training fold only)
        gm = pd.Series(y[tr]).groupby(geno_ids[tr]).mean()
        gtr = gm.reindex(geno_ids[tr]).fillna(y[tr].mean()).to_numpy()
        gte = gm.reindex(geno_ids[te]).fillna(y[tr].mean()).to_numpy()
        Xtr = np.column_stack([Xtr, gtr])
        Xte = np.column_stack([Xte, gte])
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8, n_jobs=args.n_jobs,
                                  deterministic=True, force_row_wise=True, verbose=-1)
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        rows.append({"crop": crop, "env_id": e, "pcc_lgbm": _pcc(y[te], pred)})
    df = pd.DataFrame(rows)
    print(f"[lgbm] {crop}: mean PCC = {df['pcc_lgbm'].mean():+.4f} (n={len(df)})", flush=True)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", default="barley,oat,corn,wheat")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--dir", type=Path, default=ROOT / "data/t3")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    pvals = {}
    rows = []
    for crop in args.crops.split(","):
        df = run_crop(crop, args)
        out = args.dir / f"loe_lgbm_{crop}.parquet"
        df.to_parquet(out, index=False)
        print(f"[lgbm] -> {out}", flush=True)

        ref_path = args.dir / RESULT_FILES[crop]
        if ref_path.exists():
            ref = pd.read_parquet(ref_path).set_index("env_id")
            j = df.set_index("env_id").join(ref[["pcc_gz", "pcc_fw"]], how="inner")
            row = {"crop": crop, "n_env": len(j), "pcc_lgbm": j["pcc_lgbm"].mean(),
                   "pcc_gz": j["pcc_gz"].mean(), "pcc_fw": j["pcc_fw"].mean()}
            for col in ("pcc_gz", "pcc_fw"):
                d = (j["pcc_lgbm"] - j[col]).dropna().to_numpy()
                boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(args.n_boot)])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
                row[f"vs_{col[4:]}_diff"] = float(d.mean())
                row[f"vs_{col[4:]}_lo"] = float(lo)
                row[f"vs_{col[4:]}_hi"] = float(hi)
                row[f"vs_{col[4:]}_p"] = float(min(p, 1.0))
                pvals[(crop, col)] = float(min(p, 1.0))
            rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        # BH across the 8 comparisons (4 crops x 2 refs)
        keys = sorted(pvals, key=pvals.get)
        m = len(keys)
        passing = [k for r, k in enumerate(keys) if pvals[k] <= 0.05 * (r + 1) / m]
        cutoff = max((0.05 * (keys.index(k) + 1) / m for k in passing), default=0.0)
        for col in ("gz", "fw"):
            out[f"vs_{col}_bh"] = out["crop"].map({c: pvals.get((c, f"pcc_{col}"), 1.0) <= cutoff for c in out["crop"]})
        pd.set_option("display.width", 240)
        print("\n=== LightGBM baseline vs amendment-§7.2 arms ===")
        print(out.round(4).to_string(index=False))
        out.to_csv(args.dir / "lgbm_baseline_summary.csv", index=False)
        print(f"[saved] {args.dir / 'lgbm_baseline_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
