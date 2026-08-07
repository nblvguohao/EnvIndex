"""negative_transfer_predictor.py — can we predict, for a held-out environment,
whether the neural correction will HELP or HURT (pillar A', 2026-08-07)?

Gate science: per environment we know the sign of delta_fw (Gz minus FW) and
delta vs the PCA arm from the amendment §7.2 rerun.  If the sign is
predictable from quantities available BEFORE fitting on that environment
(dist(e) to training envs, env feature summary stats, crop, n_genotypes),
then a per-environment reliability gate is viable -- the P2 deliverable that
also feeds the SRG-GxE (P1 V3) gating design.

Analyses (all CPU, no retraining):
  1. oracle ceiling: per-env best-of {Gz, FW} minus each single model
  2. predictability of sign(delta_fw): logistic regression on
     [dist(e), crop, n_geno, env-feature PCs], environment-level CV, AUC
  3. gating payoff: predicted gate vs oracle vs singles

Usage:  python scripts/negative_transfer_predictor.py [--dir data/t3]
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

from delta_dist_curve import _items_for_result_envs  # noqa: E402
from delta_fw_dist_discriminator import env_dist  # noqa: E402

LEARNED = {
    "wheat": "loe_fix_learned_wheat.parquet",
    "corn": "loe_fix_learned_corn.parquet",
    "oat": "loe_oat_fix_learned.parquet",
    "barley": "loe_barley_fix_learned.parquet",
}


def build_frame(args) -> pd.DataFrame:
    frames = []
    for crop, fname in LEARNED.items():
        path = args.dir / fname
        if not path.exists():
            print(f"[skip] {crop}: missing {fname}")
            continue
        df = pd.read_parquet(path)
        frames.append(df)
    res = pd.concat(frames, ignore_index=True)

    items_by_crop = _items_for_result_envs(res, argparse.Namespace())
    dist = {}
    for crop, items in items_by_crop.items():
        dist.update(env_dist(items))
    res["dist"] = res["env_id"].map(dist)

    n_geno = {}
    for crop, items in items_by_crop.items():
        counts = pd.Series([i["env_id"] for i in items]).value_counts()
        n_geno.update(counts.to_dict())
    res["n_geno_env"] = res["env_id"].map(n_geno)
    return res.dropna(subset=["dist", "delta_fw"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=ROOT / "data/t3")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "data/t3/negative_transfer_gate.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    df = build_frame(args)
    print(f"[gate] {len(df)} envs across {df['crop'].nunique()} crops")

    # ---- 1. oracle ceiling ------------------------------------------------
    df["pcc_oracle"] = df[["pcc_gz", "pcc_fw"]].max(axis=1)
    print("\n=== 1. oracle ceiling (upper bound of any gate) ===")
    rows = []
    for crop, g in df.groupby("crop"):
        rows.append({
            "crop": crop, "n_env": len(g),
            "pcc_gz": g["pcc_gz"].mean(), "pcc_fw": g["pcc_fw"].mean(),
            "pcc_oracle": g["pcc_oracle"].mean(),
            "oracle_gain_vs_gz": g["pcc_oracle"].mean() - g["pcc_gz"].mean(),
            "oracle_gain_vs_fw": g["pcc_oracle"].mean() - g["pcc_fw"].mean(),
            "pct_fw_wins": (g["delta_fw"] < 0).mean(),
        })
    orc = pd.DataFrame(rows)
    print(orc.round(4).to_string(index=False))

    # ---- 2. sign predictability -------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    y_sign = (df["delta_fw"] > 0).astype(int).to_numpy()
    X = np.column_stack([
        df["dist"].to_numpy(),
        np.log10(df["n_geno_env"].to_numpy()),
        pd.get_dummies(df["crop"]).to_numpy(),
    ])
    X = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    proba = cross_val_predict(LogisticRegression(max_iter=1000), X, y_sign,
                              cv=cv, method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = roc_auc_score(y_sign, proba)
    acc = accuracy_score(y_sign, proba > 0.5)
    base = y_sign.mean()
    print(f"\n=== 2. sign(delta_fw) predictability (5-fold env-level CV) ===")
    print(f"AUC = {auc:.3f} | acc = {acc:.3f} | base rate (P(Gz wins)) = {base:.3f}")

    # per-crop AUC (within-crop signal only, dist is crop-specific scale)
    for crop, g in df.groupby("crop"):
        if len(g) < 40 or g["delta_fw"].gt(0).mean() in (0.0, 1.0):
            continue
        yc = (g["delta_fw"] > 0).astype(int).to_numpy()
        Xc = np.column_stack([g["dist"].to_numpy(), np.log10(g["n_geno_env"].to_numpy())])
        Xc = StandardScaler().fit_transform(Xc)
        pc = cross_val_predict(LogisticRegression(max_iter=1000), Xc, yc,
                               cv=5, method="predict_proba")[:, 1]
        print(f"  {crop}: AUC = {roc_auc_score(yc, pc):.3f} (n={len(g)}, base={yc.mean():.2f})")

    # ---- 3. gating payoff ---------------------------------------------------
    df["gate_score"] = proba
    df["pcc_gated"] = np.where(df["gate_score"] > 0.5, df["pcc_gz"], df["pcc_fw"])
    print("\n=== 3. gating payoff (predicted gate vs singles vs oracle) ===")
    rows = []
    for crop, g in df.groupby("crop"):
        rows.append({
            "crop": crop,
            "pcc_gz": g["pcc_gz"].mean(), "pcc_fw": g["pcc_fw"].mean(),
            "pcc_gated": g["pcc_gated"].mean(), "pcc_oracle": g["pcc_oracle"].mean(),
            "gate_gain_vs_best_single": g["pcc_gated"].mean() - max(g["pcc_gz"].mean(), g["pcc_fw"].mean()),
        })
    gate = pd.DataFrame(rows)
    print(gate.round(4).to_string(index=False))

    out = pd.concat([orc.assign(table="oracle"), gate.assign(table="gate")])
    out.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
