"""gate_rich_features.py — richer pre-hoc gate signals (pillar A', route 1).

For every (crop, held env) fold, refit the FW/ridge pieces on the TRAINING
fold only (pure numpy, no GPU) and extract signals that are available BEFORE
touching the held-out environment's phenotypes:

  dist            k-NN feature-space distance to other envs (as before)
  n_geno          genotypes in the held env
  env_dev_hat     predicted env productivity index (FW ridge readout)
  abs_dev_hat     |env_dev_hat| -- how extreme the env is predicted to be
  leverage        ridge hat value of the held env's covariates -- how far
                  outside the training covariate distribution it lies
  ridge_r2        how well env productivity is predictable from covariates
                  in the training fold (aux-task-A quality proxy)
  slope_sd        sd of fitted FW slopes across genotypes -- reaction-norm
                  strength in the training fold
  slope_ident_frac  fraction of genotypes with an identified FW slope

Then: does sign(delta_fw) / delta_fw become predictable with these?
Logistic (AUC) + ridge on the continuous delta, 5-fold env-level CV,
per crop and pooled; gating payoff vs oracle and singles.

Usage:  python scripts/gate_rich_features.py [--rebuild]
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

from loe_pilot import FW_MIN_ENVS  # noqa: E402

LEARNED = {
    "wheat": "loe_fix_learned_wheat.parquet",
    "corn": "loe_fix_learned_corn.parquet",
    "oat": "loe_oat_fix_learned.parquet",
    "barley": "loe_barley_fix_learned.parquet",
}
FEAT_CACHE = ROOT / "data/t3/gate_features.parquet"


def load_crop_items(crop: str, seed: int = 0) -> list[dict]:
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


def fold_features(items: list[dict], held: str) -> dict | None:
    """Pre-hoc gate features for one held-out environment (training fold only)."""
    train = [i for i in items if i["env_id"] != held]
    held_items = [i for i in items if i["env_id"] == held]
    if not train or len(held_items) < 3:
        return None

    train_env_mean: dict[str, list] = {}
    for i in train:
        train_env_mean.setdefault(i["env_id"], []).append(i["y"])
    train_env_mean = {e: float(np.mean(v)) for e, v in train_env_mean.items()}
    global_mean = float(np.mean([i["y"] for i in train]))

    xs = np.stack([i["x"] for i in train])
    x_mean = np.nanmean(xs, axis=(0, 1))
    x_std = np.nanstd(xs, axis=(0, 1)) + 1e-6

    env_ids = sorted(train_env_mean)
    env_dev = {e: train_env_mean[e] - global_mean for e in env_ids}
    env_x = {}
    for i in train:
        env_x.setdefault(i["env_id"], i["x"])
    X = np.stack([((np.nan_to_num(env_x[e]) - x_mean) / x_std).ravel() for e in env_ids])
    y = np.array([env_dev[e] for e in env_ids], dtype=np.float64)
    Xc_mean = X.mean(0)
    Xc = X - Xc_mean
    d = Xc.shape[1]
    XtX_lam = Xc.T @ Xc + 1.0 * np.eye(d)
    w = np.linalg.solve(XtX_lam, Xc.T @ y)
    held_x = ((np.nan_to_num(held_items[0]["x"]) - x_mean) / x_std).ravel()
    env_dev_hat = float((held_x - Xc_mean) @ w)

    # ridge hat-value (leverage) of the held env's covariates
    hx = held_x - Xc_mean
    leverage = float(hx @ np.linalg.solve(XtX_lam, hx))

    # training-fold fit quality of the env-index ridge
    y_hat_tr = Xc @ w
    ss = 1.0 - float(np.sum((y - y_hat_tr) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))
    ridge_r2 = max(ss, -1.0)

    # FW slope heterogeneity across genotypes (reaction-norm strength)
    by_geno: dict = {}
    for i in train:
        by_geno.setdefault(i["geno"], {}).setdefault(i["env_id"], []).append(i["y"])
    slopes = []
    n_ident = 0
    for g, per_env in by_geno.items():
        ex = np.array([env_dev[e] for e in per_env], dtype=np.float64)
        if len(ex) >= FW_MIN_ENVS and float(np.ptp(ex)) > 1e-9:
            ey = np.array([float(np.mean(v)) for v in per_env.values()], dtype=np.float64)
            slopes.append(float(np.polyfit(ex, ey, 1)[0]))
            n_ident += 1
    slope_sd = float(np.std(slopes)) if slopes else 0.0

    return {
        "env_id": held,
        "n_geno": len(set(i["geno"] for i in held_items)),
        "env_dev_hat": env_dev_hat,
        "abs_dev_hat": abs(env_dev_hat),
        "leverage": leverage,
        "ridge_r2": ridge_r2,
        "slope_sd": slope_sd,
        "slope_ident_frac": n_ident / max(len(by_geno), 1),
    }


def build_features(crops: list[str], seed: int) -> pd.DataFrame:
    from delta_fw_dist_discriminator import env_dist

    frames = []
    for crop in crops:
        items = load_crop_items(crop, seed)
        envs = sorted(set(i["env_id"] for i in items))
        print(f"[gate_feat] {crop}: {len(envs)} envs", flush=True)
        dist = env_dist(items)
        rows = []
        for held in envs:
            f = fold_features(items, held)
            if f:
                f["dist"] = dist[held]
                f["crop"] = crop
                rows.append(f)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


FEATURES = ["dist", "n_geno", "env_dev_hat", "abs_dev_hat", "leverage",
            "ridge_r2", "slope_sd", "slope_ident_frac"]


def analyze(df: pd.DataFrame, seed: int) -> None:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    df["pcc_oracle"] = df[["pcc_gz", "pcc_fw"]].max(axis=1)

    def run_block(name: str, g: pd.DataFrame, X_cols: list[str]):
        y_sign = (g["delta_fw"] > 0).astype(int).to_numpy()
        if y_sign.mean() in (0.0, 1.0) or len(g) < 40:
            print(f"  {name}: degenerate (n={len(g)})")
            return None
        X = StandardScaler().fit_transform(g[X_cols].to_numpy())
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        proba = cross_val_predict(LogisticRegression(max_iter=2000), X, y_sign,
                                  cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y_sign, proba)
        # continuous delta regression -> gate on predicted sign
        rdg = cross_val_predict(Ridge(), X, g["delta_fw"].to_numpy(),
                                cv=KFold(5, shuffle=True, random_state=seed))
        gated_cls = np.where(proba > 0.5, g["pcc_gz"], g["pcc_fw"]).mean()
        gated_reg = np.where(rdg > 0, g["pcc_gz"], g["pcc_fw"]).mean()
        best_single = max(g["pcc_gz"].mean(), g["pcc_fw"].mean())
        print(f"  {name}: AUC={auc:.3f} | gated(cls)={gated_cls:+.4f} gated(reg)={gated_reg:+.4f} "
              f"| best_single={best_single:+.4f} oracle={g['pcc_oracle'].mean():+.4f} "
              f"| gain={max(gated_cls, gated_reg) - best_single:+.4f}")
        return auc

    print("\n=== rich-feature gate, per crop ===")
    for crop, g in df.groupby("crop"):
        run_block(crop, g, FEATURES)
    print("\n=== rich-feature gate, pooled (+crop dummies) ===")
    d = pd.concat([df, pd.get_dummies(df["crop"])], axis=1)
    crop_cols = list(pd.get_dummies(df["crop"]).columns)
    run_block("pooled", d, FEATURES + crop_cols)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild", action="store_true", help="Recompute fold features (else read cache)")
    ap.add_argument("--crops", default="wheat,corn,oat,barley")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    crops = args.crops.split(",")
    if not args.rebuild and FEAT_CACHE.exists():
        feat = pd.read_parquet(FEAT_CACHE)
    else:
        feat = build_features(crops, args.seed)
        feat.to_parquet(FEAT_CACHE, index=False)
        print(f"[gate_feat] cached -> {FEAT_CACHE}")

    res = pd.concat([pd.read_parquet(ROOT / "data/t3" / LEARNED[c]) for c in crops],
                    ignore_index=True)
    df = feat.merge(res[["env_id", "pcc_gz", "pcc_fw", "delta_fw"]], on="env_id", how="inner")
    df = df.dropna(subset=["delta_fw"] + FEATURES)
    print(f"[gate_feat] merged {len(df)} envs")
    analyze(df, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
