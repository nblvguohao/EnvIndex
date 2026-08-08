"""gate_bootstrap_significance.py -- bootstrap significance test for the
rich-feature reliability gate's deployment gain (pillar A', follow-up to
gate_rich_features.py).

gate_rich_features.py reported AUC 0.70-0.77 (oat/barley) for predicting
sign(delta_fw) from pre-hoc fold features, and a point-estimate "gain" =
gated mean PCC - best-single-arm mean PCC, but without a significance test
(flagged in reports/lit_gap_analysis_2026-08-08.md as the one open item).

This script bootstraps that gain (environment-level resampling, matching the
convention in paired_bootstrap_discriminator.py: resample envs with
replacement, percentile CI, two-sided bootstrap p, BH correction across the
four crops) to determine whether the gate provides a statistically
defensible improvement over "always use whichever of {G-dot-z, FW} is
better on average."

The two gate variants (classification-threshold gate on the logistic
out-of-fold probability, regression-sign gate on the ridge out-of-fold
prediction) are tested SEPARATELY rather than taking max(gain_cls, gain_reg)
per bootstrap draw -- picking the better of two correlated statistics on
every resample would inject winner's-curse bias into the CI. Also reports
bootstrap CI on the classification AUC itself.

Usage:
    python scripts/gate_bootstrap_significance.py [--n-boot 3000] [--rebuild]
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

from gate_rich_features import FEATURES, FEAT_CACHE, LEARNED, build_features  # noqa: E402


def bh_adjust(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Benjamini-Hochberg survival flags at FDR=alpha (mirrors paired_bootstrap_discriminator.py)."""
    m = len(pvals)
    order = sorted(pvals, key=pvals.get)
    thresh = {c: alpha * (r + 1) / m for r, c in enumerate(order)}
    passing = [c for c in order if pvals[c] <= thresh[c]]
    cutoff = max((thresh[c] for c in passing), default=0.0)
    return {c: pvals[c] <= cutoff for c in pvals}


def fold_predictions(g: pd.DataFrame, X_cols: list[str], seed: int) -> pd.DataFrame | None:
    """Out-of-fold logistic proba + ridge prediction for one crop (or pooled) group."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    y_sign = (g["delta_fw"] > 0).astype(int).to_numpy()
    if y_sign.mean() in (0.0, 1.0) or len(g) < 40:
        return None
    X = StandardScaler().fit_transform(g[X_cols].to_numpy())
    cv_cls = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = cross_val_predict(LogisticRegression(max_iter=2000), X, y_sign,
                              cv=cv_cls, method="predict_proba")[:, 1]
    cv_reg = KFold(5, shuffle=True, random_state=seed)
    rdg = cross_val_predict(Ridge(), X, g["delta_fw"].to_numpy(), cv=cv_reg)

    out = g[["env_id", "pcc_gz", "pcc_fw", "delta_fw"]].copy()
    out["proba"] = proba
    out["rdg"] = rdg
    out["y_sign"] = y_sign
    return out.reset_index(drop=True)


def boot_gain(pcc_gz: np.ndarray, pcc_fw: np.ndarray, gated: np.ndarray,
             n_boot: int, seed: int) -> tuple[float, float, float, float]:
    """Bootstrap CI/p for gain = mean(gated_pcc) - max(mean(pcc_gz), mean(pcc_fw)).

    Both the gated arm and the best-single-arm baseline are recomputed from
    the SAME resampled indices each draw, so baseline-selection noise is
    propagated into the CI rather than fixed at the point estimate.
    """
    n = len(gated)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        best_single = max(pcc_gz[idx].mean(), pcc_fw[idx].mean())
        boots[b] = gated[idx].mean() - best_single
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    point = gated.mean() - max(pcc_gz.mean(), pcc_fw.mean())
    return float(point), float(lo), float(hi), float(min(p, 1.0))


def boot_auc(y_sign: np.ndarray, proba: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score
    n = len(y_sign)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y_sign[idx]
        if ys.mean() in (0.0, 1.0):
            continue
        boots.append(roc_auc_score(ys, proba[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(roc_auc_score(y_sign, proba)), float(lo), float(hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--crops", default="wheat,corn,oat,barley")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--out-csv", type=Path, default=ROOT / "data/t3/gate_bootstrap.csv")
    args = ap.parse_args(argv)

    crops = args.crops.split(",")
    if not args.rebuild and FEAT_CACHE.exists():
        feat = pd.read_parquet(FEAT_CACHE)
    else:
        feat = build_features(crops, args.seed)
        feat.to_parquet(FEAT_CACHE, index=False)

    res = pd.concat([pd.read_parquet(ROOT / "data/t3" / LEARNED[c]) for c in crops], ignore_index=True)
    df = feat.merge(res[["env_id", "pcc_gz", "pcc_fw", "delta_fw"]], on="env_id", how="inner")
    df = df.dropna(subset=["delta_fw"] + FEATURES)
    print(f"[gate_boot] merged {len(df)} envs, n_boot={args.n_boot}")

    groups: dict[str, pd.DataFrame] = {c: g for c, g in df.groupby("crop")}
    d_pooled = pd.concat([df, pd.get_dummies(df["crop"])], axis=1)
    crop_cols = list(pd.get_dummies(df["crop"]).columns)
    groups["pooled"] = d_pooled
    x_cols = {c: FEATURES for c in crops}
    x_cols["pooled"] = FEATURES + crop_cols

    rows = []
    pvals_cls: dict[str, float] = {}
    pvals_reg: dict[str, float] = {}
    for name, g in groups.items():
        pred = fold_predictions(g, x_cols[name], args.seed)
        if pred is None:
            print(f"[skip] {name}: degenerate or n<40")
            continue
        y_sign = pred["y_sign"].to_numpy()
        pcc_gz = pred["pcc_gz"].to_numpy()
        pcc_fw = pred["pcc_fw"].to_numpy()
        gated_cls = np.where(pred["proba"].to_numpy() > 0.5, pcc_gz, pcc_fw)
        gated_reg = np.where(pred["rdg"].to_numpy() > 0, pcc_gz, pcc_fw)

        auc, auc_lo, auc_hi = boot_auc(y_sign, pred["proba"].to_numpy(), args.n_boot, args.seed)
        pt_c, lo_c, hi_c, p_c = boot_gain(pcc_gz, pcc_fw, gated_cls, args.n_boot, args.seed)
        pt_r, lo_r, hi_r, p_r = boot_gain(pcc_gz, pcc_fw, gated_reg, args.n_boot, args.seed + 1)
        if name != "pooled":
            pvals_cls[name] = p_c
            pvals_reg[name] = p_r
        rows.append({
            "group": name, "n_env": len(pred),
            "auc": auc, "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
            "gain_cls": pt_c, "gain_cls_ci_lo": lo_c, "gain_cls_ci_hi": hi_c, "gain_cls_p": p_c,
            "gain_reg": pt_r, "gain_reg_ci_lo": lo_r, "gain_reg_ci_hi": hi_r, "gain_reg_p": p_r,
        })

    out = pd.DataFrame(rows)
    sig_cls = bh_adjust(pvals_cls) if pvals_cls else {}
    sig_reg = bh_adjust(pvals_reg) if pvals_reg else {}
    out["bh_sig_cls"] = out["group"].map(lambda c: sig_cls.get(c, None))
    out["bh_sig_reg"] = out["group"].map(lambda c: sig_reg.get(c, None))

    pd.set_option("display.width", 160)
    print("\n=== gate bootstrap significance (env-level, n_boot={}) ===".format(args.n_boot))
    print(out.round(4).to_string(index=False))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
