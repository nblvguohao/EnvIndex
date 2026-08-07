"""pooled_selection_gain.py — pooled SelectionGain@10% (protocol §6) across crops.

Protocol §6 definition: SelectionGain@10% = within a crop, pool ALL test-fold
records, select the top-10% GENOTYPES (by per-genotype mean prediction over
their pooled records), then take the mean observed phenotype of the selected
genotypes minus the population mean; ties broken randomly, mean ± SD over 10
random tie-breaks.

This is the form in which the selection metric does NOT degenerate under
within-environment ranking (amendment M3: per-environment SelectionGain/NDCG
are invariant to the environment term, making G+E == geno-mean; pooling
restores discriminative power).

Arms compared (all predictions from the same LOEO folds, raw scale):
  pred_gz : EnvIndex interaction model (learned, rank=4)
  pred_fw : Finlay-Wilkinson reaction norm (M3 main baseline)
  pred_gm : genotype-mean (no-GxE null)

Inference: environment-level bootstrap (resample envs with replacement,
recompute pooled gains and diffs), percentile CI, two-sided p; BH across
crops per comparison family.

Step 1 (--run) executes the LOEO dump per crop (GPU); step 2 (--analyze-only)
recomputes statistics from saved preds parquets without GPU.

Usage:
    python scripts/pooled_selection_gain.py --run --crops barley,oat   # GPU
    python scripts/pooled_selection_gain.py --analyze-only             # CPU
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

ARMS = ["pred_gz", "pred_fw", "pred_gm"]


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true", help="Run LOEO with pred dump for --crops")
    ap.add_argument("--analyze-only", action="store_true", help="Skip LOEO; analyze saved preds")
    ap.add_argument("--crops", default="barley,oat,corn,wheat")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--fold-workers", type=int, default=12)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--tie-breaks", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0,
                    help="Cohort/loader seed (env sampling) -- ALWAYS keep 0 so the "
                         "wheat/corn env cohorts match the amendment §7.2 rerun")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="Training seed for run_loe (default: same as --seed).  "
                         "Seed-ensemble UQ runs set this to 1/2 while --seed stays 0.")
    ap.add_argument("--preds-suffix", default="",
                    help="Suffix for preds parquet names, e.g. '_s1' -> preds_sg_corn_s1.parquet")
    ap.add_argument("--dir", type=Path, default=ROOT / "data/t3")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "data/t3/pooled_selection_gain.csv")
    return ap.parse_args(argv)


# ------------------------------------------------------------------ LOEO dump

def run_crop(crop: str, args) -> Path:
    """LOEO learned arm with per-plot prediction dump; saves preds parquet."""
    from loe_pilot import run_loe

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
        items, _ = load_wheat(800, args.seed, plot_cap=100)
    elif crop == "corn":
        from loe_pilot import load_corn
        items, _ = load_corn(400, args.seed, plot_cap=100)
    else:
        raise ValueError(crop)

    # same plot cap as the amendment §7.2 rerun for the T3 loaders
    if crop in ("barley", "oat"):
        df = pd.DataFrame(items).groupby("env_id").head(100)
        items = df.to_dict("records")

    print(f"[sg] {crop}: {len(items)} items, {len(set(i['env_id'] for i in items))} envs", flush=True)
    train_seed = args.seed if args.train_seed is None else args.train_seed
    res, preds = run_loe(items, d_embed=32, d_geno=32, rank=4, epochs=args.epochs,
                         device=args.device, seed=train_seed, batch_size=512,
                         fold_workers=args.fold_workers, n_gpus=args.n_gpus,
                         embed_mode="learned", dump_preds=True)
    out = args.dir / f"preds_sg_{crop}{args.preds_suffix}.parquet"
    pd.DataFrame(preds).to_parquet(out, index=False)
    print(f"[sg] {crop}: {len(res)} folds, {len(preds)} preds -> {out}", flush=True)
    return out


# ------------------------------------------------------------------ metric

def _sel_gain_once(score: np.ndarray, y: np.ndarray, top_frac: float, rng) -> float:
    """One tie-broken SelectionGain: top-frac of genotypes by score."""
    n_sel = max(1, int(round(len(score) * top_frac)))
    order = np.argsort(score + rng.normal(scale=1e-9, size=len(score)), kind="stable")
    sel = order[-n_sel:]
    return float(y[sel].mean() - y.mean())


def selection_gain(df: pd.DataFrame, arm: str, top_frac: float, tie_breaks: int, seed: int) -> tuple[float, float]:
    """Pooled SelectionGain for one arm: per-genotype mean prediction selects
    top genotypes; gain = mean y of selected records - overall mean y.
    Returns (mean over tie-breaks, sd over tie-breaks)."""
    g = df.groupby("geno").agg(score=(arm, "mean"), y=("y", "mean"), n=("y", "size"))
    score, y = g["score"].to_numpy(), g["y"].to_numpy()
    gains = []
    for b in range(tie_breaks):
        rng = np.random.default_rng(seed + b)
        gains.append(_sel_gain_once(score, y, top_frac, rng))
    return float(np.mean(gains)), float(np.std(gains))


def analyze(args) -> pd.DataFrame:
    rows = []
    p_fam: dict[str, dict[str, float]] = {"gz_fw": {}, "gz_gm": {}}
    for crop in args.crops.split(","):
        path = args.dir / f"preds_sg_{crop}.parquet"
        if not path.exists():
            print(f"[skip] {crop}: missing {path.name}")
            continue
        df = pd.read_parquet(path)
        row: dict = {"crop": crop, "n_env": df["env_id"].nunique(),
                     "n_geno": df["geno"].nunique(), "n_rec": len(df)}
        for arm in ARMS:
            m, s = selection_gain(df, arm, args.top_frac, args.tie_breaks, args.seed)
            row[f"sg_{arm[5:]}"] = m
            row[f"sg_{arm[5:]}_sd"] = s

        # env-level bootstrap of the pairwise diffs
        rng = np.random.default_rng(args.seed)
        envs = df["env_id"].unique()
        by_env = {e: g for e, g in df.groupby("env_id")}
        diffs = {"gz_fw": [], "gz_gm": []}
        for _ in range(args.n_boot):
            samp = rng.choice(envs, size=len(envs), replace=True)
            boot = pd.concat([by_env[e] for e in samp], ignore_index=True)
            sg = {a: selection_gain(boot, a, args.top_frac, 1, args.seed)[0] for a in ARMS}
            diffs["gz_fw"].append(sg["pred_gz"] - sg["pred_fw"])
            diffs["gz_gm"].append(sg["pred_gz"] - sg["pred_gm"])
        for fam, arr in diffs.items():
            arr = np.array(arr)
            row[f"d_{fam}"] = float(arr.mean())
            row[f"d_{fam}_lo"], row[f"d_{fam}_hi"] = np.percentile(arr, [2.5, 97.5])
            p = 2.0 * min((arr <= 0).mean(), (arr >= 0).mean())
            row[f"d_{fam}_p"] = float(min(p, 1.0))
            p_fam[fam][crop] = row[f"d_{fam}_p"]
        rows.append(row)

    out = pd.DataFrame(rows)
    for fam in p_fam:
        pvals = p_fam[fam]
        order = sorted(pvals, key=pvals.get)
        m = len(order)
        passing = [c for r, c in enumerate(order) if pvals[c] <= 0.05 * (r + 1) / m]
        cutoff = max((0.05 * (order.index(c) + 1) / m for c in passing), default=0.0)
        out[f"d_{fam}_bh"] = out["crop"].map({c: pvals[c] <= cutoff for c in pvals})
    return out


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.run:
        for crop in args.crops.split(","):
            run_crop(crop, args)
    if args.run or args.analyze_only:
        df = analyze(args)
        if len(df):
            pd.set_option("display.width", 240)
            print("\n=== pooled SelectionGain@10% (protocol §6) ===")
            print(df.round(4).to_string(index=False))
            df.to_csv(args.out_csv, index=False)
            print(f"[saved] {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
