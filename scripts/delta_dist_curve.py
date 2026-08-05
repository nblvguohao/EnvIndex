"""delta_dist_curve.py — Δ-dist dose-response curve (protocol §4.4, H2 main figure).

For each held-out environment e:
    Δ(e) = PCC(G∘z, e) − PCC(G+E, e)
plotted against
    dist(e) = mean k-NN embedding distance from z_e to the other environments.

Produces per-crop and combined scatter + binned means with environment-level
bootstrap CI, and LOESS fit (protocol §4.4.1).

Usage:
    # after loe_pilot.py --out-results data/t3/loe_results.parquet:
    python scripts/delta_dist_curve.py \
        --results data/t3/loe_results.parquet \
        --n-envs-wheat 100 --n-envs-corn 100 \
        --out-fig figures/delta_dist_curve.png --out-csv data/t3/delta_dist.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "nc" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from loe_pilot import load_corn, load_wheat  # noqa: E402
from envindex.sampling import encode_all, environment_distance  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="Per-env results parquet from loe_pilot --out-results")
    parser.add_argument("--n-envs-wheat", type=int, default=100, help="Wheat envs to load for dist computation")
    parser.add_argument("--n-envs-corn", type=int, default=100, help="Corn envs to load for dist computation")
    parser.add_argument("--out-fig", type=Path, default=Path("figures/delta_dist_curve.png"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/t3/delta_dist.csv"))
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _train_quick_encoder(items, device, epochs=80):
    import torch
    import torch.nn as nn
    from envindex.train import EnvIndexModule

    rng = np.random.default_rng(0)
    geno2idx = {g: i for i, g in enumerate(sorted(set(i["geno"] for i in items)))}
    module = EnvIndexModule(
        n_stages=items[0]["x"].shape[0],
        n_features=items[0]["x"].shape[1],
        d_embed=16, d_geno=16, rank=2,
        n_genotypes=len(geno2idx) + 1,
    ).to(device)
    xs = np.stack([np.nan_to_num(i["x"]) for i in items])
    xm = np.nanmean(xs, axis=(0, 1), keepdims=False)
    xsd = np.nanstd(xs, axis=(0, 1), keepdims=False) + 1e-6
    opt = torch.optim.AdamW(module.parameters(), lr=3e-3)
    for _ in range(epochs):
        batch = rng.choice(items, size=min(256, len(items)), replace=False)
        x = torch.as_tensor(np.stack([(np.nan_to_num(b["x"]) - xm) / xsd for b in batch]), dtype=torch.float32).to(device)
        st = torch.zeros(len(batch), 0, dtype=torch.float32, device=device)
        idx = torch.as_tensor([geno2idx.get(b["geno"], len(geno2idx)) for b in batch], dtype=torch.long).to(device)
        y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32).to(device)
        y_hat, env_hat, _ = module(x, None, idx, st)
        loss = nn.functional.mse_loss(y_hat, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return module


def bootstrap_ci(x, y, n_bins, n_iter, seed):
    """Quantile-binned means with bootstrap CI over environments."""
    rng = np.random.default_rng(seed)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 2:
            continue
        yb = y[mask]
        mean = float(yb.mean())
        boot = np.array([np.mean(rng.choice(yb, size=len(yb), replace=True)) for _ in range(n_iter)])
        out.append({"dist_lo": float(lo), "dist_hi": float(hi), "n": int(mask.sum()),
                    "mean": mean, "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    args = _parse_args(argv)
    res = pd.read_parquet(args.results)
    print(f"[delta_dist] loaded {len(res)} per-env results")

    # Environment-level distance in the R1 feature space (standardized
    # Euclidean k-NN).  This spreads environments far better than the weak
    # pilot encoder's cosine embedding distance (which collapses to ~0), and is
    # the interpretable "environmental dissimilarity" behind the protocol's
    # dist(e).  Computed per crop with its own feature scale.
    dist: dict[str, float] = {}
    for crop, items in _items_for_result_envs(res, args).items():
        if not items:
            continue
        # one feature matrix per environment (identical within an env)
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
        for e, v in zip(env_ids, knn):
            dist[e] = float(v)
        print(f"[delta_dist] {crop}: env-feature dist on {len(env_ids)} envs, "
              f"range ({min(dist[e] for e in env_ids):.4f}, {max(dist[e] for e in env_ids):.4f})")

    dist_df = pd.DataFrame([{"env_id": e, "dist": d} for e, d in dist.items()])
    df = res.merge(dist_df, on="env_id", how="inner")
    print(f"[delta_dist] merged {len(df)} envs with dist (dist range {df['dist'].min():.4f}-{df['dist'].max():.4f})")

    # per-crop binned means + bootstrap CI
    frames = []
    for crop, g in df.groupby("crop"):
        bc = bootstrap_ci(g["dist"].to_numpy(), g["delta"].to_numpy(), args.n_bins, args.bootstrap, args.seed)
        bc["crop"] = crop
        frames.append(bc)
    bins_df = pd.concat(frames, ignore_index=True)

    df.to_csv(args.out_csv, index=False)
    print(f"[delta_dist] csv -> {args.out_csv}")

    # figure: scatter + binned mean ± CI + LOESS, per crop
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (crop, g) in zip(axes, df.groupby("crop")):
        ax.scatter(g["dist"], g["delta"], s=8, alpha=0.4, label=f"{crop} envs")
        bc = bins_df[bins_df["crop"] == crop]
        ax.errorbar((bc["dist_lo"] + bc["dist_hi"]) / 2, bc["mean"],
                    yerr=[bc["mean"] - bc["ci_lo"], bc["ci_hi"] - bc["mean"]],
                    fmt="o-", color="red", capsize=3, label="binned mean ± 95% CI")
        sm = lowess(g["delta"].to_numpy(), g["dist"].to_numpy(), frac=0.5)
        ax.plot(sm[:, 0], sm[:, 1], "--", color="blue", label="LOESS", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xlabel("dist(e)  (env-feature Euclidean k-NN)")
        ax.set_ylabel("Δ(e) = PCC(G∘z) − PCC(G+E)")
        ax.set_title(f"{crop}  (n={len(g)})")
        ax.legend(fontsize=8)
    fig.suptitle("G×E predictability boundary: Δ vs environmental distance")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"[delta_dist] figure -> {args.out_fig}")

    print("[delta_dist] binned summary:")
    print(bins_df.round(4).to_string(index=False))
    return 0


def _items_for_result_envs(res: pd.DataFrame, args) -> dict[str, list[dict]]:
    """Rebuild environment items restricted to the result env_ids, per crop."""
    import numpy as np
    import pandas as pd

    result_envs = set(res["env_id"])
    out: dict[str, list[dict]] = {}

    # wheat: ESWYT rows filtered to result env_ids (env_id = loc_year)
    wheat = res[res["crop"] == "wheat"]["env_id"]
    if len(wheat):
        df = pd.read_csv(ROOT / "data/cimmyt/ESWYT_Obs_Sim_Yld_Phe_Climate_All.tab", sep="\t")
        df["env_id"] = df["loc"].astype(str) + "_" + df["year"].astype(str)
        df = df[df["env_id"].isin(wheat)].dropna(subset=["yld"])
        stage = ["veg", "rep", "gfi"]
        feats = ["tavg", "tdr", "gdd30", "rs", "p", "rh", "vpd", "ws"]
        items = []
        for _, row in df.iterrows():
            mat = np.array([[row.get(f"{f}_{s}", np.nan) for f in feats] for s in stage], dtype=np.float32)
            items.append({"env_id": row["env_id"], "geno": str(row["gen"]), "y": float(row["yld"]),
                          "x": mat, "env_label": int(row["year"])})
        out["wheat"] = items

    # corn: G2F phenotype + corn_features filtered to result env_ids
    corn = res[res["crop"] == "corn"]["env_id"]
    if len(corn):
        from envindex.corn_features import load_corn_envs
        pheno = pd.read_parquet(ROOT.parent / "nc/data/processed/g2f/phenotype.parquet")
        pheno = pheno.dropna(subset=["phenotype_value", "genotype_id", "environment_id"])
        envs = load_corn_envs(str(ROOT.parent / "nc/data/processed/g2f/weather_daily.parquet"),
                              str(ROOT.parent / "nc/data/processed/g2f/environment.parquet"),
                              n_envs=500, seed=0)  # load all, filter below
        items = []
        for env_id, info in envs.items():
            if env_id not in set(corn):
                continue
            sub = pheno[pheno["environment_id"] == env_id]
            for _, row in sub.head(300).iterrows():
                items.append({"env_id": env_id, "geno": str(row["genotype_id"]),
                              "y": float(row["phenotype_value"]), "x": info["x"], "env_label": 0})
        out["corn"] = items

    return out

    # per-crop binned means + bootstrap CI
    frames = []
    for crop, g in df.groupby("crop"):
        bc = bootstrap_ci(g["dist"].to_numpy(), g["delta"].to_numpy(), args.n_bins, args.bootstrap, args.seed)
        bc["crop"] = crop
        frames.append(bc)
    bins_df = pd.concat(frames, ignore_index=True)

    df.to_csv(args.out_csv, index=False)
    print(f"[delta_dist] csv -> {args.out_csv}")

    # figure: scatter + binned mean ± CI + LOESS, per crop
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (crop, g) in zip(axes, df.groupby("crop")):
        ax.scatter(g["dist"], g["delta"], s=8, alpha=0.4, label=f"{crop} envs")
        bc = bins_df[bins_df["crop"] == crop]
        ax.errorbar((bc["dist_lo"] + bc["dist_hi"]) / 2, bc["mean"], yerr=[bc["mean"] - bc["ci_lo"], bc["ci_hi"] - bc["mean"]],
                    fmt="o-", color="red", capsize=3, label="binned mean ± 95% CI")
        sm = lowess(g["delta"].to_numpy(), g["dist"].to_numpy(), frac=0.5)
        ax.plot(sm[:, 0], sm[:, 1], "--", color="blue", label="LOESS", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xlabel("dist(e)  (k-NN embedding distance)")
        ax.set_ylabel("Δ(e) = PCC(G∘z) − PCC(G+E)")
        ax.set_title(f"{crop}  (n={len(g)})")
        ax.legend(fontsize=8)
    fig.suptitle("G×E predictability boundary: Δ vs environmental distance")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"[delta_dist] figure -> {args.out_fig}")

    print("[delta_dist] binned summary:")
    print(bins_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
