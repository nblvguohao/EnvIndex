"""envindex_v0_pilot.py — End-to-end pilot of the EnvIndex encoder on ESWYT.

Runs the protocol §4.2 EnvIndex module on a slice of the CIMMYT ESWYT data:
environment stage-feature matrices (3 stages x 8 weather vars from the IWIN
tabs) -> learned embedding z_e via MLP-Mixer, trained jointly on yield
(main), environment-mean (aux A) and same-year contrastive (aux B) losses.

This is the W5-6 "subset ~50 environments quick loop" validation.

Usage:
    python scripts/envindex_v0_pilot.py \
        --data data/cimmyt/ESWYT_Obs_Sim_Yld_Phe_Climate_All.tab \
        --n-envs 40 --epochs 60 --d-embed 16 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envindex.train import EnvIndexModule, EnvYieldDataset, collate, evaluate, train_epoch  # noqa: E402

STAGE_SUFFIXES = ["veg", "rep", "gfi"]
FEATURES = ["tavg", "tdr", "gdd30", "rs", "p", "rh", "vpd", "ws"]


def build_stage_matrix(row: pd.Series) -> np.ndarray:
    """3 (stage) x 8 (feature) matrix from a row of IWIN climate columns."""
    out = np.zeros((len(STAGE_SUFFIXES), len(FEATURES)), dtype=np.float32)
    for si, stage in enumerate(STAGE_SUFFIXES):
        for fi, feat in enumerate(FEATURES):
            col = f"{feat}_{stage}"
            out[si, fi] = row.get(col, np.nan) if col in row.index else np.nan
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/cimmyt/ESWYT_Obs_Sim_Yld_Phe_Climate_All.tab"))
    parser.add_argument("--n-envs", type=int, default=40, help="Environments to use (quick loop)")
    parser.add_argument("--n-genos", type=int, default=200, help="Genotypes to keep")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--d-embed", type=int, default=16)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--d-geno", type=int, default=32)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data, sep="\t")
    df["env_id"] = df["loc"].astype(str) + "_" + df["year"].astype(str)

    # Subset environments (keep a spread) and genotypes.
    envs = sorted(df["env_id"].unique())
    rng = np.random.default_rng(args.seed)
    keep_envs = sorted(rng.choice(envs, size=min(args.n_envs, len(envs)), replace=False))
    df = df[df["env_id"].isin(keep_envs)].copy()

    top_genos = df["gen"].value_counts().head(args.n_genos).index
    df = df[df["gen"].isin(top_genos)].reset_index(drop=True)
    geno2idx = {g: i for i, g in enumerate(sorted(df["gen"].unique()))}
    df["geno_idx"] = df["gen"].map(geno2idx)

    # Train/val split by environment (leakage-safe).
    envs = sorted(df["env_id"].unique())
    n_val = max(1, int(0.2 * len(envs)))
    val_envs = set(rng.choice(envs, size=n_val, replace=False))
    train_df, val_df = df[~df["env_id"].isin(val_envs)], df[df["env_id"].isin(val_envs)]
    print(f"[envindex_v0] {len(train_df)} train / {len(val_df)} val rows, "
          f"{len(envs)} envs ({len(val_envs)} val), {df['gen'].nunique()} genotypes")

    # Environment mean (train-only, leakage safety).  Validation environments
    # get the global train mean as a neutral aux-A target (no leakage).
    train_env_mean = train_df.groupby("env_id")["yld"].mean().to_dict()
    global_mean = float(train_df["yld"].mean())
    env_mean = {e: m for e, m in train_env_mean.items()}
    for e in envs:
        env_mean.setdefault(e, global_mean)

    # Feature normalization (train envs only).
    all_mats = np.stack([build_stage_matrix(row) for _, row in train_df.iterrows()])
    feat_mean = np.nanmean(all_mats, axis=(0, 1), keepdims=False)  # (F,)
    feat_std = np.nanstd(all_mats, axis=(0, 1), keepdims=False) + 1e-6  # (F,)

    def make_items(sub: pd.DataFrame) -> list[dict]:
        items = []
        for _, row in sub.iterrows():
            mat = (build_stage_matrix(row) - feat_mean) / feat_std
            items.append(
                {
                    "x": mat,
                    "static": None,
                    "g_emb": np.random.randn(args.d_geno).astype(np.float32),  # fixed init for v0
                    "geno_idx": int(row["geno_idx"]),
                    "y": float(row["yld"]),
                    "env_id": row["env_id"],
                    "env_label": int(row["year"]),  # same-year contrastive positives
                }
            )
        return items

    train_items = make_items(train_df)
    val_items = make_items(val_df)
    train_loader = DataLoader(EnvYieldDataset(train_items, env_mean), batch_size=256, collate_fn=collate, shuffle=True)
    val_loader = DataLoader(EnvYieldDataset(val_items, env_mean), batch_size=512, collate_fn=collate)

    module = EnvIndexModule(
        n_stages=len(STAGE_SUFFIXES),
        n_features=len(FEATURES),
        d_embed=args.d_embed,
        d_geno=args.d_geno,
        rank=args.rank,
        n_genotypes=len(geno2idx),
    ).to(args.device)
    print(f"[envindex_v0] params: {module.encoder.n_parameters():,}")

    opt = torch.optim.AdamW(module.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        tr = train_epoch(module, train_loader, opt, args.lambda_a, args.lambda_b, device=args.device)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            val = evaluate(module, val_loader, args.device)
            print(f"[envindex_v0] epoch {epoch+1:3d} | loss={tr['loss']:.4f} "
                  f"(main {tr['main']:.4f}, auxA {tr['aux_a']:.4f}, auxB {tr['aux_b']:.4f}) | "
                  f"val yld-PCC={val['pcc_yield']:.3f} envmean-PCC={val['pcc_envmean']:.3f}")

    val = evaluate(module, val_loader, args.device)
    print(f"[envindex_v0] FINAL | val yld-PCC={val['pcc_yield']:.3f} "
          f"envmean-PCC={val['pcc_envmean']:.3f} mse_main={val['mse_main']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
