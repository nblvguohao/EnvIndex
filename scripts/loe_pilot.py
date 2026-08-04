"""loe_pilot.py — cross-crop (corn + wheat) pilot leave-one-environment-out.

Runs the protocol's LOEO machinery on a small subset of environments from two
crops:
  wheat  : CIMMYT ESWYT (stage features 3x8 built-in)
  corn   : G2F (stage features 5x24 built from daily weather via
           envindex.corn_features)

For each held-out environment e the EnvIndex model (G+E + low-rank G∘z) is
retrained on the remaining environments and its per-environment yield PCC is
compared against a G+E-only baseline to give Delta(e) = PCC_Gz - PCC_GE
(protocol §4.4).  This is the "~50 env quick loop" pilot (W5-6).

Usage:
    python scripts/loe_pilot.py --n-envs-wheat 15 --n-envs-corn 12 --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "nc" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envindex.corn_features import load_corn_envs  # noqa: E402
from envindex.train import EnvIndexModule, EnvYieldDataset, collate  # noqa: E402

ESWYT = ROOT / "data/cimmyt/ESWYT_Obs_Sim_Yld_Phe_Climate_All.tab"
G2F_PHENO = ROOT.parent / "nc/data/processed/g2f/phenotype.parquet"
G2F_WEATHER = ROOT.parent / "nc/data/processed/g2f/weather_daily.parquet"
G2F_ENV = ROOT.parent / "nc/data/processed/g2f/environment.parquet"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-envs-wheat", type=int, default=15)
    parser.add_argument("--n-envs-corn", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-embed", type=int, default=16)
    parser.add_argument("--d-geno", type=int, default=16)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


# ---------------------------------------------------------------- loaders

def load_wheat(envs_to_use: int, seed: int) -> tuple[list[dict], list[str]]:
    df = pd.read_csv(ESWYT, sep="\t")
    df["env_id"] = df["loc"].astype(str) + "_" + df["year"].astype(str)
    # keep envs with enough plots for meaningful PCC
    counts = df.groupby("env_id")["gen"].count()
    usable = counts[counts >= 12].index
    df = df[df["env_id"].isin(usable)]
    rng = np.random.default_rng(seed)
    envs = sorted(rng.choice(sorted(df["env_id"].unique()), size=min(envs_to_use, len(usable)), replace=False))
    df = df[df["env_id"].isin(envs)].copy()
    stage_suffix = ["veg", "rep", "gfi"]
    feats = ["tavg", "tdr", "gdd30", "rs", "p", "rh", "vpd", "ws"]
    items = []
    for _, row in df.iterrows():
        mat = np.array([[row.get(f"{f}_{s}", np.nan) for f in feats] for s in stage_suffix], dtype=np.float32)
        items.append(
            {
                "crop": "wheat",
                "env_id": row["env_id"],
                "geno": str(row["gen"]),
                "y": float(row["yld"]),
                "x": mat,
                "env_label": int(row["year"]),
            }
        )
    return items, sorted(set(i["env_id"] for i in items))


def load_corn(envs_to_use: int, seed: int) -> tuple[list[dict], list[str]]:
    pheno = pd.read_parquet(G2F_PHENO)
    pheno = pheno.dropna(subset=["phenotype_value", "genotype_id", "environment_id"])
    envs = load_corn_envs(str(G2F_WEATHER), str(G2F_ENV), n_envs=envs_to_use, seed=seed)
    items = []
    for env_id, info in envs.items():
        sub = pheno[pheno["environment_id"] == env_id]
        for _, row in sub.head(300).iterrows():  # cap plots per env for pilot speed
            items.append(
                {
                    "crop": "corn",
                    "env_id": env_id,
                    "geno": str(row["genotype_id"]),
                    "y": float(row["phenotype_value"]),
                    "x": info["x"],
                    "env_label": int(row["year"]) if "year" in row.index and pd.notna(row.get("year")) else 0,
                }
            )
    return items, sorted(set(i["env_id"] for i in items))


# ---------------------------------------------------------------- LOEO

def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_loe(
    items: list[dict],
    n_genotypes: int,
    d_embed: int,
    d_geno: int,
    rank: int,
    epochs: int,
    device: str,
    seed: int,
) -> dict:
    """Run leave-one-environment-out; returns per-env {pcc_gz, pcc_ge, delta}."""
    envs = sorted(set(i["env_id"] for i in items))
    geno2idx = {g: i for i, g in enumerate(sorted(set(i["geno"] for i in items)))}
    # fallback index for genotypes unseen in a given train split
    UNK = len(geno2idx)
    rng = np.random.default_rng(seed)

    geno_emb = nn.Embedding(len(geno2idx) + 1, d_geno).to(device)

    results = {}
    for held in envs:
        train_items = [i for i in items if i["env_id"] != held]
        held_items = [i for i in items if i["env_id"] == held]
        if len(train_items) == 0 or len(held_items) < 3:
            continue

        # feature normalization (train only)
        xs = np.stack([i["x"] for i in train_items])
        x_mean = np.nanmean(xs, axis=(0, 1), keepdims=False)
        x_std = np.nanstd(xs, axis=(0, 1), keepdims=False) + 1e-6

        train_env_mean = {}
        for i in train_items:
            train_env_mean.setdefault(i["env_id"], []).append(i["y"])
        train_env_mean = {e: float(np.mean(v)) for e, v in train_env_mean.items()}
        global_mean = float(np.mean([i["y"] for i in train_items]))
        geno_train_mean = {}
        for i in train_items:
            geno_train_mean.setdefault(i["geno"], []).append(i["y"])
        geno_train_mean = {g: float(np.mean(v)) for g, v in geno_train_mean.items()}

        def build_dataset(source):
            ds_items = []
            for i in source:
                x = (np.nan_to_num(i["x"]) - x_mean) / x_std
                gidx = geno2idx.get(i["geno"], UNK)
                gemb = geno_emb(torch.tensor([gidx], device=device)).detach().cpu().squeeze(0).numpy()
                ds_items.append(
                    {
                        "x": x,
                        "static": None,
                        "g_emb": gemb,
                        "geno_idx": gidx,
                        "geno": i["geno"],
                        "y": i["y"],
                        "env_id": i["env_id"],
                        "env_label": i["env_label"],
                        "env_mean": train_env_mean.get(i["env_id"], global_mean),
                    }
                )
            return ds_items

        train_ds = EnvYieldDataset(build_dataset(train_items), {i["env_id"]: train_env_mean.get(i["env_id"], global_mean) for i in train_items})
        train_loader = DataLoader(train_ds, batch_size=128, collate_fn=collate, shuffle=True)

        module = EnvIndexModule(
            n_stages=items[0]["x"].shape[0],
            n_features=items[0]["x"].shape[1],
            d_embed=d_embed,
            d_geno=d_geno,
            rank=rank,
            n_genotypes=len(geno2idx) + 1,
        ).to(device)
        opt = torch.optim.AdamW(module.parameters(), lr=3e-3, weight_decay=1e-4)

        for _ in range(epochs):
            module.train()
            for batch in train_loader:
                x = batch["x"].to(device)
                static = batch["static"].to(device)
                g = batch["g_emb"].to(device)
                idx = batch["geno_idx"].to(device)
                y = batch["y"].to(device)
                y_hat, env_hat, z = module(x, g, idx, static)
                loss = nn.functional.mse_loss(y_hat, y) + 0.5 * nn.functional.mse_loss(env_hat, batch["env_mean"].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()

        # predict held-out
        module.eval()
        hd = build_dataset(held_items)
        ys, gz_pred, ge_pred = [], [], []
        with torch.no_grad():
            for it in hd:
                x = torch.as_tensor(it["x"], dtype=torch.float32).unsqueeze(0).to(device)
                st_raw = it["static"] if it["static"] is not None else []
                st = torch.as_tensor(st_raw, dtype=torch.float32).unsqueeze(0).to(device)
                g = torch.as_tensor(it["g_emb"], dtype=torch.float32).unsqueeze(0).to(device)
                idx = torch.as_tensor([it["geno_idx"]], dtype=torch.long).to(device)
                y_hat, _, _ = module(x, g, idx, st)
                gz_pred.append(float(y_hat.squeeze().cpu()))
                ys.append(it["y"])
                # G+E additive baseline: env mean + genotype deviation
                gdev = geno_train_mean.get(it["geno"], global_mean) - global_mean
                ge_pred.append(float(it["env_mean"] + gdev))
        ys = np.array(ys)
        pcc_gz = _pcc(ys, np.array(gz_pred))
        pcc_ge = _pcc(ys, np.array(ge_pred))
        results[held] = {"pcc_gz": pcc_gz, "pcc_ge": pcc_ge, "delta": pcc_gz - pcc_ge}

    return results


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("[loe_pilot] loading wheat ESWYT ...")
    w_items, w_envs = load_wheat(args.n_envs_wheat, args.seed)
    print(f"[loe_pilot] wheat: {len(w_items)} rows, {len(w_envs)} envs")
    w_res = run_loe(w_items, len(set(i["geno"] for i in w_items)), args.d_embed, args.d_geno, args.rank, args.epochs, args.device, args.seed)

    print("[loe_pilot] loading corn G2F ...")
    c_items, c_envs = load_corn(args.n_envs_corn, args.seed)
    print(f"[loe_pilot] corn: {len(c_items)} rows, {len(c_envs)} envs")
    c_res = run_loe(c_items, len(set(i["geno"] for i in c_items)), args.d_embed, args.d_geno, args.rank, args.epochs, args.device, args.seed)

    def summarize(name, res):
        rows = list(res.values())
        gz = [r["pcc_gz"] for r in rows if not np.isnan(r["pcc_gz"])]
        ge = [r["pcc_ge"] for r in rows if not np.isnan(r["pcc_ge"])]
        dl = [r["delta"] for r in rows if not np.isnan(r["delta"])]
        print(f"\n=== {name} LOEO ({len(rows)} held-out envs):")
        print(f"  PCC(Gz): {np.mean(gz):+.3f} +- {np.std(gz):.3f}  (n={len(gz)})")
        print(f"  PCC(G+E): {np.mean(ge):+.3f} +- {np.std(ge):.3f}  (n={len(ge)})")
        print(f"  Delta = PCC(Gz)-PCC(G+E): {np.mean(dl):+.3f} +- {np.std(dl):.3f}  (n={len(dl)})")
        return dl

    summarize("wheat ESWYT", w_res)
    summarize("corn G2F", c_res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
