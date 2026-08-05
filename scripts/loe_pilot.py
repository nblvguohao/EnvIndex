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
    parser.add_argument("--strata", type=int, default=0,
                        help="Distance-stratified env sampling (improvement #4): number of bins (>0 enables)")
    parser.add_argument("--target", type=int, default=30, help="Target envs after stratified sampling")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Training batch size (1024 on A100 for GPU efficiency)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process; >0 avoids data loading stalls)")
    parser.add_argument("--fold-workers", type=int, default=1,
                        help="Parallel LOEO fold workers (>1 = ProcessPool, each pinned to a GPU)")
    parser.add_argument("--n-gpus", type=int, default=1, help="GPUs to round-robin fold workers across")
    parser.add_argument("--out-results", type=str, default="",
                        help="Save per-environment results (env_id, crop, pcc_*, delta) to this parquet")
    parser.add_argument("--embed-mode", choices=["learned", "pca"], default="learned",
                        help="learned=EnvIndex encoder; pca=PCA projection control (protocol §5-13)")
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
    df = df.dropna(subset=["yld"])  # ESWYT has ~129 NaN yields; drop to keep MSE finite
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


_SHARED: dict = {}


def _run_one_fold(
    items: list[dict],
    held: str,
    d_embed: int,
    d_geno: int,
    rank: int,
    epochs: int,
    device: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    embed_mode: str = "learned",
) -> tuple[str, dict]:
    """Train on all-but-held environment, predict held-out, return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_items = [i for i in items if i["env_id"] != held]
    held_items = [i for i in items if i["env_id"] == held]
    if len(train_items) == 0 or len(held_items) < 3:
        return held, {}

    geno2idx = {g: i for i, g in enumerate(sorted(set(i["geno"] for i in items)))}
    UNK = len(geno2idx)

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
            ds_items.append(
                {
                    "x": x,
                    "static": None,
                    "g_emb": None,  # learnable via geno_idx (improvement #1)
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
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    module = EnvIndexModule(
        n_stages=items[0]["x"].shape[0],
        n_features=items[0]["x"].shape[1],
        d_embed=d_embed,
        d_geno=d_geno,
        rank=rank,
        n_genotypes=len(geno2idx) + 1,
        embed_mode=embed_mode,
    ).to(device)
    if embed_mode == "pca":
        # PCA control: fit projection on the TRAINING fold's stage features only
        # (leakage-safe), set it as the fixed encoder.
        flat = np.stack([np.nan_to_num(i["x"]).reshape(-1) for i in train_items])
        center = flat.mean(0)
        Xc = flat - center
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:d_embed].T.astype(np.float32)  # (in_features, d_embed)
        module.encoder.set_projection(torch.from_numpy(W).to(device))
    opt = torch.optim.AdamW(module.parameters(), lr=3e-3, weight_decay=1e-4)

    for _ in range(epochs):
        module.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            static = batch["static"].to(device)
            g = batch["g_emb"].to(device) if batch["g_emb"] is not None else None
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
    ys, gz_pred, ge_pred, gm_pred, rz_pred = [], [], [], [], []
    with torch.no_grad():
        for it in hd:
            x = torch.as_tensor(it["x"], dtype=torch.float32).unsqueeze(0).to(device)
            st_raw = it["static"] if it["static"] is not None else []
            st = torch.as_tensor(st_raw, dtype=torch.float32).unsqueeze(0).to(device)
            idx = torch.as_tensor([it["geno_idx"]], dtype=torch.long).to(device)
            y_hat, _, _ = module(x, None, idx, st)  # learnable geno embedding
            gz_pred.append(float(y_hat.squeeze().cpu()))
            ys.append(it["y"])
            # G+E additive baseline: env mean + genotype deviation
            gdev = geno_train_mean.get(it["geno"], global_mean) - global_mean
            ge_pred.append(float(it["env_mean"] + gdev))
            # genotype-mean baseline (GBLUP-G-lite)
            gm_pred.append(geno_train_mean.get(it["geno"], global_mean))
            # random-embedding negative control (protocol §5-12): G+E with random z
            rz = torch.randn(d_embed, device=device)
            rz_y = module.main_head.bias + module.main_head.env_main(rz.unsqueeze(0)).squeeze(0) \
                + (module.geno_embedding(idx) @ module.main_head.U * (rz.unsqueeze(0) @ module.main_head.V)).sum(-1).squeeze(0)
            rz_pred.append(float(rz_y.squeeze().cpu()))
    ys = np.array(ys)
    pcc_gz = _pcc(ys, np.array(gz_pred))
    pcc_ge = _pcc(ys, np.array(ge_pred))
    pcc_gm = _pcc(ys, np.array(gm_pred))
    pcc_rz = _pcc(ys, np.array(rz_pred))
    return held, {
        "pcc_gz": pcc_gz,
        "pcc_ge": pcc_ge,
        "pcc_gm": pcc_gm,
        "pcc_rz": pcc_rz,
        "delta": pcc_gz - pcc_ge,
    }


def _init_worker(items, params, counter, lock, n_gpus):
    """ProcessPool worker init: pin a GPU (round-robin) and share data."""
    import os

    global _SHARED
    with lock:
        gpu = counter.value
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu % n_gpus)
    _SHARED = {"items": items, "params": params}


def _run_fold_worker(held):
    global _SHARED
    items = _SHARED["items"]
    params = _SHARED["params"]
    n_held = len([i for i in items if i["env_id"] == held])
    if n_held == 0:
        print(f"[worker] WARN no items for held={held}", flush=True)
        return held, {}
    d_embed, d_geno, rank, epochs, batch_size, num_workers, seed, embed_mode = params
    return _run_one_fold(items, held, d_embed, d_geno, rank, epochs, "cuda", seed, batch_size, num_workers, embed_mode)


def run_loe(
    items: list[dict],
    d_embed: int,
    d_geno: int,
    rank: int,
    epochs: int,
    device: str,
    seed: int,
    batch_size: int = 1024,
    num_workers: int = 0,
    fold_workers: int = 1,
    n_gpus: int = 1,
    embed_mode: str = "learned",
) -> dict:
    """Leave-one-environment-out.  `fold_workers > 1` runs folds in parallel
    processes, each pinned to a GPU round-robin (improvement #4)."""
    envs = sorted(set(i["env_id"] for i in items))
    if fold_workers <= 1:
        results = {}
        for held in envs:
            h, r = _run_one_fold(items, held, d_embed, d_geno, rank, epochs, device, seed, batch_size, num_workers, embed_mode)
            if r:
                results[h] = r
        return results

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    # spawn (not fork): the parent has an initialized CUDA context; forking a
    # subprocess that uses CUDA raises "Cannot re-initialize CUDA".
    ctx = mp.get_context("spawn")
    params = (d_embed, d_geno, rank, epochs, batch_size, num_workers, seed, embed_mode)
    counter = ctx.Manager().Value("i", 0)
    lock = ctx.Manager().Lock()
    results = {}
    with ProcessPoolExecutor(
        max_workers=fold_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(items, params, counter, lock, n_gpus),
    ) as pool:
        for held, r in pool.map(_run_fold_worker, envs):
            if r:
                results[held] = r
    return results


def _save_crop_results(results: dict, crop: str, out_results: str) -> None:
    """Save one crop's per-environment results immediately (never trapped by a
    slow/failing later crop)."""
    import pandas as pd

    rows = [{"crop": crop, "env_id": env, **r} for env, r in results.items()]
    out = pd.DataFrame(rows)
    crop_path = str(out_results).replace("\\", "/").replace(".parquet", f"_{crop}.parquet")
    Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(crop_path, index=False)
    print(f"[loe_pilot] saved {len(out)} {crop} per-env results -> {crop_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("[loe_pilot] loading wheat ESWYT ...")
    w_items, w_envs = load_wheat(args.n_envs_wheat, args.seed)
    print(f"[loe_pilot] wheat: {len(w_items)} rows, {len(w_envs)} envs")
    if args.strata > 0:
        w_items = _apply_strata(w_items, args.strata, args.target, args.device)
        print(f"[loe_pilot] wheat stratified -> {len(set(i['env_id'] for i in w_items))} envs")
    w_res = run_loe(w_items, args.d_embed, args.d_geno, args.rank, args.epochs, args.device, args.seed,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    fold_workers=args.fold_workers, n_gpus=args.n_gpus,
                    embed_mode=args.embed_mode)
    if args.out_results:
        _save_crop_results(w_res, "wheat", args.out_results)

    c_res: dict = {}
    print("[loe_pilot] loading corn G2F ...")
    c_items, c_envs = load_corn(args.n_envs_corn, args.seed)
    print(f"[loe_pilot] corn: {len(c_items)} rows, {len(c_envs)} envs")
    if c_items:
        if args.strata > 0:
            c_items = _apply_strata(c_items, args.strata, args.target, args.device)
            print(f"[loe_pilot] corn stratified -> {len(set(i['env_id'] for i in c_items))} envs")
        c_res = run_loe(c_items, args.d_embed, args.d_geno, args.rank, args.epochs, args.device, args.seed,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        fold_workers=args.fold_workers, n_gpus=args.n_gpus,
                        embed_mode=args.embed_mode)
        if args.out_results:
            _save_crop_results(c_res, "corn", args.out_results)
    else:
        print("[loe_pilot] corn skipped (0 envs)")

    def summarize(name, res):
        rows = list(res.values())
        gz = [r["pcc_gz"] for r in rows if not np.isnan(r["pcc_gz"])]
        ge = [r["pcc_ge"] for r in rows if not np.isnan(r["pcc_ge"])]
        gm = [r["pcc_gm"] for r in rows if not np.isnan(r["pcc_gm"])]
        rz = [r["pcc_rz"] for r in rows if not np.isnan(r["pcc_rz"])]
        dl = [r["delta"] for r in rows if not np.isnan(r["delta"])]
        print(f"\n=== {name} LOEO ({len(rows)} held-out envs):")
        print(f"  PCC(Gz): {np.mean(gz):+.3f} +- {np.std(gz):.3f}  (n={len(gz)})")
        print(f"  PCC(G+E): {np.mean(ge):+.3f} +- {np.std(ge):.3f}  (n={len(ge)})")
        print(f"  PCC(geno-mean GBLUP-lite): {np.mean(gm):+.3f} +- {np.std(gm):.3f}  (n={len(gm)})")
        print(f"  PCC(random-z control): {np.mean(rz):+.3f} +- {np.std(rz):.3f}  (n={len(rz)})")
        print(f"  Delta = PCC(Gz)-PCC(G+E): {np.mean(dl):+.3f} +- {np.std(dl):.3f}  (n={len(dl)})")
        return dl

    summarize("wheat ESWYT", w_res)
    summarize("corn G2F", c_res)
    return 0


def _apply_strata(items: list[dict], n_bins: int, target: int, device: str) -> list[dict]:
    """Improvement #4: distance-stratified environment sampling.

    Trains a quick EnvIndex encoder on all sampled environments, encodes each
    env, computes dist(e) to k-NN, then samples `target` envs uniformly across
    dist-quantile bins.
    """
    from envindex.sampling import encode_all, environment_distance, stratify_sample

    env_ids = sorted(set(i["env_id"] for i in items))
    # quick encoder for distance computation
    from envindex.train import EnvIndexModule
    geno2idx = {g: i for i, g in enumerate(sorted(set(i["geno"] for i in items)))}
    module = EnvIndexModule(
        n_stages=items[0]["x"].shape[0],
        n_features=items[0]["x"].shape[1],
        d_embed=16,
        d_geno=16,
        rank=2,
        n_genotypes=len(geno2idx) + 1,
    ).to(device)
    # Normalize features (z-score) before the quick training — unnormalized
    # IWIN climate features (gdd30 up to ~344) make the encoder loss NaN,
    # which collapses embeddings and breaks stratification.
    xs = np.stack([np.nan_to_num(i["x"]) for i in items])
    x_mean = np.nanmean(xs, axis=(0, 1), keepdims=False)
    x_std = np.nanstd(xs, axis=(0, 1), keepdims=False) + 1e-6

    def norm(it):
        return (np.nan_to_num(it["x"]) - x_mean) / x_std

    # train briefly
    from torch.utils.data import DataLoader
    opt = torch.optim.AdamW(module.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        batch = rng.choice(items, size=min(256, len(items)), replace=False)
        x = torch.as_tensor(np.stack([norm(b) for b in batch]), dtype=torch.float32).to(device)
        st = torch.zeros(len(batch), 0, dtype=torch.float32, device=device)
        idx = torch.as_tensor([geno2idx.get(b["geno"], len(geno2idx)) for b in batch], dtype=torch.long).to(device)
        y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32).to(device)
        y_hat, env_hat, _ = module(x, None, idx, st)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        opt.zero_grad(); loss.backward(); opt.step()
    # encode with normalized features (items reference the raw x)
    norm_items = [{**i, "x": norm(i)} for i in items]
    z = encode_all(module, norm_items, device)
    dist = environment_distance(z)
    chosen = stratify_sample(env_ids, dist, n_bins=n_bins, target_total=target, seed=0)
    keep = set(chosen)
    return [i for i in items if i["env_id"] in keep]


if __name__ == "__main__":
    sys.exit(main())
