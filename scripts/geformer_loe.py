"""geformer_loe.py — GEFormer LOEO baseline on the corn cohort (protocol §5).

Adds the recent-SOTA GEFormer (Yao et al., Molecular Plant 2025) to the
EnvIndex baseline wall under the IDENTICAL LOEO protocol as the §7.2 rerun
cohort (load_corn(400, seed=0, plot_cap=100) -> 269 envs), so the comparison
with pcc_gz / pcc_fw / pcc_gm is apples-to-apples.

Data assembly (leakage-safe):
  * genotype : G2F dosage matrix (2425 markers) for the cohort genotypes
               (continuous dosages; GEFormer's gMLP takes them as input)
  * environment: fixed ENV_DAYS=150 window from the thermal season anchor
               (anchor_planting_date, the same origin the project's stage
               features use), edge-padded to ENV_DAYS.  Standardization is
               fit on TRAINING-fold envs only (never the held-out env).
  * temporal : month / day / weekday of each day in the window (as the paper)
  * LOEO     : train GEFormer on all-but-one env, predict held-out env,
               within-env PCC.  y is standardized per fold on training items
               (D1 rule) and predictions unscaled.

Fold parallelism: --fold-workers>1 uses a spawn ProcessPool (GPU round-robin
across --n-gpus, honoring LOE_GPU_BASE), mirroring run_loe.  Per-env weather
windows and per-genotype dosage vectors are shared read-only.

Output: per-env PCC parquet + paired bootstrap vs pcc_gz / pcc_fw / pcc_gm.

Usage:
    python scripts/geformer_loe.py --epochs 40 --device cuda \
        --fold-workers 16 --n-gpus 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT.parent / "nc" / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envindex.corn_features import anchor_planting_date  # noqa: E402
from envindex.geformer import GEFormer  # noqa: E402
from gxe_budget.data.preprocessing import normalize_weather_dates  # noqa: E402
from loe_pilot import _pcc, load_corn  # noqa: E402

G2F_WEATHER = ROOT.parent / "nc/data/processed/g2f/weather_daily.parquet"
G2F_GENO = ROOT.parent / "nc/data/processed/g2f/genotype.parquet"
REF_RESULTS = ROOT / "data/t3/loe_fix_learned_corn.parquet"

ENV_DAYS = 150          # fixed season window (days from thermal anchor)
ENV_FACTORS = ["tmax", "tmin", "tmean", "precipitation", "solar_radiation", "relative_humidity"]
OUT = ROOT / "data/t3/loe_corn_geformer.parquet"


def build_geno_matrix(geno_ids: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """(n_geno, n_marker) continuous dosage matrix for the cohort genotypes."""
    g = pd.read_parquet(G2F_GENO)
    g = g[g["genotype_id"].isin(geno_ids)]
    wide = g.pivot_table(index="genotype_id", columns="marker_id",
                         values="allele_dosage", aggfunc="mean")
    markers = sorted(wide.columns)
    wide = wide.reindex(geno_ids)
    X = wide.to_numpy(dtype=np.float64)          # (n_geno, n_marker)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)       # impute (rare)
    gpos = {gid: i for i, gid in enumerate(geno_ids)}
    return X.astype(np.float32), gpos


def build_env_windows() -> dict[str, dict]:
    """Per-env fixed ENV_DAYS daily window: {env_id: {series (D,C), marks (D,3)}}."""
    weather = pd.read_parquet(G2F_WEATHER)
    weather = normalize_weather_dates(weather)
    weather["environment_id"] = weather["environment_id"].astype(str)

    out = {}
    for env_id, g in weather.groupby("environment_id"):
        anchor = anchor_planting_date(g)
        if anchor is None:
            continue
        start = anchor - pd.Timedelta(days=1)    # day-0 = anchor day
        w = g.dropna(subset=["_weather_date"]).sort_values("_weather_date").copy()
        w = w[(w["_weather_date"] > start) & (w["_weather_date"] <= start + pd.Timedelta(days=ENV_DAYS))]
        if len(w) == 0:
            continue
        # edge-pad to exactly ENV_DAYS with the last available row
        if len(w) < ENV_DAYS:
            pad = w.iloc[[-1] * (ENV_DAYS - len(w))]
            w = pd.concat([w, pad], ignore_index=True)
        series = w[ENV_FACTORS].astype(np.float64).to_numpy()
        dts = pd.to_datetime(w["_weather_date"])
        marks = np.stack([dts.dt.month / 12.0, dts.dt.day / 31.0, dts.dt.weekday / 6.0], axis=1)
        out[env_id] = {"series": series.astype(np.float32),
                       "marks": marks.astype(np.float32)}
    return out


def _init_worker(items, params, counter, lock, n_gpus):
    """ProcessPool init: pin GPU (round-robin, LOE_GPU_BASE-aware) and share data."""
    global _SHARED
    device = params["device"]
    if device == "cuda":
        with lock:
            gpu = counter.value
            counter.value += 1
        base = int(os.environ.get("LOE_GPU_BASE", "0"))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(base + gpu % n_gpus)
    _SHARED = {"items": items, "params": params}


_SHARED: dict = {}


def _run_fold_worker(held):
    """One GEFormer LOEO fold in a worker process."""
    global _SHARED
    items, P = _SHARED["items"], _SHARED["params"]
    X = P["X"]                      # (n_geno, n_marker)
    win = P["win"]                  # {env_id: {series, marks}}
    device = P["device"]

    train_items = [i for i in items if i["env_id"] != held]
    held_items = [i for i in items if i["env_id"] == held]
    if len(train_items) < 50 or len(held_items) < 3:
        return held, None

    # --- fold-local standardization (training envs only; leakage-safe)
    y_mean = float(np.mean([i["y"] for i in train_items]))
    y_std = float(np.std([i["y"] for i in train_items]) + 1e-6)
    tr_ser = np.stack([win[i["env_id"]]["series"] for i in train_items])
    s_mean = tr_ser.mean(axis=(0, 1))
    s_std = tr_ser.std(axis=(0, 1)) + 1e-6

    def prep(it):
        env = win[it["env_id"]]
        return {"geno": X[it["geno_idx"]],
                "series": (env["series"] - s_mean) / s_std,
                "marks": env["marks"],
                "y": (it["y"] - y_mean) / y_std}

    tr = [prep(it) for it in train_items]
    he = [prep(it) for it in held_items]

    torch.manual_seed(0)
    np.random.seed(0)
    model = GEFormer(snp_len=X.shape[1], env_factor=len(ENV_FACTORS),
                     env_days=ENV_DAYS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"])
    n = len(tr)
    batch_size = P["batch_size"]
    model.train()
    for _ in range(P["epochs"]):
        perm = np.random.permutation(n)
        for j in range(0, n, batch_size):
            idx = perm[j : j + batch_size]
            geno = torch.from_numpy(np.stack([tr[k]["geno"] for k in idx])).to(device)
            ser = torch.from_numpy(np.stack([tr[k]["series"] for k in idx])).to(device)
            mark = torch.from_numpy(np.stack([tr[k]["marks"] for k in idx])).to(device)
            y = torch.from_numpy(np.array([tr[k]["y"] for k in idx], dtype=np.float32)).to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(geno, ser, mark).flatten(), y)
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for j in range(0, len(he), 512):
            batch = he[j : j + 512]
            geno = torch.from_numpy(np.stack([b["geno"] for b in batch])).to(device)
            ser = torch.from_numpy(np.stack([b["series"] for b in batch])).to(device)
            mark = torch.from_numpy(np.stack([b["marks"] for b in batch])).to(device)
            preds.append(model(geno, ser, mark).flatten().cpu().numpy())
    pred = np.concatenate(preds) * y_std + y_mean
    y_true = np.array([i["y"] for i in held_items])
    return held, _pcc(y_true, pred)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-envs", type=int, default=0, help="Cap envs for a quick pilot (0 = all 269)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--fold-workers", type=int, default=1,
                    help=">1 = spawn ProcessPool, GPU round-robin (mirrors run_loe)")
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    print(f"[geformer] loading corn cohort (load_corn 400/seed0/plotcap100) ...", flush=True)
    items, envs = load_corn(400, args.seed, plot_cap=100)
    if args.max_envs:
        envs = envs[: args.max_envs]
        items = [i for i in items if i["env_id"] in envs]
    print(f"[geformer] cohort: {len(items)} items, {len(envs)} envs", flush=True)

    genos = sorted(set(i["geno"] for i in items))
    X, gpos = build_geno_matrix(genos)
    print(f"[geformer] geno matrix {X.shape} (continuous dosages)", flush=True)
    for it in items:
        it["geno_idx"] = gpos[it["geno"]]

    win = build_env_windows()
    envs = [e for e in envs if e in win]
    items = [i for i in items if i["env_id"] in envs]
    print(f"[geformer] env windows ready for {len(envs)} envs", flush=True)

    # run_loe uses spawn only when >1 workers (CUDA can't re-init in a forked child)
    if args.fold_workers > 1:
        args.device = "cuda"
    params = {
        "X": X, "win": win, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "device": args.device,
    }
    envs = sorted(envs)
    t0 = time.perf_counter()

    if args.fold_workers <= 1:
        results = {}
        for fi, held in enumerate(envs):
            _, pcc = _run_fold_worker(held)
            if pcc is not None:
                results[held] = pcc
            if (fi + 1) % 10 == 0 or fi == len(envs) - 1:
                el = time.perf_counter() - t0
                print(f"[geformer] {fi+1}/{len(envs)} held={held} pcc={pcc:+.3f} "
                      f"(elapsed {el/60:.1f} min, {el/(fi+1):.1f}s/fold)", flush=True)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        counter = ctx.Manager().Value("i", 0)
        lock = ctx.Manager().Lock()
        results = {}
        with ProcessPoolExecutor(max_workers=args.fold_workers, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(items, params, counter, lock, args.n_gpus)) as pool:
            for fi, (held, pcc) in enumerate(pool.map(_run_fold_worker, envs)):
                if pcc is not None:
                    results[held] = pcc
                if (fi + 1) % 10 == 0 or fi == len(envs) - 1:
                    el = time.perf_counter() - t0
                    print(f"[geformer] {fi+1}/{len(envs)} held={held} pcc={pcc:+.3f} "
                          f"(elapsed {el/60:.1f} min, {el/(fi+1):.1f}s/fold)", flush=True)

    print(f"[geformer] mean within-env PCC = {np.nanmean(list(results.values())):+.4f} "
          f"(n={len(results)})", flush=True)
    df = pd.DataFrame([{"crop": "corn", "env_id": e, "pcc_geformer": v}
                       for e, v in results.items()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[geformer] -> {args.out}", flush=True)

    # paired bootstrap vs reference arms
    if REF_RESULTS.exists():
        ref = pd.read_parquet(REF_RESULTS).set_index("env_id")
        common = df.set_index("env_id").join(ref[["pcc_gz", "pcc_fw", "pcc_gm"]], how="inner")
        rng = np.random.default_rng(0)
        print("\n=== GEFormer vs reference arms (paired bootstrap, n={}) ===".format(args.n_boot))
        for col in ("pcc_gz", "pcc_fw", "pcc_gm"):
            d = (common["pcc_geformer"] - common[col]).dropna().to_numpy()
            boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(args.n_boot)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
            print(f"  GEFormer - {col}: {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] p={min(p,1):.4f} (n={len(d)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
