"""synthetic_gxe_control.py — positive control: can the LOEO pipeline detect
a KNOWN interaction strength?

Motivation (2026-08-06 session): D1/D2 showed the genotype main-effect path
was silently dead for a long time while overall PCC looked "normal" -- we
had no way to notice.  Before trusting "Delta_FW <= 0 / rank=4 < rank=0" as
a real finding about G×E, we need to know the pipeline can recover a G×E
signal of known strength at all, and how strong that signal must be to be
detectable given per-crop N and noise.

Design: use REAL environment feature matrices (barley cache, realistic
input distribution) with SYNTHETIC genotypes and a known low-rank
interaction of strength lambda:

    y_ij = mu + g_i + h(e_j) + lambda * <a_i, b_j> + noise

`g_i` genotype main effects, `h(e_j)` a smooth environment main effect (both
independent of the interaction so removing rank does not remove them),
`<a_i, b_j>` a rank-r bilinear interaction (env loadings b_j drawn from a
random projection of the real env feature matrix, so the model has to find
it from x, not memorize it), lambda controls injected strength.

For each lambda, run the SAME rank=0 vs rank=4 LOEO comparison used on real
data.  Report PCC(rank=4) - PCC(rank=0) vs lambda -- this is the detection
curve.  If the pipeline is healthy, this gain must be ~0 at lambda=0 and
increase with lambda; if a real dataset's gain is <= 0, that's only
interpretable once we know the detection floor from this curve.

Usage:
    python scripts/synthetic_gxe_control.py --lambdas 0,0.05,0.1,0.25,0.5,1.0,2.0
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

from loe_pilot import run_loe  # noqa: E402


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env-features", type=Path, default=ROOT / "data/t3/barley_env_features.pkl",
                        help="Pickle of {env_id: x_matrix} to reuse as realistic environment inputs")
    parser.add_argument("--n-geno", type=int, default=80)
    parser.add_argument("--interaction-rank", type=int, default=2,
                        help="True rank of the injected interaction (model is fit at --model-rank)")
    parser.add_argument("--model-rank", type=int, default=4)
    parser.add_argument("--main-effect-sd", type=float, default=1.0)
    parser.add_argument("--noise-sd", type=float, default=0.5)
    parser.add_argument("--lambdas", type=str, default="0,0.1,0.25,0.5,1.0,2.0",
                        help="Comma-separated injected interaction strengths")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--fold-workers", type=int, default=4)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=ROOT / "data/t3/synthetic_gxe_detection_curve.csv")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def build_synthetic_items(env_x: dict, n_geno: int, true_rank: int, main_sd: float,
                           noise_sd: float, lam: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    env_ids = sorted(env_x)
    n_env = len(env_ids)
    flat_dim = int(np.prod(env_x[env_ids[0]].shape))

    # environment main effect: smooth function of the real features (a random
    # linear readout), NOT the interaction signal -- present at every lambda
    # so removing the interaction (rank=0) still lets env_main do its job,
    # mirroring the real model's env_main(z) head.
    w_env_main = rng.normal(size=flat_dim) / np.sqrt(flat_dim)
    # interaction: env loadings from a DIFFERENT random projection of the raw
    # features (rank = true_rank), genotype loadings iid.
    W_int = rng.normal(size=(flat_dim, true_rank)) / np.sqrt(flat_dim)
    geno_main = rng.normal(scale=main_sd, size=n_geno)
    geno_load = rng.normal(size=(n_geno, true_rank))

    env_main_val = {}
    env_load = {}
    for e in env_ids:
        flat = np.nan_to_num(env_x[e]).ravel()
        env_main_val[e] = float(flat @ w_env_main) * 0.3
        env_load[e] = flat @ W_int  # (true_rank,)

    items = []
    for gi in range(n_geno):
        geno = f"syn_g{gi}"
        for e in env_ids:
            interaction = lam * float(geno_load[gi] @ env_load[e])
            y = geno_main[gi] + env_main_val[e] + interaction + rng.normal(scale=noise_sd)
            items.append({
                "env_id": e, "geno": geno, "y": float(y),
                "x": env_x[e], "env_label": 0,
            })
    return items


def main(argv=None):
    args = _parse_args(argv)
    import pickle
    with open(args.env_features, "rb") as f:
        env_x = pickle.load(f)
    print(f"[synthetic] {len(env_x)} real environments loaded from {args.env_features}")

    lambdas = [float(x) for x in args.lambdas.split(",")]
    rows = []
    for lam in lambdas:
        items = build_synthetic_items(env_x, args.n_geno, args.interaction_rank,
                                       args.main_effect_sd, args.noise_sd, lam, args.seed)
        print(f"\n[synthetic] lambda={lam}  {len(items)} items, "
              f"{len(set(i['env_id'] for i in items))} envs, {args.n_geno} genotypes (dense design)")

        res0 = run_loe(items, d_embed=32, d_geno=32, rank=0, epochs=args.epochs, device=args.device,
                       seed=args.seed, batch_size=512, fold_workers=args.fold_workers, n_gpus=args.n_gpus)
        res4 = run_loe(items, d_embed=32, d_geno=32, rank=args.model_rank, epochs=args.epochs, device=args.device,
                       seed=args.seed, batch_size=512, fold_workers=args.fold_workers, n_gpus=args.n_gpus)

        common = sorted(set(res0) & set(res4))
        gz0 = np.array([res0[e]["pcc_gz"] for e in common])
        gz4 = np.array([res4[e]["pcc_gz"] for e in common])
        gain = gz4 - gz0
        rng = np.random.default_rng(0)
        boot = [rng.choice(gain, len(gain), replace=True).mean() for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        row = {"lambda": lam, "pcc_rank0": float(np.nanmean(gz0)), "pcc_rank4": float(np.nanmean(gz4)),
               "gain": float(np.nanmean(gain)), "gain_ci_lo": float(lo), "gain_ci_hi": float(hi),
               "n_env": len(common)}
        rows.append(row)
        print(f"[synthetic] lambda={lam}: PCC(rank0)={row['pcc_rank0']:+.4f}  "
              f"PCC(rank{args.model_rank})={row['pcc_rank4']:+.4f}  "
              f"gain={row['gain']:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
              f"{'DETECTED' if lo > 0 else 'not detected'}")

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[synthetic] detection curve -> {args.out_csv}")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
