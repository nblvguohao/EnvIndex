"""envindex.sampling — distance-stratified environment sampling for LOEO.

Implements the recommendation in specs/loe_scaling_assessment.md: full LOEO
over thousands of environments is ~32 GPU-weeks, so the H2 boundary curve is
estimated on a distance-stratified sample of environments.

Steps
-----
1. encode every environment with a trained EnvIndex encoder -> z_e
2. for each environment, dist(e) = mean distance from z_e to its k nearest
   training-set neighbours (protocol §4.4)
3. stratify the environments into quantile bins by dist(e)
4. sample a target total uniformly across bins (strata)
"""

from __future__ import annotations

import numpy as np
import torch

from envindex.train import EnvIndexModule


@torch.no_grad()
def encode_all(
    module: EnvIndexModule,
    items: list[dict],
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Encode every unique environment in `items` to its z_e embedding.

    items: list of dicts with keys x, static, geno_idx, env_id (like the LOEO
    pilot items).  Returns {env_id: z_e}.
    """
    module.eval()
    out: dict[str, np.ndarray] = {}
    by_env: dict[str, list[torch.Tensor]] = {}
    for it in items:
        env = it["env_id"]
        x = torch.as_tensor(np.nan_to_num(it["x"]), dtype=torch.float32).unsqueeze(0).to(device)
        static = it.get("static")
        static_t = torch.as_tensor(static if static is not None else [], dtype=torch.float32).unsqueeze(0).to(device)
        idx = torch.as_tensor([it.get("geno_idx", 0)], dtype=torch.long).to(device)
        g_emb = it.get("g_emb")
        if g_emb is None:
            z = module.encode(x, static_t)
        else:
            g = torch.as_tensor(g_emb, dtype=torch.float32).unsqueeze(0).to(device)
            _, _, z = module(x, g, idx, static_t)
        by_env.setdefault(env, []).append(z.squeeze(0).cpu())
    for env, zs in by_env.items():
        out[env] = torch.stack(zs).mean(dim=0).numpy()
    return out


def environment_distance(
    z: dict[str, np.ndarray],
    k: int = 5,
) -> dict[str, float]:
    """dist(e) = mean distance to the k nearest OTHER environments.

    Note: a stricter definition (protocol §4.4) uses distance to the TRAINING
    set; here we use all other environments as a stand-in for the pilot.
    """
    ids = list(z.keys())
    mat = np.stack([z[i] for i in ids])
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    # pairwise cosine distances
    sim = norm @ norm.T
    dist = 1.0 - np.clip(sim, -1, 1)
    np.fill_diagonal(dist, np.inf)
    out = {}
    for j, i in enumerate(ids):
        kk = min(k, len(ids) - 1)
        out[i] = float(np.mean(np.sort(dist[j])[:kk]))
    return out


def stratify_sample(
    env_ids: list[str],
    dist: dict[str, float],
    n_bins: int = 8,
    target_total: int = 500,
    seed: int = 0,
) -> list[str]:
    """Uniformly sample `target_total` environments across dist-quantile bins."""
    rng = np.random.default_rng(seed)
    dists = np.array([dist[e] for e in env_ids])
    edges = np.quantile(dists, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9  # include max in last bin
    per_bin = max(1, target_total // n_bins)
    chosen: list[str] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [e for e in env_ids if lo <= dist[e] < hi]
        if not in_bin:
            continue
        take = min(per_bin, len(in_bin))
        chosen.extend(rng.choice(in_bin, size=take, replace=False).tolist())
    return chosen
