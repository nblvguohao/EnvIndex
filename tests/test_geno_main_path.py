"""Regression tests for the genotype main-effect path (amendment 2026-08-06(b) D2).

`InteractionHead.geno_main` used to inherit `nn.Embedding`'s default N(0,1)
init.  A genotype main effect is small next to an sd~1 random offset, and AdamW
moves a parameter only ~lr per step, so the random init dominated: the additive
path emitted noise (within-environment PCC ~0) while the multiplicative
interaction term -- initialised at 0.02 -- silently absorbed the main effect.
On barley this inflated the rank=4 arm to +0.388 and collapsed rank=0 to +0.007;
after the fix rank=0 reaches +0.360 and rank=4 drops to +0.299, i.e. the sign of
the interaction term's contribution reverses.

These tests fail loudly if that path ever goes silent again.
"""

from __future__ import annotations

import numpy as np
import torch

from envindex.encoder import InteractionHead


def test_geno_main_is_zero_initialised():
    """A main-effect term must start at zero, not at N(0,1) noise."""
    head = InteractionHead(d_geno=8, d_embed=8, rank=2, n_genotypes=64)
    w = head.geno_main.weight.detach()
    assert torch.allclose(w, torch.zeros_like(w)), (
        f"geno_main must be zero-initialised; got sd={w.std():.3f}, "
        f"range [{w.min():.2f}, {w.max():.2f}]"
    )


def test_rank0_head_learns_genotype_ranking():
    """A pure-additive head (rank=0) must recover genotype ranking.

    With no interaction term there is no other path to genotype signal, so a
    near-zero correlation here means the main-effect path is broken -- exactly
    the D2 failure mode.
    """
    torch.manual_seed(0)
    n_geno, n_env, d = 40, 12, 8
    true_g = torch.randn(n_geno)  # genotype main effects
    z_env = torch.randn(n_env, d)  # environment embeddings

    gi = torch.arange(n_geno).repeat(n_env)
    ei = torch.arange(n_env).repeat_interleave(n_geno)
    z = z_env[ei]
    y = true_g[gi] + 0.5 * z.sum(-1) + 0.05 * torch.randn(len(gi))
    y = (y - y.mean()) / y.std()  # standardised target, as the fold code does

    head = InteractionHead(d_geno=d, d_embed=d, rank=0, n_genotypes=n_geno)
    g_emb = torch.zeros(len(gi), d)  # rank=0 ignores g_emb
    opt = torch.optim.AdamW(head.parameters(), lr=3e-2)
    for _ in range(400):
        loss = torch.nn.functional.mse_loss(head(z, g_emb, gi), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    learned = head.geno_main.weight.detach().squeeze(-1).numpy()
    r = float(np.corrcoef(learned, true_g.numpy())[0, 1])
    assert r > 0.8, f"genotype main-effect path did not learn (corr={r:.3f})"


def test_rank0_forward_has_no_interaction():
    """rank=0 must degenerate cleanly to bias + geno_main + env_main."""
    head = InteractionHead(d_geno=8, d_embed=8, rank=0, n_genotypes=16)
    assert head.U.shape == (8, 0) and head.V.shape == (8, 0)
    z = torch.randn(5, 8)
    idx = torch.arange(5)
    out = head(z, torch.randn(5, 8), idx)
    # with g_emb varying but rank=0, predictions must not depend on g_emb
    out2 = head(z, torch.randn(5, 8) * 100, idx)
    assert torch.allclose(out, out2), "rank=0 head still responds to g_emb"
