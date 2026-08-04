"""Tests for EnvIndex v0 encoder / heads / training loop."""

from __future__ import annotations

import torch

from envindex.encoder import EnvMeanHead, InteractionHead, MLPMixerEncoder, infonce_loss
from envindex.train import EnvIndexModule, train_epoch


def test_mlp_mixer_shapes_and_param_count():
    torch.manual_seed(0)
    enc = MLPMixerEncoder(n_stages=3, n_features=8, d_embed=16, n_static=4)
    x = torch.randn(5, 3, 8)
    static = torch.randn(5, 4)
    z = enc(x, static)
    assert z.shape == (5, 16)
    assert enc.n_parameters() < 2_000_000  # protocol < 2M params


def test_encoder_without_static():
    enc = MLPMixerEncoder(n_stages=5, n_features=8, d_embed=8)
    z = enc(torch.randn(3, 5, 8), None)
    assert z.shape == (3, 8)


def test_interaction_head_rank_contribution():
    torch.manual_seed(0)
    head = InteractionHead(d_geno=32, d_embed=16, rank=2, n_genotypes=10)
    z = torch.randn(4, 16)
    g = torch.randn(4, 32)
    idx = torch.tensor([0, 1, 2, 3])
    y = head(z, g, idx)
    assert y.shape == (4,)
    # Interaction term is rank-r: perturbing z changes y.
    y2 = head(z + 0.1, g, idx)
    assert not torch.allclose(y, y2, atol=1e-4)


def test_env_mean_head():
    head = EnvMeanHead(16)
    out = head(torch.randn(6, 16))
    assert out.shape == (6,)


def test_infonce_loss_discriminates_groups():
    # Same-year items should be pulled together; uniform labels -> lower loss.
    torch.manual_seed(0)
    z_clustered = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    z_flat = torch.randn(8, 16)
    labels_flat = torch.arange(8)  # each its own group -> no positives
    l_cluster = infonce_loss(z_clustered, labels)
    l_flat = infonce_loss(z_flat, labels_flat)
    assert l_cluster < l_flat + 1e-6


def test_train_epoch_reduces_loss():
    torch.manual_seed(1)
    module = EnvIndexModule(n_stages=3, n_features=8, d_embed=8, d_geno=4, rank=2, n_genotypes=6)
    # build a tiny dataset
    items = []
    env_mean = {}
    for e in range(4):
        env_mean[f"e{e}"] = float(10 + e)
        for g in range(6):
            items.append(
                {
                    "x": torch.randn(3, 8),
                    "static": torch.zeros(0),
                    "g_emb": torch.randn(4),
                    "geno_idx": g,
                    "y": float(10 + e + 0.1 * g),
                    "env_id": f"e{e}",
                    "env_label": e,
                }
            )
    from torch.utils.data import DataLoader
    from envindex.train import EnvYieldDataset, collate

    ds = EnvYieldDataset(items, env_mean)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)
    opt = torch.optim.Adam(module.parameters(), lr=1e-2)
    l0 = train_epoch(module, loader, opt)
    l1 = train_epoch(module, loader, opt)
    assert l1["loss"] < l0["loss"] + 1e-6
