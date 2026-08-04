"""envindex.encoder — the learned stage-aware environmental encoder (R2).

Implements the EnvIndex module from protocol_freeze_paper2.md §4.2:

  input:  stage-aware feature matrix X_e in R^{S x F} (S stages, F features)
          + static covariates s_e (soil / climate zone / grid cell — NOT
          precise lat/lon)
  model:  2-layer MLP-Mixer (token-mixing across stages, channel-mixing
          across features), < 2M parameters
  output: z_e in R^d, d in {8, 16, 32} (ablation dimension)

Multi-task training signals (see train.py):
  1. main   : z_e-conditioned yield prediction (jointly with the prediction
              head in heads.py)
  2. aux A  : environment mean yield read out linearly from z_e
  3. aux B  : InfoNCE contrastive (same year / same mega-environment closer;
              site identity excluded per protocol §4.2)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPMixerEncoder(nn.Module):
    """Two-layer MLP-Mixer over a (S, F) stage-feature matrix.

    Parameters
    ----------
    n_stages : number of phenological stages (S)
    n_features : number of weather/static features per stage (F)
    d_embed : embedding dimension d for z_e
    n_hidden : hidden width for mixer MLPs
    n_static : number of static covariates (0 = none)
    token_mix_factor / channel_mix_factor : expansion factors
    """

    def __init__(
        self,
        n_stages: int,
        n_features: int,
        d_embed: int = 16,
        n_hidden: int = 64,
        n_static: int = 0,
        token_mix_factor: int = 2,
        channel_mix_factor: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_stages = n_stages
        self.n_features = n_features
        self.d_embed = d_embed

        self.input_proj = nn.Linear(n_features, n_hidden)

        # MLP-Mixer blocks: token-mixing (across stages) + channel-mixing
        # (across features).
        self.mixer_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm([n_hidden]),
                    _TokenMixing(n_stages, n_hidden, token_mix_factor),
                    nn.LayerNorm([n_hidden]),
                    _ChannelMixing(n_hidden, n_hidden, channel_mix_factor),
                )
                for _ in range(2)
            ]
        )
        self.pre_pool_norm = nn.LayerNorm([n_hidden])

        in_dim = n_hidden + n_static
        self.embed_head = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, d_embed),
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        """Encode environment feature matrices to embeddings.

        x     : (batch, S, F) float tensor
        static: (batch, C) float tensor or None
        returns z_e : (batch, d_embed)
        """
        h = self.input_proj(x)  # (B, S, H)
        for block in self.mixer_blocks:
            h = block(h)
        h = self.pre_pool_norm(h)
        h = h.mean(dim=1)  # global mean pool over stages -> (B, H)
        if static is not None:
            h = torch.cat([h, static], dim=-1)
        return self.embed_head(h)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _TokenMixing(nn.Module):
    """MLP applied across the stage (token) dimension (transposed mixer)."""

    def __init__(self, n_stages: int, hidden: int, factor: int = 2) -> None:
        super().__init__()
        inner = max(1, int(n_stages * factor))
        self.mlp = nn.Sequential(
            nn.Linear(n_stages, inner),
            nn.GELU(),
            nn.Linear(inner, n_stages),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, H) -> transpose to mix across S
        return self.mlp(x.transpose(-1, -2)).transpose(-1, -2)


class _ChannelMixing(nn.Module):
    """MLP applied across the feature (channel) dimension."""

    def __init__(self, in_features: int, hidden: int, factor: int = 2) -> None:
        super().__init__()
        inner = max(1, int(hidden * factor))
        self.mlp = nn.Sequential(
            nn.Linear(in_features, inner),
            nn.GELU(),
            nn.Linear(inner, in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# --------------------------------------------------------------------------
# multi-task heads
# --------------------------------------------------------------------------

class EnvMeanHead(nn.Module):
    """Auxiliary task A: linear readout of environment mean yield from z_e."""

    def __init__(self, d_embed: int) -> None:
        super().__init__()
        self.head = nn.Linear(d_embed, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(-1)


class InteractionHead(nn.Module):
    """Main yield head: additive G + E + low-rank G∘z interaction (§4.3).

    y_hat_ij = mu + g_i + e(z_j) + <U g_i, V z_j>
    where g_i is a learned per-genotype main effect, e(z_j) a small MLP on the
    environment embedding, and U, V the rank-r interaction factors.
    """

    def __init__(
        self,
        d_geno: int,
        d_embed: int,
        rank: int = 2,
        n_genotypes: int | None = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        if n_genotypes is not None:
            self.geno_main = nn.Embedding(n_genotypes, 1)
        else:
            self.geno_main = None
        self.env_main = nn.Sequential(nn.Linear(d_embed, 32), nn.GELU(), nn.Linear(32, 1))
        self.U = nn.Parameter(torch.randn(d_geno, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(d_embed, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        z: torch.Tensor,
        g_emb: torch.Tensor,
        geno_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """z: (B, d_embed); g_emb: (B, d_geno); geno_idx: optional (B,) ints."""
        out = self.bias.expand(z.shape[0])
        if self.geno_main is not None and geno_idx is not None:
            out = out + self.geno_main(geno_idx).squeeze(-1)
        out = out + self.env_main(z).squeeze(-1)
        interaction = (g_emb @ self.U) * (z @ self.V)  # (B, rank)
        out = out + interaction.sum(dim=-1)
        return out


def infonce_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE auxiliary task B.

    z      : (B, d) normalized? embeddings
    labels : (B,) positive-pair group ids (e.g. same year, same mega-environment)
    Returns the contrastive loss; within a batch, positives are items sharing
    a label, negatives are all others.
    """
    z = torch.nn.functional.normalize(z, dim=-1)
    sim = z @ z.T  # (B, B)
    sim = sim / temperature
    mask = (labels[:, None] == labels[None, :]).float()
    mask.fill_diagonal_(0.0)
    # positive logits: where mask==1; negative: where mask==0
    # InfoNCE: -log( sum_pos exp(sim) / sum_all exp(sim) )
    exp_sim = torch.exp(sim)
    exp_sim = exp_sim * (1.0 - torch.eye(sim.shape[0], device=sim.device))
    sum_pos = (exp_sim * mask).sum(dim=-1)
    sum_all = exp_sim.sum(dim=-1) + 1e-8
    loss = -(torch.log((sum_pos + 1e-8) / sum_all)).mean()
    return loss
