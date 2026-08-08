"""GEFormer — reimplementation of the GEFormer genotype-environment
interaction predictor (Yao et al., Molecular Plant 2025, doi:10.1016/j.molp.2025.01.020)
as a leakage-safe LOEO baseline for the EnvIndex benchmark wall.

Why reimplement rather than vendor the authors' release:
    The official repository (github.com/Deep-Breeding/GEFormer) is hardcoded
    to the authors' exact maize data layout:
      * TimeFeatureBlock:  DataEmbedding(d_model=128) feeds an encoder with
        d_model=126 (silent width mismatch), and ODConv1d(env_days, env_days, 3)
        is called on a (B, d_model, env_days) tensor, which only lines up when
        d_model == env_days.  `conv_env` (Conv1d in_channels=75) and `fc2`
        (Linear(125, 76)) are dead layers tuned to the 75-factor demo env set.
      * gMLPVision:  `num_patches = 200` is hardcoded while the SNP input is
        laid out as a single patch (patch_size == image_size), so the
        SpatialGatingUnit weight (200 x 200) is mis-scaled for the actual
        sequence length.
    None of these are architectural choices; they are dataset-bound dimension
    accidents.  This module re-implements the same architecture with
    consistent, parameterized dimensions so it can run on the EnvIndex
    six-factor daily-weather schema (tmax, tmin, tmean, precipitation,
    solar_radiation, relative_humidity) under the project's LOEO protocol.

Architecture (per the paper, kept faithful):
    * genotype branch   : gMLP over the SNP dosage vector  -> g in R^126
    * environment branch: DataEmbedding (token + positional + temporal
        month/day/weekday) -> ODConv (omni-dim dynamic conv) -> ProbSparse
        linear-attention encoder (2 layers, 6 heads) -> temporal pooling
                                                          -> e in R^126
    * fusion            : CrossGatedMLP on the pairings (g,e), (g,g*e),
                          (e,g*e) -> concat  -> 756-dim
    * head              : MLP 756 -> 256 -> 32 -> 1  (LeakyReLU + dropout)

LOEO usage: see scripts/geformer_loe.py.  The environment window is fixed at
ENV_DAYS = 140 days from the season anchor (thermal origin), standardized per
fold on TRAINING environments only (leakage-safe).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# --------------------------------------------------------------------------
# genotype branch: gMLP over the SNP vector
# --------------------------------------------------------------------------


class _SpatialGatingUnit(nn.Module):
    """Spatial gating unit from the gMLP paper (single-head, single token)."""

    def __init__(self, dim: int, dim_seq: int) -> None:
        super().__init__()
        dim_out = dim // 2
        self.norm = nn.LayerNorm(dim_out)
        weight = torch.zeros(1, dim_seq, dim_seq)
        nn.init.uniform_(weight, -1e-3 / dim_seq, 1e-3 / dim_seq)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.ones(1, dim_seq))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, dim_ff); n is the token count (=1 for whole-genome token)
        res, gate = x.chunk(2, dim=-1)                 # (B, n, dim_ff/2)
        gate = self.norm(gate)
        gate = rearrange(gate, "b n (h d) -> b h n d", h=1)
        gate = torch.einsum("b h n d, h m n -> b h m d", gate, self.weight)
        gate = gate + rearrange(self.bias, "h n -> () h n ()")
        gate = rearrange(gate, "b h n d -> b n (h d)")
        return gate * res


class _GMLPBlock(nn.Module):
    def __init__(self, dim: int, dim_ff: int, seq_len: int) -> None:
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(dim, dim_ff), nn.GELU())
        self.sgu = _SpatialGatingUnit(dim_ff, seq_len)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.sgu(x)
        return self.proj_out(x)


class GenotypeGMLP(nn.Module):
    """gMLP over the SNP dosage vector (a single whole-genome token).

    x: (B, snp_len) dosages -> (B, dim)
    """

    def __init__(self, snp_len: int, dim: int = 126, depth: int = 2,
                 ff_mult: int = 4) -> None:
        super().__init__()
        dim_ff = dim * ff_mult
        self.to_patch_embed = nn.Linear(snp_len, dim)   # (B, snp_len) -> (B, dim)
        self.layers = nn.ModuleList(
            [_GMLPBlock(dim, dim_ff, seq_len=1) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embed(x).unsqueeze(1)   # (B, 1, dim)
        for layer in self.layers:
            x = layer(x)
        return self.head(x).squeeze(1)            # (B, dim)


# --------------------------------------------------------------------------
# environment branch: token embedding + ODConv + ProbSparse attention
# --------------------------------------------------------------------------


class _TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, kernel: int = 3) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.conv = nn.Conv1d(c_in, d_model, kernel, padding=padding,
                              padding_mode="circular")
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in",
                                nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L) -> (B, L, d_model)
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = (torch.arange(0, d_model, 2).float()
               * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class _TemporalEmbedding(nn.Module):
    """month/day/weekday -> d_model (3 temporal features, as the paper)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Linear(3, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class _DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.value = _TokenEmbedding(c_in, d_model)
        self.position = _PositionalEmbedding(d_model)
        self.temporal = _TemporalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value(x) + self.position(x) + self.temporal(x_mark))


class _ProbAttention(nn.Module):
    """ProbSparse self-attention (Informer): top-u queries by KL sparsity."""

    def __init__(self, factor: int = 5, attention_dropout: float = 0.05) -> None:
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_qk(self, Q, K, sample_k: int, n_top: int):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        QK_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = QK_sample.max(-1)[0] - torch.div(QK_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None], M_top, :]
        QK = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return QK, M_top

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        U_part = min(self.factor * int(math.ceil(math.log(L_K))), L_K)
        u = min(self.factor * int(math.ceil(math.log(L_Q))), L_Q)
        scores, index = self._prob_qk(queries, keys, sample_k=U_part, n_top=u)
        scores = scores / math.sqrt(D)
        # mean initial context (no causal mask)
        context = values.mean(dim=-2).unsqueeze(-2).expand(B, H, L_Q, D).clone()
        attn = torch.softmax(scores, dim=-1)
        context[torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, values)
        return context.transpose(2, 1).contiguous(), attn


class _AttentionLayer(nn.Module):
    def __init__(self, attention, d_model: int, n_heads: int) -> None:
        super().__init__()
        d_k = d_model // n_heads
        self.inner = attention
        self.q = nn.Linear(d_model, d_k * n_heads)
        self.k = nn.Linear(d_model, d_k * n_heads)
        self.v = nn.Linear(d_model, d_k * n_heads)
        self.out = nn.Linear(d_k * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):
        B, L, _ = q.shape
        H = self.n_heads
        q = self.q(q).view(B, L, H, -1)
        k = self.k(k).view(B, L, H, -1)
        v = self.v(v).view(B, L, H, -1)
        out, _ = self.inner(q, k, v, attn_mask)
        return self.out(out.view(B, L, -1))


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048,
                 dropout: float = 0.05) -> None:
        super().__init__()
        self.attn = _AttentionLayer(_ProbAttention(factor=5, attention_dropout=dropout),
                                    d_model, n_heads)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(x, x, x))
        y = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class _ConvLayer(nn.Module):
    """Downsampling conv (halves sequence length)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="circular")
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        x = self.norm(x)
        x = F.elu(x)
        return self.pool(x).transpose(1, 2)


class TimeFeatureBlock(nn.Module):
    """Environment encoder: embedding -> ODConv -> attention encoder -> pooling.

    x_enc  : (B, L, C)  standardized daily weather factors
    x_mark : (B, L, 3)  temporal features (month, day, weekday)
    -> (B, d_model)
    """

    def __init__(self, env_factor: int, env_days: int, d_model: int = 126,
                 n_heads: int = 6) -> None:
        super().__init__()
        self.embedding = _DataEmbedding(env_factor, d_model, dropout=0.05)
        self.odconv = ODConv1d(d_model, d_model, kernel_size=3)
        self.encoder = nn.Sequential(
            _EncoderLayer(d_model, n_heads), _ConvLayer(d_model),
            _EncoderLayer(d_model, n_heads),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_enc, x_mark):
        e = self.embedding(x_enc, x_mark)          # (B, L, d_model)
        e = self.odconv(e.transpose(1, 2))          # (B, d_model, L)
        e = self.encoder(e.transpose(1, 2))          # (B, L/2, d_model)
        e = self.pool(e.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        return e


# --------------------------------------------------------------------------
# omni-dimensional dynamic convolution (ODConv1d)
# --------------------------------------------------------------------------

class _ODAttn(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int,
                 reduction: float = 0.0625, kernel_num: int = 4) -> None:
        super().__init__()
        attention_channel = max(int(in_planes * reduction), 16)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(attention_channel)
        self.channel_fc = nn.Conv1d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = nn.Conv1d(attention_channel, out_planes, 1, bias=True)
        self.spatial_fc = nn.Conv1d(attention_channel, kernel_size, 1, bias=True)
        self.kernel_fc = nn.Conv1d(attention_channel, kernel_num, 1, bias=True)

    def forward(self, x):
        y = self.avgpool(x)
        y = self.bn(F.relu(self.fc(y)))
        return (torch.sigmoid(self.channel_fc(y).view(x.size(0), -1, 1)),
                torch.sigmoid(self.filter_fc(y).view(x.size(0), -1, 1)),
                torch.sigmoid(self.spatial_fc(y).view(x.size(0), 1, 1, 1, self.kernel_size)),
                F.softmax(self.kernel_fc(y).view(x.size(0), -1, 1, 1, 1), dim=1))


class ODConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, kernel_num=4,
                 reduction=0.0625) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.attn = _ODAttn(in_planes, out_planes, kernel_size,
                            reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes,
                                               in_planes, kernel_size))
        for i in range(kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode="fan_out",
                                    nonlinearity="relu")

    def forward(self, x):
        # x: (B, in_planes, L)
        ca, fa, sa, ka = self.attn(x)
        x = x * ca
        agg = sa * ka * self.weight.unsqueeze(0)
        agg = torch.sum(agg, dim=1).view(-1, self.in_planes, self.kernel_size)
        out = F.conv1d(x.view(1, -1, x.size(-1)), weight=agg, groups=x.size(0),
                       padding=self.kernel_size // 2)
        out = out.view(x.size(0), self.out_planes, -1) * fa
        return out


# --------------------------------------------------------------------------
# fusion + head
# --------------------------------------------------------------------------

class _CrossGatedMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                  nn.Linear(dim, dim), nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                  nn.Linear(dim, dim), nn.GELU())
        self.gate1 = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x1, x2):
        h1, h2 = self.mlp1(x1), self.mlp2(x2)
        g1, g2 = self.gate1(x1), self.gate2(x2)
        return torch.cat([(1 - g1) * h1 + g2 * h2, (1 - g2) * h2 + g1 * h1], dim=1)


class GEFormer(nn.Module):
    """GEFormer: gMLP(geno) + TimeFeatureBlock(env) + CrossGatedMLP fusion.

    Args:
        snp_len   : number of SNP dosage markers per genotype
        env_factor: number of daily weather factors per day
        env_days  : fixed number of days in the season window
        d_model   : latent dim (paper uses 126)
        depth     : gMLP depth
        neurons   : (256, 32) head hidden sizes
    """

    def __init__(self, snp_len: int, env_factor: int, env_days: int,
                 d_model: int = 126, depth: int = 2, dropout: float = 0.3,
                 neurons: tuple[int, int] = (256, 32)) -> None:
        super().__init__()
        self.gmlp = GenotypeGMLP(snp_len, dim=d_model, depth=depth)
        self.env_block = TimeFeatureBlock(env_factor, env_days, d_model=d_model)
        self.cgmlp = _CrossGatedMLP(d_model)
        # each CrossGatedMLP returns 2*d_model (concat of two 126-dim fusions);
        # three pairings -> 6*d_model = 756 at d_model=126 (paper head input).
        self.head = nn.Sequential(
            nn.Linear(6 * d_model, neurons[0]), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(neurons[0], neurons[1]), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(neurons[1], 1),
        )

    def forward(self, geno: torch.Tensor, x_enc: torch.Tensor,
                x_mark: torch.Tensor) -> torch.Tensor:
        """geno: (B, snp_len); x_enc: (B, L, C); x_mark: (B, L, 3)."""
        g = self.gmlp(geno)                # (B, d_model)
        e = self.env_block(x_enc, x_mark)  # (B, d_model)
        ge = g * e
        fused = torch.cat([self.cgmlp(g, e), self.cgmlp(g, ge),
                           self.cgmlp(e, ge)], dim=1)  # (B, 6*d_model)
        return self.head(fused)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
