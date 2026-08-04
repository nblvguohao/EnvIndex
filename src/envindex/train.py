"""envindex.train — multi-task training loop for the EnvIndex encoder.

Loss = L_main + lambda_a * L_auxA + lambda_b * L_auxB

  L_main : yield prediction via InteractionHead (G+E + low-rank G∘z)
  L_auxA : environment mean yield read out linearly from z_e
  L_auxB : InfoNCE contrastive over environment embeddings (same-year / same
           mega-environment positives; site identity excluded per §4.2)

Only environments in the training fold contribute gradient (leakage safety,
protocol §6): environment-level statistics (means) are computed from the
training set only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from envindex.encoder import EnvMeanHead, InteractionHead, MLPMixerEncoder, infonce_loss


class EnvIndexModule(nn.Module):
    """Bundle: encoder + aux A head + main interaction head."""

    def __init__(
        self,
        n_stages: int,
        n_features: int,
        d_embed: int = 16,
        d_geno: int = 32,
        rank: int = 2,
        n_static: int = 0,
        n_genotypes: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = MLPMixerEncoder(
            n_stages=n_stages,
            n_features=n_features,
            d_embed=d_embed,
            n_static=n_static,
        )
        self.aux_env_mean = EnvMeanHead(d_embed)
        self.main_head = InteractionHead(
            d_geno=d_geno,
            d_embed=d_embed,
            rank=rank,
            n_genotypes=n_genotypes,
        )
        self.d_embed = d_embed

    def encode(self, x: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x, static)

    def forward(
        self,
        x: torch.Tensor,
        g_emb: torch.Tensor,
        geno_idx: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (y_hat, env_mean_hat, z)."""
        z = self.encode(x, static)
        y_hat = self.main_head(z, g_emb, geno_idx)
        env_mean_hat = self.aux_env_mean(z)
        return y_hat, env_mean_hat, z


class EnvYieldDataset(Dataset):
    """Dataset of environment stage-features + genotypes + yields.

    items : list of dicts with keys
        x        (S, F) stage-feature matrix
        static   (C,) static covariates or None
        g_emb    (d_geno,) genotype embedding (fixed, e.g. marker PCA)
        geno_idx int genotype index (for learned per-genotype main effect)
        y        float yield
        env_id   str
        env_label  int group id for InfoNCE (e.g. year index)
    """

    def __init__(self, items: list[dict], env_mean_by_id: dict[str, float]) -> None:
        self.items = items
        self.env_mean_by_id = env_mean_by_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        static = it.get("static")
        if static is None:
            static = []
        elif isinstance(static, torch.Tensor) and static.numel() == 0:
            static = static
        return {
            "x": torch.as_tensor(it["x"], dtype=torch.float32),
            "static": torch.as_tensor(static, dtype=torch.float32),
            "g_emb": torch.as_tensor(it["g_emb"], dtype=torch.float32),
            "geno_idx": torch.as_tensor(it.get("geno_idx", -1), dtype=torch.long),
            "y": torch.as_tensor(it["y"], dtype=torch.float32),
            "env_mean": torch.as_tensor(self.env_mean_by_id[it["env_id"]], dtype=torch.float32),
            "env_label": torch.as_tensor(it["env_label"], dtype=torch.long),
        }


def collate(items: list[dict]) -> dict:
    return {
        "x": torch.stack([i["x"] for i in items]),
        "static": torch.stack([i["static"] for i in items]),
        "g_emb": torch.stack([i["g_emb"] for i in items]),
        "geno_idx": torch.stack([i["geno_idx"] for i in items]),
        "y": torch.stack([i["y"] for i in items]),
        "env_mean": torch.stack([i["env_mean"] for i in items]),
        "env_label": torch.stack([i["env_label"] for i in items]),
    }


def train_epoch(
    module: EnvIndexModule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_a: float = 1.0,
    lambda_b: float = 1.0,
    temperature: float = 0.1,
    device: str = "cpu",
) -> dict[str, float]:
    module.train()
    total = {"loss": 0.0, "main": 0.0, "aux_a": 0.0, "aux_b": 0.0, "n": 0}
    for batch in loader:
        x = batch["x"].to(device)
        static = batch["static"].to(device)
        g_emb = batch["g_emb"].to(device)
        geno_idx = batch["geno_idx"].to(device)
        y = batch["y"].to(device)
        env_mean = batch["env_mean"].to(device)
        env_label = batch["env_label"].to(device)

        y_hat, env_mean_hat, z = module(x, g_emb, geno_idx, static)

        loss_main = nn.functional.mse_loss(y_hat, y)
        loss_aux_a = nn.functional.mse_loss(env_mean_hat, env_mean)
        loss_aux_b = infonce_loss(z, env_label, temperature=temperature)
        loss = loss_main + lambda_a * loss_aux_a + lambda_b * loss_aux_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total["loss"] += loss.item() * len(y)
        total["main"] += loss_main.item() * len(y)
        total["aux_a"] += loss_aux_a.item() * len(y)
        total["aux_b"] += loss_aux_b.item() * len(y)
        total["n"] += len(y)

    return {k: v / max(total["n"], 1) for k, v in total.items() if k != "n"}


@torch.no_grad()
def evaluate(
    module: EnvIndexModule,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Return validation metrics: main MSE, aux A MSE, yield-PCC, envmean-PCC."""
    module.eval()
    ys, yhats, ems, emhats = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        static = batch["static"].to(device)
        g_emb = batch["g_emb"].to(device)
        geno_idx = batch["geno_idx"].to(device)
        y_hat, env_mean_hat, _ = module(x, g_emb, geno_idx, static)
        ys.append(batch["y"])
        yhats.append(y_hat.cpu())
        ems.append(batch["env_mean"])
        emhats.append(env_mean_hat.cpu())
    y = torch.cat(ys)
    yh = torch.cat(yhats)
    em = torch.cat(ems)
    emh = torch.cat(emhats)
    mse_main = float(nn.functional.mse_loss(yh, y))
    mse_aux = float(nn.functional.mse_loss(emh, em))
    return {
        "mse_main": mse_main,
        "mse_envmean": mse_aux,
        "pcc_yield": _pcc(yh, y),
        "pcc_envmean": _pcc(emh, em),
    }


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / (denom + 1e-8))
