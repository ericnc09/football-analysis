"""
calibration.py
--------------
Post-hoc temperature scaling for HybridXGModel / HybridGATModel.

Temperature scaling (Guo et al., 2017) learns a single scalar T that divides
every raw logit before the sigmoid, pulling over-confident predictions toward
the calibrated centre without touching any model weights.

    p̂ = sigmoid(logit / T)

    T > 1  →  smooths distribution   (fixes over-confidence)
    T = 1  →  no-op
    T < 1  →  sharpens (never needed here)

T is found by minimising NLL on the held-out validation set with model
weights fully frozen. This is a 1-parameter convex problem; LBFGS converges
in < 50 steps.

Usage
-----
    from src.calibration import TemperatureScaler

    scaler = TemperatureScaler(trained_model)
    result = scaler.fit(val_loader, device="cpu")
    # {"T": 1.43, "brier_before": 0.178, "brier_after": 0.121, ...}

    scaler.save(Path("data/processed/pool_7comp_T.pt"))

    # Inference:
    scaler = TemperatureScaler.load(model, Path("data/processed/pool_7comp_T.pt"))
    logits = scaler(x, edge_index, batch, meta)          # T-scaled logits
    probs  = torch.sigmoid(logits).squeeze()
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


class TemperatureScaler(nn.Module):
    """
    Wraps any HybridXGModel / HybridGATModel and applies a learnable scalar T.

    Model weights are FROZEN during fit(); only log_T is optimised.

    Parameters
    ----------
    model   : trained model with .forward(x, edge_index, batch, metadata, ...)
    init_T  : starting temperature (>1 = already expects some smoothing)
    """

    def __init__(self, model: nn.Module, init_T: float = 1.5) -> None:
        super().__init__()
        self.model = model
        # Work in log-space so T stays positive throughout optimisation
        self.log_T = nn.Parameter(torch.tensor(float(np.log(init_T))))

    # ------------------------------------------------------------------ API

    @property
    def temperature(self) -> float:
        """Current temperature value (always > 0)."""
        return float(self.log_T.exp().item())

    def forward(
        self,
        x:         torch.Tensor,
        edge_index: torch.Tensor,
        batch:     torch.Tensor,
        metadata:  torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run base model (no_grad), then divide logit by T."""
        with torch.no_grad():
            logits = self.model(x, edge_index, batch, metadata, **kwargs)
        return logits / self.log_T.exp()

    def fit(
        self,
        val_loader: DataLoader,
        device:     str = "cpu",
        meta_fn:    Callable | None = None,
        lr:         float = 0.05,
        max_iter:   int   = 200,
    ) -> dict:
        """
        Optimise T on val_loader via NLL (binary cross-entropy). Weights frozen.

        Parameters
        ----------
        val_loader : DataLoader over validation set graphs
        device     : torch device string ("cpu" or "cuda")
        meta_fn    : optional callable(batch) → meta Tensor [n, META_DIM]
                     defaults to the standard 12-dim layout
        lr         : LBFGS learning rate
        max_iter   : maximum LBFGS iterations

        Returns
        -------
        dict
            T            : optimal temperature
            brier_before : Brier score before scaling
            brier_after  : Brier score after scaling
            nll_before   : NLL before scaling
            nll_after    : NLL after scaling
        """
        self.model.eval()
        self.model.to(device)
        self.log_T = nn.Parameter(self.log_T.detach().to(device))

        # ── 1. Collect frozen logits + labels from the base model ────────────
        logits_list, y_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                meta  = (_default_meta(batch, device)
                         if meta_fn is None else meta_fn(batch))
                logits = self.model(batch.x, batch.edge_index, batch.batch, meta)
                logits_list.append(logits.squeeze())
                y_list.append(batch.y.squeeze().float())

        logits_all = torch.cat(logits_list).detach()
        y_all      = torch.cat(y_list).detach()
        y_np       = y_all.cpu().numpy()

        # ── 2. Pre-calibration metrics ───────────────────────────────────────
        probs_before = torch.sigmoid(logits_all).cpu().numpy()
        brier_before = float(np.mean((probs_before - y_np) ** 2))
        nll_before   = float(F.binary_cross_entropy_with_logits(
            logits_all, y_all).item())

        # ── 3. Optimise T with LBFGS (ideal for 1-param convex problems) ────
        optimizer = torch.optim.LBFGS(
            [self.log_T], lr=lr, max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def _closure():
            optimizer.zero_grad()
            scaled = logits_all / self.log_T.exp()
            loss = F.binary_cross_entropy_with_logits(scaled, y_all)
            loss.backward()
            return loss

        optimizer.step(_closure)

        # ── 4. Post-calibration metrics ──────────────────────────────────────
        with torch.no_grad():
            scaled_logits = logits_all / self.log_T.exp()
            probs_after   = torch.sigmoid(scaled_logits).cpu().numpy()
            brier_after   = float(np.mean((probs_after - y_np) ** 2))
            nll_after     = float(F.binary_cross_entropy_with_logits(
                scaled_logits, y_all).item())

        result = {
            "T":            self.temperature,
            "brier_before": brier_before,
            "brier_after":  brier_after,
            "nll_before":   nll_before,
            "nll_after":    nll_after,
        }
        print(
            f"  [TemperatureScaler] T={result['T']:.4f}  "
            f"Brier {result['brier_before']:.4f} → {result['brier_after']:.4f}  "
            f"NLL {result['nll_before']:.4f} → {result['nll_after']:.4f}"
        )
        return result

    def save(self, path: Path) -> None:
        """Persist T (and only T) to a .pt file."""
        path = Path(path)
        torch.save({"T": self.temperature}, path)
        print(f"  [TemperatureScaler] saved T={self.temperature:.4f} → {path}")

    @classmethod
    def load(cls, model: nn.Module, path: Path) -> "TemperatureScaler":
        """
        Wrap *model* with the T stored in *path*.

        Parameters
        ----------
        model : trained model (weights already loaded)
        path  : .pt file produced by TemperatureScaler.save()
        """
        path = Path(path)
        if not path.exists():
            print(f"  [TemperatureScaler] no T file at {path}, using T=1.0 (no scaling)")
            return cls(model, init_T=1.0)
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        T    = float(ckpt["T"])
        scaler = cls(model, init_T=1.0)   # init doesn't matter — overwritten below
        with torch.no_grad():
            scaler.log_T.fill_(float(np.log(T)))
        print(f"  [TemperatureScaler] loaded T={T:.4f} from {path}")
        return scaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_meta(batch, device: str) -> torch.Tensor:
    """
    Build the standard 18-dim metadata tensor for HybridXGModel.
    Mirrors _metadata_tensor() in train_xg_hybrid.py.

    Layout: [shot_dist, shot_angle, is_header, is_open_play, technique×8,
             gk_dist, n_def_in_cone, gk_off_centre,
             gk_perp_offset, n_def_direct_line, is_right_foot]
    """
    n = batch.shot_dist.shape[0]

    base = torch.stack([
        batch.shot_dist.squeeze(),
        batch.shot_angle.squeeze(),
        batch.is_header.squeeze().float(),
        batch.is_open_play.squeeze().float(),
    ], dim=1)                              # [n, 4]
    tech = batch.technique.view(-1, 8)    # [n, 8]
    gk   = torch.stack([
        batch.gk_dist.squeeze(),
        batch.n_def_in_cone.squeeze(),
        batch.gk_off_centre.squeeze(),
    ], dim=1)                              # [n, 3]

    def _safe(attr, default):
        if hasattr(batch, attr):
            return getattr(batch, attr).squeeze()
        return torch.full((n,), default)

    new = torch.stack([
        _safe("gk_perp_offset",    3.0),
        _safe("n_def_direct_line", 0.0),
        _safe("is_right_foot",     0.5),
    ], dim=1)                              # [n, 3]

    return torch.cat([base, tech, gk, new], dim=1).to(device)  # [n, 18]
