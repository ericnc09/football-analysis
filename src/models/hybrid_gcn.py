"""
hybrid_gcn.py
-------------
Canonical HybridXGModel — the GCN-based hybrid xG model.

Single source of truth for the class formerly duplicated across
`app.py`, `scripts/train_xg_hybrid.py`, `scripts/calibrate_and_train_gat.py`
and `scripts/generate_match_report.py`.

Architecture
------------
  GCN path  : node features → 3× GCNConv → global_mean_pool → hidden_dim
  Meta path : hand-crafted shot features (length = meta_dim)
  Head      : Linear(hidden_dim + meta_dim, hidden_dim) → ReLU → Dropout
              → Linear(hidden_dim, 1)  — raw logit

Contract
--------
Every saved checkpoint ships with a `.meta.json` sidecar declaring
`node_in`, `hidden_dim`, `meta_dim`, `dropout`, and
`feature_schema_version`. At load time the consumer (serving layer,
calibration, tests) reads those from the sidecar rather than peeking at
tensor shapes — see `src/model_metadata.py` and `src/serving.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class HybridXGModel(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Number of node-feature dimensions.
    hidden_dim : int, default 64
        Width of the GCN layers and the first MLP-head layer.
    meta_dim : int, default 27
        Dimensionality of the shot-metadata tensor concatenated before the
        head. Must equal the `meta_dim` declared in the sidecar of any
        checkpoint loaded into this model.
    dropout : float, default 0.3
        Dropout applied inside the GCN stack and between head layers.

    Notes
    -----
    `forward` accepts an optional `edge_attr` keyword for API parity with
    `HybridGATModel` (GATv2Conv uses edge features). GCNConv does not
    consume edge attributes, so the argument is silently ignored here —
    this lets the serving layer call both models through a uniform
    interface without branching.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        meta_dim: int = 27,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_dim),
            GCNConv(hidden_dim,  hidden_dim),
            GCNConv(hidden_dim,  hidden_dim),
        ])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    # ---------------------------------------------------------------- encode

    def encode(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
    ) -> torch.Tensor:
        """Graph-level embedding before the MLP head."""
        for conv in self.convs:
            x = F.dropout(
                F.relu(conv(x, edge_index)),
                p=self.dropout,
                training=self.training,
            )
        return global_mean_pool(x, batch)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
        metadata:   torch.Tensor,
        edge_attr:  torch.Tensor | None = None,   # noqa: ARG002 — API parity
    ) -> torch.Tensor:
        """Return (B, 1) raw logits. `edge_attr` ignored (see class docstring)."""
        emb = self.encode(x, edge_index, batch)
        return self.head(torch.cat([emb, metadata], dim=1))
