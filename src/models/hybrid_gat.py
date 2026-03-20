"""
hybrid_gat.py
-------------
HybridGATModel — drop-in replacement for HybridXGModel that uses GATv2Conv
instead of GCNConv so per-edge attention weights can be extracted at inference
time and visualised in the Shot Inspector ("attention overlay").

Architecture
------------
  GATv2Conv × n_layers  →  global_mean_pool
  →  cat(graph_embedding, shot_metadata)  →  MLP head  →  logit

Attention extraction
--------------------
  Call forward_with_attention() instead of forward().
  Returns (logit, [alpha_layer_0, ..., alpha_layer_N])
  Each alpha_i has shape (E, heads).

  To get a single scalar weight per edge for visualisation:
      alpha = torch.stack(alphas).mean(dim=0)   # (E, heads)  — mean over layers
      alpha = alpha.mean(dim=1)                 # (E,)         — mean over heads
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

META_DIM = 12   # shot_dist, shot_angle, is_header, is_open_play + technique×8


class HybridGATModel(nn.Module):
    """
    GATv2-based hybrid xG model with built-in attention extraction.

    Parameters
    ----------
    node_in  : number of node feature dimensions
    edge_dim : edge feature dimensions (0 = no edge features)
    meta_dim : shot metadata dimensions (default 12)
    hidden   : hidden channels per attention head
    heads    : number of attention heads in all layers
    n_layers : number of GATv2Conv message-passing layers (≥ 2)
    dropout  : dropout rate applied between layers
    """

    def __init__(
        self,
        node_in:  int,
        edge_dim: int   = 0,
        meta_dim: int   = META_DIM,
        hidden:   int   = 32,
        heads:    int   = 4,
        n_layers: int   = 3,
        dropout:  float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout  = dropout
        self.n_layers = n_layers
        edge_dim_arg  = edge_dim if edge_dim > 0 else None

        self.convs = nn.ModuleList()

        # ── Layer 0: node_in → hidden*heads  (concat) ──────────────────────
        self.convs.append(
            GATv2Conv(
                node_in, hidden, heads=heads,
                edge_dim=edge_dim_arg, dropout=dropout, concat=True,
            )
        )

        # ── Hidden layers: hidden*heads → hidden*heads  (concat) ───────────
        for _ in range(n_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden * heads, hidden, heads=heads,
                    edge_dim=edge_dim_arg, dropout=dropout, concat=True,
                )
            )

        # ── Final layer: hidden*heads → hidden  (1 head, no concat) ────────
        if n_layers > 1:
            self.convs.append(
                GATv2Conv(
                    hidden * heads, hidden, heads=1,
                    edge_dim=edge_dim_arg, dropout=dropout, concat=False,
                )
            )
            pool_dim = hidden
        else:
            pool_dim = hidden * heads

        # ── MLP head: (pool_dim + meta_dim) → 1 ────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(pool_dim + meta_dim, pool_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pool_dim, 1),
        )

        self._pool_dim = pool_dim

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
        metadata:   torch.Tensor,
        edge_attr:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard forward pass. Returns (B, 1) logits."""
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        emb = global_mean_pool(x, batch)                     # (B, pool_dim)
        return self.head(torch.cat([emb, metadata], dim=1))  # (B, 1)

    def forward_with_attention(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor,
        metadata:   torch.Tensor,
        edge_attr:  torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass that also returns per-layer attention weights.

        Returns
        -------
        logit  : (B, 1)  raw logit (before sigmoid)
        alphas : list of (E, heads) tensors — one tensor per GATv2Conv layer

        To produce a single (E,) edge-importance vector for visualisation::

            alpha_stack = torch.stack(alphas)       # (n_layers, E, heads)
            edge_alpha  = alpha_stack.mean(0).mean(1)   # (E,) — mean over layers & heads
        """
        alphas = []
        for conv in self.convs:
            x, (_, alpha) = conv(
                x, edge_index, edge_attr=edge_attr,
                return_attention_weights=True,
            )
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            alphas.append(alpha)        # (E, heads)

        emb   = global_mean_pool(x, batch)
        logit = self.head(torch.cat([emb, metadata], dim=1))
        return logit, alphas
