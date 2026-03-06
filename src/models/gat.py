"""
gat.py
------
Graph Attention Network for football graph classification.

Key difference from GCN: attention weights are learned per edge, allowing
the model to focus on the most tactically relevant player relationships.
Edge features are incorporated via a concatenation trick before attention.

Architecture:
  GATv2Conv × n_layers  →  global mean pool  →  MLP head  →  scalar output
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class FootballGAT(nn.Module):
    """
    Parameters
    ----------
    in_channels       : node feature dimensions
    edge_dim          : edge feature dimensions (set 0 to ignore edge features)
    hidden_dim        : width per attention head
    out_channels      : output size
    n_layers          : number of GAT message-passing layers
    heads             : number of attention heads (all layers except last)
    dropout           : dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int = 0,
        hidden_dim: int = 32,
        out_channels: int = 1,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()

        edge_dim_arg = edge_dim if edge_dim > 0 else None

        # input layer — multi-head, concat output
        self.convs.append(
            GATv2Conv(in_channels, hidden_dim, heads=heads,
                      edge_dim=edge_dim_arg, dropout=dropout, concat=True)
        )

        # hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads,
                          edge_dim=edge_dim_arg, dropout=dropout, concat=True)
            )

        # final conv layer — single head, no concat
        if n_layers > 1:
            self.convs.append(
                GATv2Conv(hidden_dim * heads, hidden_dim, heads=1,
                          edge_dim=edge_dim_arg, dropout=dropout, concat=False)
            )
            pool_dim = hidden_dim
        else:
            pool_dim = hidden_dim * heads

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, pool_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pool_dim // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor,
                edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, in_channels)
        edge_index : (2, E)
        batch      : (N,)
        edge_attr  : (E, edge_dim)  optional

        Returns
        -------
        (B, out_channels)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.mlp(x)
