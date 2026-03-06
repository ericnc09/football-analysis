"""
gcn.py
------
Graph Convolutional Network baseline for football graph classification.

Architecture:
  GCNConv × n_layers  →  global mean pool  →  MLP head  →  scalar output

Use cases:
  - Possession outcome prediction  (binary classification, label = shot follows)
  - Formation classification        (multi-class)
  - xG contribution per event       (regression)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class FootballGCN(nn.Module):
    """
    Parameters
    ----------
    in_channels  : number of node feature dimensions
    hidden_dim   : width of GCN layers
    out_channels : output size (1 for binary/regression, K for K-class)
    n_layers     : number of GCN message-passing layers (default 3)
    dropout      : dropout rate applied between layers
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        out_channels: int = 1,
        n_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # input layer
        self.convs.append(GCNConv(in_channels, hidden_dim))
        # hidden layers
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # MLP head after pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, in_channels)   node features
        edge_index : (2, E)             COO edge list
        batch      : (N,)               batch vector (graph membership per node)

        Returns
        -------
        (B, out_channels)  one prediction per graph in the batch
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # graph-level representation via mean pooling
        x = global_mean_pool(x, batch)  # (B, hidden_dim)
        return self.mlp(x)              # (B, out_channels)
