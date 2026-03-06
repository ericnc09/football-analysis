"""
graph_builder.py
----------------
Converts football event/tracking data into PyTorch Geometric Data objects.

Two main entry points:
  - build_graph_from_freeze_frame()  StatsBomb 360 data → graph per event
  - build_graph_from_tracking_frame() Metrica/TRACAB frame → dynamic graph

Graph structure
---------------
Nodes  : players visible on the pitch at a given moment
         features: [x, y, team, role]  (extended to [x, y, vx, vy, ...] when
         velocity is available from tracking data)

Edges  : constructed by one of three strategies (configurable):
         - "proximity"   connect players within `radius` meters
         - "delaunay"    Delaunay triangulation of player positions
         - "knn"         k-nearest neighbours per player

Edge features: [distance, delta_x, delta_y, same_team]
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0  # metres
PITCH_WIDTH = 68.0    # metres

# StatsBomb pitch dimensions (used to normalise raw coordinates)
SB_PITCH_LENGTH = 120.0
SB_PITCH_WIDTH = 80.0


class EdgeStrategy(str, Enum):
    PROXIMITY = "proximity"
    DELAUNAY = "delaunay"
    KNN = "knn"


# ---------------------------------------------------------------------------
# Coordinate normalisation helpers
# ---------------------------------------------------------------------------

def normalise_statsbomb(x: float, y: float) -> tuple[float, float]:
    """Scale StatsBomb (120×80) coordinates to metres (105×68)."""
    return (x / SB_PITCH_LENGTH) * PITCH_LENGTH, (y / SB_PITCH_WIDTH) * PITCH_WIDTH


def normalise_metrica(x: float, y: float) -> tuple[float, float]:
    """Metrica uses [0,1]×[0,1]; scale to metres."""
    return x * PITCH_LENGTH, y * PITCH_WIDTH


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def _edges_proximity(positions: np.ndarray, radius: float) -> np.ndarray:
    """Return edge index array for all pairs within `radius` metres."""
    n = len(positions)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= radius:
                src.append(i)
                dst.append(j)
    return np.array([src, dst], dtype=np.int64)


def _edges_delaunay(positions: np.ndarray) -> np.ndarray:
    """Return undirected edges from Delaunay triangulation of positions."""
    if len(positions) < 3:
        return np.zeros((2, 0), dtype=np.int64)
    tri = Delaunay(positions)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    # make undirected (both directions)
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    return np.array([src, dst], dtype=np.int64)


def _edges_knn(positions: np.ndarray, k: int) -> np.ndarray:
    """Return directed edges to the k nearest neighbours of each node."""
    n = len(positions)
    k = min(k, n - 1)
    src, dst = [], []
    for i in range(n):
        dists = [
            (np.linalg.norm(positions[i] - positions[j]), j)
            for j in range(n) if j != i
        ]
        dists.sort()
        for _, j in dists[:k]:
            src.append(i)
            dst.append(j)
    return np.array([src, dst], dtype=np.int64)


def _compute_edge_features(positions: np.ndarray, teams: np.ndarray,
                            edge_index: np.ndarray) -> np.ndarray:
    """
    For each edge (src → dst) compute:
      [distance, delta_x, delta_y, same_team]
    Returns float32 array of shape (num_edges, 4).
    """
    if edge_index.shape[1] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    diff = positions[dst_idx] - positions[src_idx]          # (E, 2)
    dist = np.linalg.norm(diff, axis=1, keepdims=True)      # (E, 1)
    same = (teams[src_idx] == teams[dst_idx]).astype(np.float32).reshape(-1, 1)
    return np.hstack([dist, diff, same]).astype(np.float32)


# ---------------------------------------------------------------------------
# Node feature builders
# ---------------------------------------------------------------------------

def _make_node_features(positions: np.ndarray, teams: np.ndarray,
                        velocities: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build node feature matrix.
    Without velocities : [x, y, team]          → shape (N, 3)
    With velocities    : [x, y, vx, vy, team]  → shape (N, 5)
    """
    if velocities is not None:
        return np.hstack([positions, velocities,
                          teams.reshape(-1, 1)]).astype(np.float32)
    return np.hstack([positions,
                      teams.reshape(-1, 1)]).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph_from_freeze_frame(
    freeze_frame: list[dict],
    strategy: EdgeStrategy = EdgeStrategy.DELAUNAY,
    radius: float = 10.0,
    k: int = 5,
    label: Optional[float] = None,
) -> Data:
    """
    Build a PyG Data object from a StatsBomb 360 freeze-frame.

    Parameters
    ----------
    freeze_frame : list of dicts with keys:
        "location"   : [x, y]  (StatsBomb 120×80 coords)
        "teammate"   : bool
        "actor"      : bool     (True for the player performing the event)
        "keeper"     : bool
    strategy      : edge construction method
    radius        : proximity threshold in metres (used when strategy=PROXIMITY)
    k             : neighbours per node (used when strategy=KNN)
    label         : optional graph-level target (e.g. 1.0 = shot follows)

    Returns
    -------
    torch_geometric.data.Data
    """
    positions, teams = [], []
    for player in freeze_frame:
        x, y = normalise_statsbomb(*player["location"])
        positions.append([x, y])
        teams.append(0 if player.get("teammate", True) else 1)

    positions = np.array(positions, dtype=np.float32)
    teams = np.array(teams, dtype=np.float32)

    x = torch.tensor(_make_node_features(positions, teams), dtype=torch.float)

    edge_index = _build_edges(positions, strategy, radius, k)
    edge_attr = torch.tensor(
        _compute_edge_features(positions, teams, edge_index), dtype=torch.float
    )
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data


def build_graph_from_tracking_frame(
    home_positions: np.ndarray,
    away_positions: np.ndarray,
    home_velocities: Optional[np.ndarray] = None,
    away_velocities: Optional[np.ndarray] = None,
    coord_system: str = "metrica",
    strategy: EdgeStrategy = EdgeStrategy.DELAUNAY,
    radius: float = 10.0,
    k: int = 5,
    label: Optional[float] = None,
) -> Data:
    """
    Build a PyG Data object from a single tracking frame.

    Parameters
    ----------
    home_positions  : (N_home, 2) array of x,y coordinates
    away_positions  : (N_away, 2) array of x,y coordinates
    home_velocities : optional (N_home, 2) array of vx,vy
    away_velocities : optional (N_away, 2) array of vx,vy
    coord_system    : "metrica" ([0,1]²) or "tracab" (already in metres)
    strategy        : edge construction method
    radius          : proximity threshold in metres
    k               : neighbours per node (KNN only)
    label           : optional graph-level target

    Returns
    -------
    torch_geometric.data.Data
    """
    normalise = normalise_metrica if coord_system == "metrica" else (lambda x, y: (x, y))

    home_pos = np.array([normalise(p[0], p[1]) for p in home_positions], dtype=np.float32)
    away_pos = np.array([normalise(p[0], p[1]) for p in away_positions], dtype=np.float32)
    positions = np.vstack([home_pos, away_pos])

    n_home = len(home_pos)
    n_away = len(away_pos)
    teams = np.array([0] * n_home + [1] * n_away, dtype=np.float32)

    velocities = None
    if home_velocities is not None and away_velocities is not None:
        velocities = np.vstack([
            np.array(home_velocities, dtype=np.float32),
            np.array(away_velocities, dtype=np.float32),
        ])

    x = torch.tensor(_make_node_features(positions, teams, velocities), dtype=torch.float)

    edge_index = _build_edges(positions, strategy, radius, k)
    edge_attr = torch.tensor(
        _compute_edge_features(positions, teams, edge_index), dtype=torch.float
    )
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

def _build_edges(positions: np.ndarray, strategy: EdgeStrategy,
                 radius: float, k: int) -> np.ndarray:
    if strategy == EdgeStrategy.PROXIMITY:
        return _edges_proximity(positions, radius)
    if strategy == EdgeStrategy.DELAUNAY:
        return _edges_delaunay(positions)
    if strategy == EdgeStrategy.KNN:
        return _edges_knn(positions, k)
    raise ValueError(f"Unknown edge strategy: {strategy}")
