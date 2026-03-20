"""
features.py
-----------
Enriches PyG Data objects with additional node and edge features.

Node features added on top of graph_builder output:
  - distance to each goal (attacking, defending)
  - angle to attacking goal
  - pressure index  (# opponents within `pressure_radius` metres)
  - role encoding   (if role labels are provided)

Edge features added on top of graph_builder output:
  - pass angle (direction of edge relative to goal)
  - velocity alignment (dot product of velocities if tracking data)

Main entry point: enrich_graph()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

# Goal centre positions in metres (105×68 pitch)
GOAL_LEFT  = np.array([0.0,   34.0], dtype=np.float32)
GOAL_RIGHT = np.array([105.0, 34.0], dtype=np.float32)

# StatsBomb role strings → integer index (0 = unknown)
ROLE_INDEX: dict[str, int] = {
    "goalkeeper":         1,
    "center back":        2,
    "left back":          3,
    "right back":         4,
    "left wing back":     5,
    "right wing back":    6,
    "defensive midfield": 7,
    "center midfield":    8,
    "left midfield":      9,
    "right midfield":     10,
    "attacking midfield": 11,
    "left wing":          12,
    "right wing":         13,
    "center forward":     14,
    "second striker":     15,
}
NUM_ROLES = len(ROLE_INDEX) + 1  # +1 for unknown

# StatsBomb shot_technique strings → integer index (0 = unknown)
TECHNIQUE_INDEX: dict[str, int] = {
    "normal":         1,
    "volley":         2,
    "half volley":    3,
    "lob":            4,
    "backheel":       5,
    "overhead kick":  6,
    "diving header":  7,
}
NUM_TECHNIQUES = len(TECHNIQUE_INDEX) + 1  # 8 (0 = unknown)


def encode_technique(technique: str) -> np.ndarray:
    """One-hot encode a shot_technique string → (NUM_TECHNIQUES,) float32 array."""
    enc = np.zeros(NUM_TECHNIQUES, dtype=np.float32)
    idx = TECHNIQUE_INDEX.get((technique or "").lower().strip(), 0)
    enc[idx] = 1.0
    return enc


# ---------------------------------------------------------------------------
# Individual feature computations
# ---------------------------------------------------------------------------

def _dist_and_angle_to_goal(positions: np.ndarray,
                             goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (distances, angles) arrays of shape (N,).
    Angle is in radians, measured from positive x-axis.
    """
    diff = goal - positions                                  # (N, 2)
    dist = np.linalg.norm(diff, axis=1)                     # (N,)
    angle = np.arctan2(diff[:, 1], diff[:, 0])              # (N,)
    return dist.astype(np.float32), angle.astype(np.float32)


def _pressure_index(positions: np.ndarray, teams: np.ndarray,
                    radius: float = 5.0) -> np.ndarray:
    """
    For each player, count the number of opponents within `radius` metres.
    Returns (N,) float32 array.
    """
    n = len(positions)
    counts = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j or teams[i] == teams[j]:
                continue
            if np.linalg.norm(positions[i] - positions[j]) <= radius:
                counts[i] += 1.0
    return counts


def _encode_roles(roles: list[str]) -> np.ndarray:
    """
    One-hot encode a list of role strings.
    Returns (N, NUM_ROLES) float32 array.
    """
    n = len(roles)
    enc = np.zeros((n, NUM_ROLES), dtype=np.float32)
    for i, role in enumerate(roles):
        idx = ROLE_INDEX.get(role.lower().strip(), 0)
        enc[i, idx] = 1.0
    return enc


def _edge_pass_angle(positions: np.ndarray, edge_index: np.ndarray,
                     attacking_goal: np.ndarray) -> np.ndarray:
    """
    For each directed edge (src→dst), compute the cosine similarity between
    the edge direction and the direction toward the attacking goal from src.
    Returns (E,) float32 array in [-1, 1].
    """
    if edge_index.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)

    src, dst = edge_index[0], edge_index[1]
    edge_vec = positions[dst] - positions[src]                # (E, 2)
    goal_vec = attacking_goal - positions[src]                # (E, 2)

    edge_norm = np.linalg.norm(edge_vec, axis=1, keepdims=True).clip(min=1e-8)
    goal_norm = np.linalg.norm(goal_vec, axis=1, keepdims=True).clip(min=1e-8)

    cosine = np.sum((edge_vec / edge_norm) * (goal_vec / goal_norm), axis=1)
    return cosine.astype(np.float32)


def _edge_velocity_alignment(velocities: np.ndarray,
                              edge_index: np.ndarray) -> np.ndarray:
    """
    Dot product of normalised velocities of src and dst nodes.
    Returns (E,) float32 array in [-1, 1].
    """
    if edge_index.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)

    src, dst = edge_index[0], edge_index[1]
    v_src = velocities[src]
    v_dst = velocities[dst]

    norm_src = np.linalg.norm(v_src, axis=1, keepdims=True).clip(min=1e-8)
    norm_dst = np.linalg.norm(v_dst, axis=1, keepdims=True).clip(min=1e-8)

    alignment = np.sum((v_src / norm_src) * (v_dst / norm_dst), axis=1)
    return alignment.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_graph(
    data: Data,
    attacking_right: bool = True,
    pressure_radius: float = 5.0,
    roles: Optional[list[str]] = None,
) -> Data:
    """
    Add richer node and edge features to an existing PyG Data object.

    Expects `data.x` columns to be  [x, y, ...]  (first two are coordinates)
    and optionally [x, y, vx, vy, ...]  (first four are position + velocity).

    Parameters
    ----------
    data            : PyG Data object from graph_builder
    attacking_right : if True, home team attacks toward x=105 goal
    pressure_radius : radius in metres for pressure index
    roles           : optional list of role strings, one per node

    Returns
    -------
    Updated Data object with enriched x and edge_attr.
    """
    x_np = data.x.numpy()
    edge_index_np = data.edge_index.numpy()
    positions = x_np[:, :2]
    teams = x_np[:, -1]  # last column is always team flag from graph_builder

    attacking_goal = GOAL_RIGHT if attacking_right else GOAL_LEFT
    defending_goal = GOAL_LEFT  if attacking_right else GOAL_RIGHT

    # --- node features ---
    dist_atk, angle_atk = _dist_and_angle_to_goal(positions, attacking_goal)
    dist_def, _         = _dist_and_angle_to_goal(positions, defending_goal)
    pressure            = _pressure_index(positions, teams, pressure_radius)

    new_node_feats = [
        x_np,
        dist_atk.reshape(-1, 1),
        dist_def.reshape(-1, 1),
        angle_atk.reshape(-1, 1),
        pressure.reshape(-1, 1),
    ]

    if roles is not None:
        role_enc = _encode_roles(roles)
        new_node_feats.append(role_enc)

    data.x = torch.tensor(np.hstack(new_node_feats), dtype=torch.float)

    # --- edge features ---
    pass_angle = _edge_pass_angle(positions, edge_index_np, attacking_goal)
    new_edge_feats = [data.edge_attr.numpy(), pass_angle.reshape(-1, 1)]

    # velocity alignment (only if vx,vy present — columns 2,3)
    if x_np.shape[1] >= 5:  # [x, y, vx, vy, team, ...]
        velocities = x_np[:, 2:4]
        vel_align = _edge_velocity_alignment(velocities, edge_index_np)
        new_edge_feats.append(vel_align.reshape(-1, 1))

    data.edge_attr = torch.tensor(np.hstack(new_edge_feats), dtype=torch.float)

    return data
