"""
features.py
-----------
Enriches PyG Data objects with additional node and edge features, AND owns
the single canonical implementation of `build_meta()` — the shot-metadata
tensor consumed by the hybrid model's MLP head.

Why `build_meta()` lives here
-----------------------------
Historically, two near-identical implementations coexisted:
  - `app.py::_build_meta`         (serving)
  - `src/calibration.py::_default_meta` (calibration)
  - (implicit third copy via ad-hoc tensors in compute_node_saliency and
    compute_gat_attention inside app.py)

They drifted silently because each imputed plausible defaults (`is_right_foot
= 0.5`, `gk_perp_offset = 3.0`) for missing attributes. The ML review flagged
this as a "silent wrong prediction" bug: a newer `meta_dim=27` model served
against older graphs gave sensible-looking but wrong outputs with no warning.

This module now hosts the canonical `build_meta(batch, schema_version)`. It
validates that every required attribute exists on the batch and **raises** on
missing attributes rather than fabricating a default. The loader (app.py)
calls `assert_graph_schema_compatible()` at startup against every graph file
so failure happens at service-start, not mid-inference.

Node features added on top of graph_builder output:
  - distance to each goal (attacking, defending)
  - angle to attacking goal
  - pressure index  (# opponents within `pressure_radius` metres)
  - role encoding   (if role labels are provided)

Edge features added on top of graph_builder output:
  - pass angle (direction of edge relative to goal)
  - velocity alignment (dot product of velocities if tracking data)

Public API:
  - enrich_graph()                      — add node/edge features to a Data
  - encode_technique() / encode_placement() — one-hot helpers
  - build_meta(batch, schema)           — canonical metadata tensor builder
  - assert_graph_schema_compatible()    — startup-time contract check
  - GraphSchemaMismatch                 — raised on missing attributes
"""

from __future__ import annotations

from typing import Iterable, Optional

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


# StatsBomb shot_placement strings → goal-zone index (0 = unknown / wide / off-target)
# Zones 1-8 map a 2×4 grid of the goal face (top/bottom × left/centre-left/centre-right/right)
# plus special zones for the goalkeeper and post.
#
#   Zone layout (facing the goal):
#     ┌───────┬────────┬────────┬───────┐
#     │  3    │   4    │   4    │   5   │  ← top row
#     │TL/HL  │  TC/H  │  TC/H  │ TR/HR │
#     ├───────┼────────┼────────┼───────┤
#     │  6    │   7    │   7    │   8   │  ← bottom row
#     │BL/LL  │BC/LC/L │BC/LC/L │BR/LR/R│
#     └───────┴────────┴────────┴───────┘
#   Zone 1 = Goalkeeper / Saved (central keeper area)
#   Zone 2 = Post / Bar
#   Zone 0 = Unknown / Wide / Blocked / Missed / No Touch
PLACEMENT_INDEX: dict[str, int] = {
    # Zone 1 — goalkeeper / central save
    "goalkeeper":           1,
    "saved":                1,
    "saved off target":     1,
    "saved to post":        1,
    # Zone 2 — post / bar
    "post":                 2,
    # Zone 3 — top left
    "top left corner":      3,
    "high left":            3,
    # Zone 4 — top centre
    "top centre":           4,
    "high centre":          4,
    "high":                 4,
    # Zone 5 — top right
    "top right corner":     5,
    "high right":           5,
    # Zone 6 — bottom left
    "bottom left corner":   6,
    "low left":             6,
    "left":                 6,
    # Zone 7 — bottom centre
    "bottom centre":        7,
    "low centre":           7,
    "low":                  7,
    # Zone 8 — bottom right
    "bottom right corner":  8,
    "low right":            8,
    "right":                8,
    # Zone 0 (default) — unknown, wide, blocked, missed, no touch
}
NUM_PLACEMENTS = 9  # 0 = unknown/wide, 1-8 = goal zones


def encode_placement(placement: str) -> np.ndarray:
    """One-hot encode a shot_placement string → (NUM_PLACEMENTS,) float32 array.

    This is a post-shot feature (PSxG-style): it describes WHERE on the goal
    face the ball ended up, enabling the model to distinguish top-corner strikes
    from central saves. Zone 0 covers wide / off-target / unknown shots.
    """
    enc = np.zeros(NUM_PLACEMENTS, dtype=np.float32)
    idx = PLACEMENT_INDEX.get((placement or "").lower().strip(), 0)
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


# ---------------------------------------------------------------------------
# Canonical shot-metadata tensor builder
# ---------------------------------------------------------------------------
#
# `build_meta()` is the single implementation of the metadata tensor that the
# hybrid model's MLP head consumes. Historically three near-duplicate copies
# existed (`app.py::_build_meta`, `src/calibration.py::_default_meta`, and
# ad-hoc tensors in `compute_node_saliency` / `compute_gat_attention`). They
# drifted silently because each imputed plausible defaults for missing
# attributes (`is_right_foot=0.5`, `gk_perp_offset=3.0`, `shot_placement=0`).
#
# The ML review flagged this as a *silent wrong prediction* class of bug: a
# newer `meta_dim=27` model served against older graphs gave sensible-looking
# but **wrong** outputs with no warning.
#
# This implementation inverts that design: missing attributes are a
# **hard error**, not a defaultable field. The caller is responsible for
# ensuring the graphs loaded on disk match the feature_schema_version
# recorded in the model's `.meta.json` sidecar.
# ---------------------------------------------------------------------------

class GraphSchemaMismatch(RuntimeError):
    """
    Raised when a shot graph is missing one or more attributes that the
    requested `feature_schema_version` requires. Fail loudly — the caller
    should rebuild shot graphs with `scripts/build_shot_graphs.py` at a
    matching schema, or retrain against the current graph schema.

    This exception is the graph-side twin of
    `src.model_metadata.FeatureSchemaMismatch`: one gates the model, the
    other gates the inputs fed into it.
    """


# Required PyG `Data` / `Batch` attribute names per schema version.
# Must stay lock-stepped with `KNOWN_SCHEMAS` in `src/model_metadata.py`.
#
# These are the attributes `build_meta()` reads off each graph. A graph
# can have MORE attributes than required (e.g. home_team, sb_xg, match_id)
# without triggering the check — only MISSING ones are a contract break.
REQUIRED_ATTRS_PER_SCHEMA: dict[str, tuple[str, ...]] = {
    # 12-dim: base (4) + technique one-hot (8)
    "v1-base": (
        "shot_dist", "shot_angle", "is_header", "is_open_play",
        "technique",
    ),
    # 18-dim: v1-base + gk_original (3) + gk_precision (3)
    "v2-gk": (
        "shot_dist", "shot_angle", "is_header", "is_open_play",
        "technique",
        "gk_dist", "n_def_in_cone", "gk_off_centre",
        "gk_perp_offset", "n_def_direct_line", "is_right_foot",
    ),
    # 27-dim: v2-gk + shot_placement one-hot (9)
    "v3-psxg": (
        "shot_dist", "shot_angle", "is_header", "is_open_play",
        "technique",
        "gk_dist", "n_def_in_cone", "gk_off_centre",
        "gk_perp_offset", "n_def_direct_line", "is_right_foot",
        "shot_placement",
    ),
}


def _require(batch, attr: str, schema_version: str) -> torch.Tensor:
    """
    Fetch `batch.<attr>` or raise `GraphSchemaMismatch` with a message that
    tells the operator exactly what to do. No silent fallbacks.
    """
    if not hasattr(batch, attr):
        raise GraphSchemaMismatch(
            f"Shot graph is missing required attribute {attr!r} for "
            f"feature_schema_version={schema_version!r}. Either (a) rebuild "
            f"the graphs with scripts/build_shot_graphs.py at a matching "
            f"schema, or (b) serve an older model whose sidecar declares a "
            f"schema that does not require {attr!r}. Do NOT impute defaults — "
            f"silent padding produced the v2→v3 wrong-prediction bug that "
            f"this gate exists to prevent."
        )
    return getattr(batch, attr)


def build_meta(batch, schema_version: str | None = None) -> torch.Tensor:
    """
    Build the canonical metadata tensor for a PyG `Data` / `Batch` of shots.

    Parameters
    ----------
    batch : torch_geometric.data.Data | Batch
        Must carry every attribute listed in
        `REQUIRED_ATTRS_PER_SCHEMA[schema_version]`. Missing attributes
        raise `GraphSchemaMismatch` — there are no defaults.
    schema_version : str, optional
        One of the keys in `REQUIRED_ATTRS_PER_SCHEMA`. Defaults to
        `model_metadata.CURRENT_FEATURE_SCHEMA_VERSION`. Callers that load
        a model should pass `model._model_meta.feature_schema_version` so
        the meta tensor is built for exactly that model's contract.

    Returns
    -------
    torch.Tensor
        Shape `[N, meta_dim]` where `meta_dim` is
        12 for `v1-base`, 18 for `v2-gk`, 27 for `v3-psxg`.

    Raises
    ------
    GraphSchemaMismatch
        On any missing required attribute OR an unknown `schema_version`.
    """
    # Lazy-import to avoid a circular import at module load time —
    # `model_metadata` doesn't depend on this module but we want the
    # default to track CURRENT_FEATURE_SCHEMA_VERSION without forcing
    # every importer of `features` to pay for it.
    if schema_version is None:
        from src.model_metadata import CURRENT_FEATURE_SCHEMA_VERSION
        schema_version = CURRENT_FEATURE_SCHEMA_VERSION

    if schema_version not in REQUIRED_ATTRS_PER_SCHEMA:
        raise GraphSchemaMismatch(
            f"Unknown feature_schema_version={schema_version!r}. "
            f"Known: {list(REQUIRED_ATTRS_PER_SCHEMA.keys())}. Add a new "
            f"entry to REQUIRED_ATTRS_PER_SCHEMA and KNOWN_SCHEMAS if "
            f"this is a new schema; otherwise check for a typo."
        )

    # ── v1-base components (12 dims) ─────────────────────────────────────────
    base = torch.stack([
        _require(batch, "shot_dist",    schema_version).squeeze(),
        _require(batch, "shot_angle",   schema_version).squeeze(),
        _require(batch, "is_header",    schema_version).squeeze().float(),
        _require(batch, "is_open_play", schema_version).squeeze().float(),
    ], dim=1)                                              # [N, 4]
    tech = _require(batch, "technique", schema_version).view(-1, 8)   # [N, 8]

    if schema_version == "v1-base":
        return torch.cat([base, tech], dim=1)              # [N, 12]

    # ── v2-gk extras (+6 → 18 dims) ──────────────────────────────────────────
    gk = torch.stack([
        _require(batch, "gk_dist",        schema_version).squeeze(),
        _require(batch, "n_def_in_cone",  schema_version).squeeze(),
        _require(batch, "gk_off_centre",  schema_version).squeeze(),
    ], dim=1)                                              # [N, 3]
    gk_prec = torch.stack([
        _require(batch, "gk_perp_offset",    schema_version).squeeze(),
        _require(batch, "n_def_direct_line", schema_version).squeeze(),
        _require(batch, "is_right_foot",     schema_version).squeeze(),
    ], dim=1)                                              # [N, 3]

    if schema_version == "v2-gk":
        return torch.cat([base, tech, gk, gk_prec], dim=1)  # [N, 18]

    # ── v3-psxg extras (+9 → 27 dims) ────────────────────────────────────────
    placement = _require(batch, "shot_placement", schema_version).view(-1, 9)

    return torch.cat([base, tech, gk, gk_prec, placement], dim=1)   # [N, 27]


def assert_graph_schema_compatible(
    graphs: Iterable,
    schema_version: str,
    *,
    source_name: str = "<unknown>",
) -> None:
    """
    Walk `graphs` and raise `GraphSchemaMismatch` on the first one that
    doesn't satisfy the attribute contract for `schema_version`.

    Intended for startup-time validation (e.g. right after loading a
    shot-graph file and before running inference), so failures happen
    deterministically at service-start instead of mid-prediction.

    Parameters
    ----------
    graphs : Iterable of PyG `Data` objects
    schema_version : "v1-base" | "v2-gk" | "v3-psxg"
    source_name : human-readable identifier of the graph source
        (filename, competition key, etc.) — appears in error messages so
        the operator knows which artefact is stale.
    """
    if schema_version not in REQUIRED_ATTRS_PER_SCHEMA:
        raise GraphSchemaMismatch(
            f"Unknown feature_schema_version={schema_version!r}. "
            f"Known: {list(REQUIRED_ATTRS_PER_SCHEMA.keys())}."
        )

    required = REQUIRED_ATTRS_PER_SCHEMA[schema_version]
    for idx, g in enumerate(graphs):
        missing = [a for a in required if not hasattr(g, a)]
        if missing:
            raise GraphSchemaMismatch(
                f"Graph source {source_name!r} is not compatible with "
                f"feature_schema_version={schema_version!r}. Graph #{idx} "
                f"is missing attribute(s): {missing}. "
                f"Rebuild with scripts/build_shot_graphs.py at a matching "
                f"schema, or load a model whose sidecar declares an older "
                f"schema."
            )
