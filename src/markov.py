"""
markov.py
---------
Markov chain-based Expected Threat (xT) computation.

Divides the pitch into a grid (default 12x8 = 96 zones) and computes:
  - Transition matrix T[i][j]: P(move from zone i to zone j)
  - Shot probability s[i]: P(shot taken from zone i)
  - Goal probability g[i]: P(goal | shot from zone i)
  - xT[i]: expected threat value per zone, solved iteratively

Reference: Karun Singh (2018) "Introducing Expected Threat (xT)"
"""
from __future__ import annotations

import numpy as np

PITCH_LENGTH = 120  # StatsBomb coordinate system
PITCH_WIDTH  = 80
DEFAULT_COLS = 12
DEFAULT_ROWS = 8


def loc_to_zone(x: float, y: float,
                n_cols: int = DEFAULT_COLS,
                n_rows: int = DEFAULT_ROWS) -> tuple[int, int]:
    """Convert StatsBomb (x, y) coordinates to grid zone (col, row)."""
    col = min(int(x / PITCH_LENGTH * n_cols), n_cols - 1)
    row = min(int(y / PITCH_WIDTH  * n_rows), n_rows - 1)
    return col, row


def zone_to_idx(col: int, row: int, n_cols: int = DEFAULT_COLS) -> int:
    """Flatten (col, row) to single index."""
    return row * n_cols + col


def idx_to_zone(idx: int, n_cols: int = DEFAULT_COLS) -> tuple[int, int]:
    """Convert flat index back to (col, row)."""
    return idx % n_cols, idx // n_cols


def compute_xt(
    move_counts: np.ndarray,
    shot_counts: np.ndarray,
    goal_counts: np.ndarray,
    total_actions: np.ndarray,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Solve xT iteratively from zone-level statistics.

    Parameters
    ----------
    move_counts : (n_zones, n_zones) — count of moves from zone i to zone j
    shot_counts : (n_zones,) — count of shots from each zone
    goal_counts : (n_zones,) — count of goals from each zone
    total_actions : (n_zones,) — total actions from each zone (moves + shots)

    Returns
    -------
    xt : (n_zones,) — expected threat per zone
    """
    n_zones = len(total_actions)

    # Avoid division by zero
    safe_total = np.maximum(total_actions, 1)

    # s[i] = P(shot from zone i)
    s = shot_counts / safe_total

    # g[i] = P(goal | shot from zone i)
    safe_shots = np.maximum(shot_counts, 1)
    g = goal_counts / safe_shots
    g[shot_counts == 0] = 0.0

    # T[i][j] = P(move to zone j | action from zone i, not a shot)
    move_total = np.maximum(total_actions - shot_counts, 1)
    T = move_counts / move_total[:, None]

    # Iterative solution: xT[i] = s[i]*g[i] + (1-s[i]) * sum_j T[i][j]*xT[j]
    xt = np.zeros(n_zones)
    for _ in range(n_iter):
        xt_new = s * g + (1 - s) * (T @ xt)
        if np.allclose(xt, xt_new, atol=1e-8):
            break
        xt = xt_new

    return xt
