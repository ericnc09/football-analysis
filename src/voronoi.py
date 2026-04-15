"""
voronoi.py
----------
Voronoi tessellation and pitch control computation from player positions.

Given a set of player positions and their team assignments, computes:
  - Voronoi tessellation of the pitch
  - Pitch control: fraction of pitch area controlled by each team
  - Per-cell control map on a regular grid
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi

# Pitch dimensions in metres (normalised StatsBomb coordinates)
PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

# Grid resolution for pitch control map
GRID_RES = 50  # 50x34 grid points


def compute_pitch_control(
    positions: np.ndarray,
    teams: np.ndarray,
    grid_nx: int = GRID_RES,
    grid_ny: int = int(GRID_RES * PITCH_WIDTH / PITCH_LENGTH),
) -> dict:
    """
    Compute nearest-player pitch control on a regular grid.

    Parameters
    ----------
    positions : (n_players, 2) — player (x, y) in metres
    teams : (n_players,) — 0 = shooting team, 1 = defending team

    Returns
    -------
    dict with:
      control_map: (grid_ny, grid_nx) — 0=shooting team, 1=defending
      shooting_pct: float — fraction controlled by shooting team
      defending_pct: float — fraction controlled by defending team
      voronoi: scipy.spatial.Voronoi object (or None if < 4 players)
    """
    xs = np.linspace(0, PITCH_LENGTH, grid_nx)
    ys = np.linspace(0, PITCH_WIDTH,  grid_ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Nearest player for each grid point
    dists = np.linalg.norm(
        grid_points[:, None, :] - positions[None, :, :], axis=2
    )
    nearest = np.argmin(dists, axis=1)
    control = teams[nearest].reshape(grid_ny, grid_nx)

    shooting_pct = float((control == 0).mean())
    defending_pct = float((control == 1).mean())

    # Voronoi object for visualization
    vor = None
    if len(positions) >= 4:
        try:
            vor = Voronoi(positions)
        except Exception:
            pass

    return {
        "control_map":   control,
        "shooting_pct":  shooting_pct,
        "defending_pct": defending_pct,
        "voronoi":       vor,
    }
