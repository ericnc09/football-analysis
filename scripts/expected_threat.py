#!/usr/bin/env python3
"""
expected_threat.py
------------------
Expected Threat (xT) model via Markov chains on StatsBomb event data.

Divides the pitch into a 12x8 grid and computes zone-level threat values
from possession-chain transitions. Values every pass, carry, and dribble
by the change in xT (ΔxT = xT[end_zone] - xT[start_zone]).

Outputs
-------
  data/processed/xt_results.json
  assets/fig_expected_threat.png

Usage
-----
  python scripts/expected_threat.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from statsbombpy import sb

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.markov import (
    loc_to_zone, zone_to_idx, idx_to_zone, compute_xt,
    DEFAULT_COLS, DEFAULT_ROWS, PITCH_LENGTH, PITCH_WIDTH,
)

PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

N_COLS = DEFAULT_COLS
N_ROWS = DEFAULT_ROWS
N_ZONES = N_COLS * N_ROWS

COMPETITIONS = [
    {"comp_id": 43, "season_id": 106, "label": "wc2022"},
    {"comp_id": 72, "season_id": 107, "label": "wwc2023"},
    {"comp_id": 55, "season_id": 43,  "label": "euro2020"},
    {"comp_id": 55, "season_id": 282, "label": "euro2024"},
    {"comp_id": 9,  "season_id": 281, "label": "bundesliga2324"},
    {"comp_id": 53, "season_id": 106, "label": "weuro2022"},
    {"comp_id": 53, "season_id": 315, "label": "weuro2025"},
]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def fetch_match_ids() -> list[tuple[int, str]]:
    """Return list of (match_id, comp_label) for all competitions."""
    match_ids = []
    for comp in COMPETITIONS:
        matches = sb.matches(competition_id=comp["comp_id"],
                             season_id=comp["season_id"])
        for mid in matches["match_id"]:
            match_ids.append((int(mid), comp["label"]))
    print(f"  Total matches: {len(match_ids)}")
    return match_ids


def process_events(match_ids: list[tuple[int, str]]) -> dict:
    """
    Process all events and build zone-level statistics for xT.

    Returns dict with:
      move_counts: (N_ZONES, N_ZONES) — transitions between zones
      shot_counts: (N_ZONES,) — shots from each zone
      goal_counts: (N_ZONES,) — goals from each zone
      total_actions: (N_ZONES,) — total actions from each zone
      player_xt: {player: total ΔxT} (filled after xT computation)
      all_actions: list of action dicts for per-event analysis
    """
    move_counts   = np.zeros((N_ZONES, N_ZONES), dtype=np.float64)
    shot_counts   = np.zeros(N_ZONES, dtype=np.float64)
    goal_counts   = np.zeros(N_ZONES, dtype=np.float64)
    total_actions = np.zeros(N_ZONES, dtype=np.float64)
    all_actions   = []

    for i, (mid, comp) in enumerate(match_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing match {i+1}/{len(match_ids)} …")

        try:
            events = sb.events(match_id=mid)
        except Exception as e:
            print(f"    Skipping match {mid}: {e}")
            continue

        # Filter to possession actions: Pass, Carry, Dribble, Shot
        for _, ev in events.iterrows():
            etype = ev.get("type", "")
            loc   = ev.get("location")
            if not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            x_start, y_start = float(loc[0]), float(loc[1])
            col_s, row_s = loc_to_zone(x_start, y_start, N_COLS, N_ROWS)
            idx_s = zone_to_idx(col_s, row_s, N_COLS)

            player = ev.get("player", "Unknown")
            team   = ev.get("possession_team", ev.get("team", "Unknown"))

            if etype == "Shot":
                shot_counts[idx_s]   += 1
                total_actions[idx_s] += 1
                is_goal = (ev.get("shot_outcome", "") == "Goal")
                if is_goal:
                    goal_counts[idx_s] += 1
                all_actions.append({
                    "type": "shot", "player": player, "team": team,
                    "comp": comp, "match_id": mid,
                    "start_zone": idx_s, "end_zone": idx_s,
                    "is_goal": is_goal,
                })

            elif etype == "Pass":
                end_loc = ev.get("pass_end_location")
                if not isinstance(end_loc, (list, tuple)) or len(end_loc) < 2:
                    continue
                x_end, y_end = float(end_loc[0]), float(end_loc[1])
                col_e, row_e = loc_to_zone(x_end, y_end, N_COLS, N_ROWS)
                idx_e = zone_to_idx(col_e, row_e, N_COLS)
                move_counts[idx_s, idx_e] += 1
                total_actions[idx_s]      += 1
                all_actions.append({
                    "type": "pass", "player": player, "team": team,
                    "comp": comp, "match_id": mid,
                    "start_zone": idx_s, "end_zone": idx_e,
                })

            elif etype == "Carry":
                end_loc = ev.get("carry_end_location")
                if not isinstance(end_loc, (list, tuple)) or len(end_loc) < 2:
                    continue
                x_end, y_end = float(end_loc[0]), float(end_loc[1])
                col_e, row_e = loc_to_zone(x_end, y_end, N_COLS, N_ROWS)
                idx_e = zone_to_idx(col_e, row_e, N_COLS)
                move_counts[idx_s, idx_e] += 1
                total_actions[idx_s]      += 1
                all_actions.append({
                    "type": "carry", "player": player, "team": team,
                    "comp": comp, "match_id": mid,
                    "start_zone": idx_s, "end_zone": idx_e,
                })

            elif etype == "Dribble":
                # Dribbles don't have end_location in StatsBomb;
                # treat as staying in same zone (no move)
                total_actions[idx_s] += 1

    return {
        "move_counts":   move_counts,
        "shot_counts":   shot_counts,
        "goal_counts":   goal_counts,
        "total_actions": total_actions,
        "all_actions":   all_actions,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(xt_grid, top_players, top_passers, zone_stats):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Expected Threat (xT) — Markov Chain Model",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: xT heatmap on pitch ─────────────────────────────────────
    ax = axes[0, 0]
    pitch = Pitch(pitch_type="statsbomb", line_color="white",
                  pitch_color="#1a1a2e")
    pitch.draw(ax=ax)
    # Draw xT grid as rectangles
    cell_w = PITCH_LENGTH / N_COLS
    cell_h = PITCH_WIDTH  / N_ROWS
    max_xt = xt_grid.max()
    for col in range(N_COLS):
        for row in range(N_ROWS):
            val = xt_grid[row, col]
            alpha = val / max_xt if max_xt > 0 else 0
            rect = plt.Rectangle(
                (col * cell_w, row * cell_h), cell_w, cell_h,
                fc=plt.cm.YlOrRd(alpha), ec="white", lw=0.3, alpha=0.85,
            )
            ax.add_patch(rect)
            if val > max_xt * 0.3:
                ax.text(col * cell_w + cell_w/2, row * cell_h + cell_h/2,
                        f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color="white" if alpha > 0.5 else "black")
    ax.set_title("Expected Threat (xT) Heatmap", color="white")

    # ── Panel 2: Top 15 players by total xT generated ────────────────────
    ax = axes[0, 1]
    names = [p[0] for p in top_players[:15]][::-1]
    vals  = [p[1] for p in top_players[:15]][::-1]
    colors = ["steelblue" if v >= 0 else "salmon" for v in vals]
    ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Total xT Generated (ΣΔxT)")
    ax.set_title("Top 15 Players by xT Generated")
    ax.axvline(0, color="black", ls=":", lw=0.5)

    # ── Panel 3: Top 15 passers by xT generated ──────────────────────────
    ax = axes[1, 0]
    names = [p[0] for p in top_passers[:15]][::-1]
    vals  = [p[1] for p in top_passers[:15]][::-1]
    ax.barh(range(len(names)), vals, color="teal", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Pass xT Generated (ΣΔxT from passes)")
    ax.set_title("Top 15 Passers by xT")

    # ── Panel 4: Zone activity (shots + goals overlay) ────────────────────
    ax = axes[1, 1]
    shot_grid = zone_stats["shot_grid"]
    goal_grid = zone_stats["goal_grid"]
    ax.imshow(shot_grid, origin="lower", cmap="Blues", aspect="auto", alpha=0.7)
    # Overlay goals as red dots
    for row in range(N_ROWS):
        for col in range(N_COLS):
            if goal_grid[row, col] > 0:
                ax.plot(col, row, "ro", ms=max(3, goal_grid[row, col] * 0.3),
                        alpha=0.7)
    ax.set_xlabel("Column (attacking direction →)")
    ax.set_ylabel("Row")
    ax.set_title("Shot Distribution by Zone (red = goals)")
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print(f"  Expected Threat (xT) — Markov Chain Model")
    print(f"  Grid: {N_COLS}x{N_ROWS} = {N_ZONES} zones")
    print("=" * 64)

    # ── Fetch match IDs ───────────────────────────────────────────────────
    print("\n── Fetching match list ─────────────────────────────────────")
    match_ids = fetch_match_ids()

    # ── Process events ────────────────────────────────────────────────────
    print("\n── Processing events ───────────────────────────────────────")
    stats = process_events(match_ids)

    move_counts   = stats["move_counts"]
    shot_counts   = stats["shot_counts"]
    goal_counts   = stats["goal_counts"]
    total_actions = stats["total_actions"]
    all_actions   = stats["all_actions"]

    print(f"\n  Actions processed: {int(total_actions.sum()):,}")
    print(f"  Total shots: {int(shot_counts.sum()):,}")
    print(f"  Total goals: {int(goal_counts.sum()):,}")
    print(f"  Total moves (pass+carry): {int(move_counts.sum()):,}")

    # ── Compute xT ────────────────────────────────────────────────────────
    print("\n── Computing xT (iterative Markov solution) …")
    xt = compute_xt(move_counts, shot_counts, goal_counts, total_actions)

    xt_grid = xt.reshape(N_ROWS, N_COLS)
    print(f"\n  xT range: [{xt.min():.4f}, {xt.max():.4f}]")
    print(f"  Max xT zone: col={xt.argmax() % N_COLS}, row={xt.argmax() // N_COLS} "
          f"(value={xt.max():.4f})")

    # Print xT grid
    print("\n  xT grid (rows=pitch width, cols=attacking direction →):")
    for row in range(N_ROWS - 1, -1, -1):
        vals = " ".join(f"{xt_grid[row, c]:.3f}" for c in range(N_COLS))
        print(f"    [{vals}]")

    # ── Per-action ΔxT ────────────────────────────────────────────────────
    print("\n── Computing per-action ΔxT …")
    player_xt_total = defaultdict(float)
    player_xt_pass  = defaultdict(float)
    player_actions  = defaultdict(int)
    team_xt         = defaultdict(float)

    for act in all_actions:
        if act["type"] == "shot":
            continue
        delta = xt[act["end_zone"]] - xt[act["start_zone"]]
        player_xt_total[act["player"]] += delta
        player_actions[act["player"]] += 1
        team_xt[act["team"]] += delta
        if act["type"] == "pass":
            player_xt_pass[act["player"]] += delta

    top_players = sorted(player_xt_total.items(), key=lambda x: -x[1])
    top_passers = sorted(player_xt_pass.items(),  key=lambda x: -x[1])
    top_teams   = sorted(team_xt.items(),         key=lambda x: -x[1])

    print("\n  Top 10 players by total xT generated:")
    for i, (p, v) in enumerate(top_players[:10], 1):
        n = player_actions[p]
        print(f"    {i:>2}. {p:<30} ΣΔxT={v:+.3f}  ({n} actions)")

    print("\n  Top 10 passers by pass xT:")
    for i, (p, v) in enumerate(top_passers[:10], 1):
        print(f"    {i:>2}. {p:<30} ΣΔxT={v:+.3f}")

    print("\n  Top 10 teams by total xT:")
    for i, (t, v) in enumerate(top_teams[:10], 1):
        print(f"    {i:>2}. {t:<30} ΣΔxT={v:+.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    shot_grid = shot_counts.reshape(N_ROWS, N_COLS)
    goal_grid = goal_counts.reshape(N_ROWS, N_COLS)

    results = {
        "description": "Expected Threat (xT) via Markov chains on StatsBomb events",
        "grid": {"cols": N_COLS, "rows": N_ROWS, "n_zones": N_ZONES},
        "total_actions": int(total_actions.sum()),
        "total_shots": int(shot_counts.sum()),
        "total_goals": int(goal_counts.sum()),
        "xt_grid": xt_grid.tolist(),
        "xt_flat": xt.tolist(),
        "top_players_xt": [
            {"player": p, "total_xt": round(v, 4), "actions": player_actions[p]}
            for p, v in top_players[:50]
        ],
        "top_passers_xt": [
            {"player": p, "pass_xt": round(v, 4)}
            for p, v in top_passers[:50]
        ],
        "top_teams_xt": [
            {"team": t, "total_xt": round(v, 4)}
            for t, v in top_teams[:30]
        ],
    }
    out_path = PROCESSED / "xt_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Figure ────────────────────────────────────────────────────────────
    print("\n── Generating figure …")
    zone_stats = {"shot_grid": shot_grid, "goal_grid": goal_grid}
    fig = make_figure(xt_grid, top_players, top_passers, zone_stats)
    fig_path = ASSETS / "fig_expected_threat.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
