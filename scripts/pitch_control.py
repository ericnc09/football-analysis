#!/usr/bin/env python3
"""
pitch_control.py
----------------
Voronoi tessellation and pitch control analysis from shot freeze-frames.

For each shot in the dataset, computes what fraction of the pitch is
controlled by the shooting team vs the defending team at the moment
of the shot, using nearest-player Voronoi tessellation.

Analyses:
  - Pitch control distribution for goals vs misses
  - Correlation between pitch control and xG
  - Per-competition pitch control statistics
  - Example Voronoi visualizations

Outputs
-------
  data/processed/pitch_control_results.json
  assets/fig_pitch_control.png

Usage
-----
  python scripts/pitch_control.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import pearsonr
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.voronoi import compute_pitch_control, PITCH_LENGTH, PITCH_WIDTH

PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graphs() -> list:
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
        print(f"  {pt.name}: {len(gs)}")
    print(f"  Total: {len(graphs)}")
    return graphs


# ---------------------------------------------------------------------------
# Extract positions and teams from graph
# ---------------------------------------------------------------------------

def extract_positions(g) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract player positions and team assignments from a shot graph.

    Node features: [x_m, y_m, teammate(0/1), actor, keeper, ...]
    teammate=1 means same team as the shooter (shooting team).
    """
    x = g.x.numpy()
    positions = x[:, :2]  # (n_players, 2)
    # teammate=1 → shooting team (label 0), teammate=0 → defending (label 1)
    teams = (1 - x[:, 2]).astype(int)
    return positions, teams


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results_list, graphs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Voronoi Pitch Control Analysis — Shot Freeze Frames",
                 fontsize=13, fontweight="bold")

    shooting_pct = np.array([r["shooting_pct"] for r in results_list])
    goals   = np.array([r["is_goal"] for r in results_list])
    xg_vals = np.array([r["sb_xg"] for r in results_list])

    # ── Panel 1: Pitch control distribution (goals vs misses) ─────────────
    ax = axes[0, 0]
    ax.hist(shooting_pct[goals == 1], bins=30, alpha=0.6, color="green",
            label=f"Goals (n={goals.sum()})", density=True)
    ax.hist(shooting_pct[goals == 0], bins=30, alpha=0.6, color="gray",
            label=f"Misses (n={(1-goals).sum()})", density=True)
    ax.axvline(shooting_pct[goals == 1].mean(), color="green", ls="--", lw=1.5)
    ax.axvline(shooting_pct[goals == 0].mean(), color="gray",  ls="--", lw=1.5)
    ax.set_xlabel("Shooting team pitch control %")
    ax.set_ylabel("Density")
    ax.set_title("Pitch Control: Goals vs Misses")
    ax.legend(fontsize=9)

    # ── Panel 2: Pitch control vs xG scatter ──────────────────────────────
    ax = axes[0, 1]
    ax.scatter(shooting_pct, xg_vals, s=8, alpha=0.3, c="steelblue")
    r_val, p_val = pearsonr(shooting_pct, xg_vals)
    ax.set_xlabel("Shooting team pitch control %")
    ax.set_ylabel("StatsBomb xG")
    ax.set_title(f"Pitch Control vs xG (r={r_val:.3f}, p={p_val:.1e})")

    # ── Panel 3: Example Voronoi for a high-xG goal ──────────────────────
    ax = axes[1, 0]
    # Find a goal with high xG for a clear visualization
    goal_indices = [i for i, r in enumerate(results_list)
                    if r["is_goal"] and r["sb_xg"] > 0.3]
    if goal_indices:
        idx = goal_indices[0]
        g = graphs[idx]
        positions, teams = extract_positions(g)

        pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH,
                      pitch_width=PITCH_WIDTH, line_color="white",
                      pitch_color="#1a1a2e")
        pitch.draw(ax=ax)

        # Plot Voronoi regions
        if len(positions) >= 4:
            # Add bounding points for bounded Voronoi
            bounded_pos = np.vstack([
                positions,
                [[-10, -10], [-10, PITCH_WIDTH+10],
                 [PITCH_LENGTH+10, -10], [PITCH_LENGTH+10, PITCH_WIDTH+10]]
            ])
            try:
                vor = Voronoi(bounded_pos)
                for r_idx, region in enumerate(vor.regions):
                    if not region or -1 in region or r_idx >= len(positions):
                        continue
                    # Find which point this region belongs to
                    point_idx = [i for i, ri in enumerate(vor.point_region)
                                 if ri == vor.point_region[r_idx]]
                    if not point_idx or point_idx[0] >= len(teams):
                        continue
                    polygon = [vor.vertices[v] for v in region]
                    polygon = np.array(polygon)
                    color = "steelblue" if teams[point_idx[0]] == 0 else "salmon"
                    ax.fill(*polygon.T, alpha=0.2, color=color)
            except Exception:
                pass

        # Plot players
        shoot_mask = teams == 0
        ax.scatter(positions[shoot_mask, 0], positions[shoot_mask, 1],
                   c="steelblue", s=50, zorder=5, edgecolors="white",
                   label="Shooting team")
        ax.scatter(positions[~shoot_mask, 0], positions[~shoot_mask, 1],
                   c="salmon", s=50, zorder=5, edgecolors="white",
                   label="Defending team")
        ax.set_title(f"Voronoi — {g.player_name} goal "
                     f"(xG={results_list[idx]['sb_xg']:.2f})")
        ax.legend(fontsize=7, loc="upper left")

    # ── Panel 4: Per-competition mean pitch control ───────────────────────
    ax = axes[1, 1]
    comp_data = defaultdict(list)
    for r in results_list:
        comp_data[r["comp_label"]].append(r["shooting_pct"])
    comps = sorted(comp_data.keys())
    means = [np.mean(comp_data[c]) for c in comps]
    stds  = [np.std(comp_data[c])  for c in comps]
    x = np.arange(len(comps))
    short = [c.replace("bundesliga", "buli") for c in comps]
    bars = ax.bar(x, means, yerr=stds, color="steelblue", alpha=0.7,
                  capsize=3, error_kw={"lw": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean shooting team control %")
    ax.set_title("Pitch Control at Shot Moment by Competition")
    ax.axhline(0.5, color="black", ls=":", lw=1)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Voronoi Pitch Control Analysis")
    print("=" * 64)

    print("\n── Loading graphs ──────────────────────────────────────────")
    graphs = load_graphs()

    print(f"\n── Computing pitch control for {len(graphs)} shots …")
    results_list = []

    for i, g in enumerate(graphs):
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{len(graphs)} …")

        positions, teams = extract_positions(g)
        pc = compute_pitch_control(positions, teams)

        results_list.append({
            "match_id":     int(g.match_id.item()),
            "player_name":  g.player_name,
            "team_name":    g.team_name,
            "comp_label":   getattr(g, "comp_label", "unknown"),
            "is_goal":      int(g.y.item()),
            "sb_xg":        float(g.sb_xg.item()),
            "shooting_pct": pc["shooting_pct"],
            "defending_pct": pc["defending_pct"],
            "n_players":    len(positions),
        })

    # ── Statistics ────────────────────────────────────────────────────────
    shooting_pcts = np.array([r["shooting_pct"] for r in results_list])
    goals = np.array([r["is_goal"] for r in results_list])
    xg_vals = np.array([r["sb_xg"] for r in results_list])

    print(f"\n── Results ──��──────────────────────────────────────────────")
    print(f"  Mean shooting team control: {shooting_pcts.mean():.3f} "
          f"(±{shooting_pcts.std():.3f})")
    print(f"  Goals mean control:  {shooting_pcts[goals==1].mean():.3f}")
    print(f"  Misses mean control: {shooting_pcts[goals==0].mean():.3f}")
    print(f"  Δ (goals - misses):  {shooting_pcts[goals==1].mean() - shooting_pcts[goals==0].mean():+.4f}")

    r_xg, p_xg = pearsonr(shooting_pcts, xg_vals)
    print(f"\n  Correlation with StatsBomb xG: r={r_xg:.4f} (p={p_xg:.2e})")

    r_goal, p_goal = pearsonr(shooting_pcts, goals)
    print(f"  Correlation with goal outcome: r={r_goal:.4f} (p={p_goal:.2e})")

    # Per-competition
    comp_stats = defaultdict(list)
    for r in results_list:
        comp_stats[r["comp_label"]].append(r["shooting_pct"])
    print(f"\n  Per-competition shooting team control:")
    print(f"  {'Comp':<20} {'N':>5}  {'Mean':>6}  {'Std':>6}")
    print(f"  {'-'*42}")
    comp_summary = {}
    for comp in sorted(comp_stats.keys()):
        vals = comp_stats[comp]
        m, s = np.mean(vals), np.std(vals)
        print(f"  {comp:<20} {len(vals):>5}  {m:>6.3f}  {s:>6.3f}")
        comp_summary[comp] = {"n": len(vals), "mean": round(m, 4), "std": round(s, 4)}

    # ── Save ────────��─────────────────────────────────────────────────────
    out = {
        "description": "Voronoi pitch control from shot freeze-frames",
        "n_shots": len(results_list),
        "mean_shooting_pct": round(float(shooting_pcts.mean()), 4),
        "goals_mean_pct": round(float(shooting_pcts[goals==1].mean()), 4),
        "misses_mean_pct": round(float(shooting_pcts[goals==0].mean()), 4),
        "corr_xg": round(r_xg, 4),
        "corr_goal": round(r_goal, 4),
        "per_competition": comp_summary,
        "shots": results_list,
    }
    out_path = PROCESSED / "pitch_control_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Figure ─────────────────────────────────────���──────────────────────
    print("\n── Generating figure …")
    fig = make_figure(results_list, graphs)
    fig_path = ASSETS / "fig_pitch_control.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
