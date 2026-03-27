#!/usr/bin/env python3
"""
generate_paper_figures.py
--------------------------
Generates two paper-ready figures:

  Fig 1 — Graph construction pipeline
           freeze-frame positions → Delaunay triangulation → annotated node diagram
           Saved: assets/fig_graph_construction.png

  Fig 2 — HybridGATv2 architecture block diagram
           GATv2 branch + metadata MLP branch → concat → head → xG
           Saved: assets/fig_architecture.png

Usage
-----
  python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.spatial import Delaunay
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
PROCESSED  = REPO_ROOT / "data" / "processed"
ASSETS     = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────
C_BLUE   = "#2563EB"   # attacking team / graph branch
C_RED    = "#DC2626"   # defending team
C_GK     = "#16A34A"   # goalkeeper
C_EDGE   = "#64748B"   # edges
C_META   = "#7C3AED"   # metadata branch
C_HEAD   = "#EA580C"   # head / output
C_CONCAT = "#0891B2"   # concat block
C_BG     = "#F8FAFC"
GREY     = "#CBD5E1"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.facecolor": "white",
})

# ===========================================================================
# Fig 1 — Graph construction pipeline
# ===========================================================================

def _load_example_shot() -> dict | None:
    """Try to load a real freeze-frame from saved graph data."""
    for pt in sorted(PROCESSED.glob("statsbomb_wc2022_shot_graphs.pt")):
        graphs = torch.load(pt, weights_only=False)
        # pick a shot with a decent number of players
        for g in graphs:
            if g.x.shape[0] >= 10:
                return g
    return None


def _synthetic_freeze_frame():
    """Synthetic representative freeze-frame if real data not available."""
    np.random.seed(7)
    # shooter at ~22m from goal, centre-left
    shooter = np.array([[100.0, 40.0]])
    # GK on line
    gk      = np.array([[120.0, 40.0]])
    # 3 defenders between shooter and goal
    defenders = np.array([
        [112.0, 37.0],
        [113.5, 42.0],
        [110.0, 39.5],
    ])
    # 4 attackers supporting
    attackers = np.array([
        [98.0,  33.0],
        [95.0,  47.0],
        [103.0, 52.0],
        [101.0, 28.0],
    ])
    # 5 extra defenders scattered
    def_extra = np.array([
        [108.0, 31.0],
        [107.0, 48.0],
        [115.0, 33.0],
        [116.0, 46.0],
        [118.0, 41.0],
    ])
    positions = np.vstack([shooter, attackers, gk, defenders, def_extra])
    # roles: 0=shooter, 1=attacker, 2=gk, 3=defender
    roles = (
        [0] * 1 +
        [1] * len(attackers) +
        [2] * 1 +
        [3] * (len(defenders) + len(def_extra))
    )
    return positions, roles


def draw_pitch_outline(ax, length=120, width=80, alpha=0.4):
    """Draw a minimal half-pitch outline (attacking third)."""
    # Field outline — attacking third only
    ax.set_xlim(85, 123)
    ax.set_ylim(22, 58)
    ax.set_aspect("equal")

    lw = 1.2
    col = "#94A3B8"
    # pitch boundary lines (partial)
    ax.plot([85, 120], [22, 22], color=col, lw=lw)
    ax.plot([85, 120], [58, 58], color=col, lw=lw)
    ax.plot([120, 120], [22, 58], color=col, lw=lw)
    # penalty box
    ax.plot([102, 102, 120, 120], [23.7, 56.3, 56.3, 23.7],
            color=col, lw=lw, linestyle="--", alpha=0.6)
    # goal
    ax.plot([120, 120], [36, 44], color=col, lw=3, solid_capstyle="round")
    ax.set_axis_off()


def fig1_graph_construction():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="white")
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.88, bottom=0.04)

    positions, roles = _synthetic_freeze_frame()
    tri = Delaunay(positions)

    role_color = {0: C_BLUE, 1: C_BLUE, 2: C_GK, 3: C_RED}
    role_label = {0: "Shooter", 1: "Attacker", 2: "GK", 3: "Defender"}
    node_size  = {0: 180, 1: 120, 2: 180, 3: 120}

    titles = [
        "① Freeze-frame positions",
        "② Delaunay triangulation\n(edges = spatial proximity)",
        "③ Typed graph with node features",
    ]

    for col_idx, ax in enumerate(axes):
        draw_pitch_outline(ax)
        ax.set_title(titles[col_idx], fontsize=11, fontweight="bold",
                     color="#1E293B", pad=8)

        # Panel 1 & 2: just dots
        if col_idx >= 1:
            # Draw triangulation edges
            for simplex in tri.simplices:
                pts = positions[simplex]
                triangle = plt.Polygon(pts, fill=False, edgecolor=C_EDGE,
                                       linewidth=0.9, alpha=0.45, zorder=1)
                ax.add_patch(triangle)

        for i, (pos, role) in enumerate(zip(positions, roles)):
            ax.scatter(*pos, s=node_size[role], c=role_color[role],
                       zorder=3, edgecolors="white", linewidths=1.0)

        # Panel 3: annotate node features on a couple of key nodes
        if col_idx == 2:
            # Shooter node annotation
            ax.annotate(
                "dist=22m\nangle=14°\nteam=1\nactor=1",
                xy=positions[0], xytext=(-14, 8),
                textcoords="offset points",
                fontsize=7, color="#1E293B",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C_BLUE, lw=1.1, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.9),
                zorder=5,
            )
            # GK annotation
            gk_idx = 5
            ax.annotate(
                "dist=1m\nkeeper=1\nteam=0\nactor=0",
                xy=positions[gk_idx],
                xytext=(6, 8), textcoords="offset points",
                fontsize=7, color="#1E293B",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C_GK, lw=1.1, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_GK, lw=0.9),
                zorder=5,
            )
            # Edge annotation
            mid = (positions[0] + positions[6]) / 2
            ax.annotate(
                "edge: [dist, Δx,\nΔy, same_team,\npass_angle]",
                xy=mid, xytext=(-28, -24),
                textcoords="offset points",
                fontsize=6.5, color="#475569",
                bbox=dict(boxstyle="round,pad=0.3", fc="#F1F5F9",
                          ec=C_EDGE, lw=0.9, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=C_EDGE, lw=0.8),
                zorder=5,
            )

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=C_BLUE,  label="Shooter / Attacker"),
        mpatches.Patch(facecolor=C_GK,    label="Goalkeeper"),
        mpatches.Patch(facecolor=C_RED,   label="Defender"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    path = ASSETS / "fig_graph_construction.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


# ===========================================================================
# Fig 2 — HybridGATv2 architecture block diagram
# ===========================================================================

def _box(ax, x, y, w, h, label, sublabel="", color="#2563EB",
         fontsize=9, sublabel_size=7.5, radius=0.04, text_color="white"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=color, edgecolor="white",
                          linewidth=1.5, zorder=2)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.035, label,  ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)
        ax.text(x, y - 0.055, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color=text_color, alpha=0.85, zorder=3)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color="#64748B", lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12),
                zorder=1)


def fig2_architecture():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    bw, bh = 0.145, 0.10   # standard box width / height
    bh_sm  = 0.08

    # ── LEFT BRANCH: GATv2 graph branch ────────────────────────────────────
    lx = 0.22    # x-centre of left branch

    _box(ax, lx, 0.88, bw, bh_sm, "Freeze-frame",
         "9 node features × N nodes", C_BLUE)

    _box(ax, lx, 0.72, bw, bh_sm, "GATv2Conv × 3",
         "hidden=32, heads=4, edge_dim=6", C_BLUE)
    _arrow(ax, lx, 0.83, lx, 0.77)

    _box(ax, lx, 0.56, bw, bh_sm, "Global Mean Pool",
         "→ [batch, 128]", C_BLUE)
    _arrow(ax, lx, 0.67, lx, 0.61)

    _box(ax, lx, 0.40, bw, bh_sm, "Linear 128→32",
         "graph embedding", C_BLUE)
    _arrow(ax, lx, 0.51, lx, 0.45)

    # edge features label
    ax.text(lx - 0.13, 0.72, "Edge attr\n[dist, Δx, Δy,\nsame_team,\npass_angle,\nvel_align]",
            ha="center", va="center", fontsize=6.8, color="#475569",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec=C_BLUE, lw=0.8, alpha=0.9))
    _arrow(ax, lx - 0.065, 0.72, lx - 0.075, 0.72, color=C_BLUE, lw=0.9)

    # ── RIGHT BRANCH: metadata MLP branch ──────────────────────────────────
    rx = 0.78    # x-centre of right branch

    _box(ax, rx, 0.88, bw, bh_sm, "Shot Metadata",
         "27-dim vector", C_META)

    # metadata breakdown annotation
    meta_lines = (
        "4d  distance, angle, header, open_play\n"
        "8d  technique (one-hot)\n"
        "3d  GK: dist, cone, off-centre\n"
        "3d  GK perp, defenders, foot\n"
        "9d  shot placement zone (PSxG)"
    )
    ax.text(rx + 0.145, 0.88, meta_lines,
            ha="left", va="center", fontsize=6.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FAF5FF",
                      ec=C_META, lw=0.8, alpha=0.9))

    _box(ax, rx, 0.72, bw, bh_sm, "MLP Layer 1",
         "27→64, ReLU, Dropout 0.3", C_META)
    _arrow(ax, rx, 0.83, rx, 0.77)

    _box(ax, rx, 0.56, bw, bh_sm, "MLP Layer 2",
         "64→32, ReLU", C_META)
    _arrow(ax, rx, 0.67, rx, 0.61)

    _box(ax, rx, 0.40, bw, bh_sm, "Metadata embed",
         "→ [batch, 32]", C_META)
    _arrow(ax, rx, 0.51, rx, 0.45)

    # ── CONCAT ─────────────────────────────────────────────────────────────
    cx = 0.50
    _box(ax, cx, 0.27, 0.18, bh_sm, "Concat",
         "graph (32) ‖ meta (32) → 64", C_CONCAT)
    _arrow(ax, lx, 0.35, cx - 0.06, 0.27, color=C_CONCAT)
    _arrow(ax, rx, 0.35, cx + 0.06, 0.27, color=C_CONCAT)

    # ── HEAD ────────────────────────────────────────────────────────────────
    _box(ax, cx, 0.14, bw, bh_sm, "MLP Head",
         "64→32→1, ReLU, Dropout 0.3", C_HEAD)
    _arrow(ax, cx, 0.22, cx, 0.19)

    _box(ax, cx, 0.03, bw, 0.07, "σ(logit / T)",
         "calibrated xG ∈ (0,1)", C_HEAD)
    _arrow(ax, cx, 0.095, cx, 0.065)

    # ── Temperature annotation ─────────────────────────────────────────────
    ax.text(cx + 0.155, 0.03,
            "T = per-competition\ntemperature scalar\n(0.544 – 0.872)",
            ha="left", va="center", fontsize=7, color="#374151",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFF7ED",
                      ec=C_HEAD, lw=0.8, alpha=0.9))

    # ── Training loss annotation ────────────────────────────────────────────
    ax.text(0.02, 0.03,
            "Training: Weighted BCE\n(class_weight=balanced)\nOptimiser: AdamW lr=1e-3",
            ha="left", va="center", fontsize=7, color="#374151",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F0FDF4",
                      ec="#16A34A", lw=0.8, alpha=0.9))

    # ── Branch labels ───────────────────────────────────────────────────────
    ax.text(lx, 0.98, "① Graph Branch (GATv2)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_BLUE)
    ax.text(rx, 0.98, "② Metadata Branch (MLP)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_META)
    ax.text(cx, 0.98, "③ Head",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_HEAD)

    # vertical dividers
    for x_div in [0.42, 0.58]:
        ax.axvline(x_div, color=GREY, lw=0.8, linestyle="--",
                   ymin=0.05, ymax=0.96, alpha=0.6)

    fig.suptitle("HybridGATv2 Architecture — Freeze-frame Graph + Shot Metadata → xG",
                 fontsize=13, fontweight="bold", color="#1E293B", y=1.01)

    path = ASSETS / "fig_architecture.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("Generating Fig 1 — Graph construction …")
    fig1_graph_construction()

    print("Generating Fig 2 — Architecture diagram …")
    fig2_architecture()

    print("\nDone.")
