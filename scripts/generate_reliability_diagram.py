#!/usr/bin/env python3
"""
generate_reliability_diagram.py
--------------------------------
Reliability diagram (calibration curve) comparing three xG sources:

  • HybridGAT (raw, uncalibrated logits)
  • HybridGAT+T (per-competition temperature scaled)
  • StatsBomb xG (industry reference)

Produces: assets/fig_reliability.png

Usage
-----
  python scripts/generate_reliability_diagram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import random

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.hybrid_gat import HybridGATModel

PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SEED = 42
BATCH = 256
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ── colours ────────────────────────────────────────────────────────────────
C_RAW  = "#94A3B8"   # raw uncalibrated — grey
C_CAL  = "#2563EB"   # calibrated GAT+T — blue
C_SB   = "#DC2626"   # StatsBomb — red
C_DIAG = "#64748B"   # perfect calibration diagonal


# ---------------------------------------------------------------------------
# Data helpers (exact mirror of ablation_rq123.py)
# ---------------------------------------------------------------------------

def load_graphs():
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
    return graphs


def stratified_split(graphs, train_frac=0.70, val_frac=0.15):
    rng = random.Random(SEED)
    goals    = [g for g in graphs if g.y.item() == 1]
    no_goals = [g for g in graphs if g.y.item() == 0]
    rng.shuffle(goals); rng.shuffle(no_goals)

    def spl(lst):
        t = int(len(lst) * train_frac)
        v = int(len(lst) * (train_frac + val_frac))
        return lst[:t], lst[t:v], lst[v:]

    g_tr, g_va, g_te = spl(goals)
    n_tr, n_va, n_te = spl(no_goals)
    tr = g_tr + n_tr; rng.shuffle(tr)
    va = g_va + n_va; rng.shuffle(va)
    te = g_te + n_te; rng.shuffle(te)
    return tr, va, te


def build_meta(batch) -> torch.Tensor:
    """27-dim metadata (with shot_placement) — same as ablation_rq123."""
    n = batch.shot_dist.shape[0]

    def _safe(attr, default):
        if hasattr(batch, attr):
            return getattr(batch, attr).squeeze()
        return torch.full((n,), default)

    base = torch.stack([
        batch.shot_dist.squeeze(),
        batch.shot_angle.squeeze(),
        batch.is_header.squeeze().float(),
        batch.is_open_play.squeeze().float(),
    ], dim=1)
    tech = batch.technique.view(-1, 8)
    gk   = torch.stack([
        batch.gk_dist.squeeze(),
        batch.n_def_in_cone.squeeze(),
        batch.gk_off_centre.squeeze(),
    ], dim=1)
    new  = torch.stack([
        _safe("gk_perp_offset", 3.0),
        _safe("n_def_direct_line", 0.0),
        _safe("is_right_foot", 0.5),
    ], dim=1)
    plc = (batch.shot_placement.view(-1, 9)
           if hasattr(batch, "shot_placement")
           else torch.zeros(n, 9))
    return torch.cat([base, tech, gk, new, plc], dim=1)


@torch.no_grad()
def infer(model, graphs, T_map: dict, T_global: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs_raw, probs_cal) for the full graph list."""
    model.eval()
    raw_list, cal_list, comp_list = [], [], []
    for batch in DataLoader(graphs, batch_size=BATCH):
        meta      = build_meta(batch)
        edge_attr = batch.edge_attr if batch.edge_attr is not None else None
        logits    = model(batch.x, batch.edge_index, batch.batch, meta,
                          edge_attr=edge_attr).squeeze()
        raw_list.append(torch.sigmoid(logits).numpy())
        # gather comp labels for this batch
        comp_list.extend(getattr(batch, "comp_label",
                         ["unknown"] * batch.num_graphs))
    probs_raw = np.concatenate(raw_list)

    # Per-competition temperature scaling
    comp_labels = [getattr(g, "comp_label", "unknown") or "unknown"
                   for g in graphs]
    logit_raw = np.log(probs_raw / (1 - probs_raw + 1e-9) + 1e-9)
    probs_cal = np.array([
        1 / (1 + np.exp(-logit_raw[i] / T_map.get(cl, T_global)))
        for i, cl in enumerate(comp_labels)
    ])
    return probs_raw, probs_cal


def compute_ece(y, p, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i+1])
        if m.sum() == 0:
            continue
        ece += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_diagram(y_true: np.ndarray,
                 probs_raw: np.ndarray,
                 probs_cal: np.ndarray,
                 probs_sb:  np.ndarray,
                 n_bins: int = 15):

    fig = plt.figure(figsize=(9, 8), facecolor="white")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # ── Calibration curves ────────────────────────────────────────────────
    sources = [
        ("HybridGAT (raw)", probs_raw, C_RAW, "--", 1.5),
        ("HybridGAT+T (per-comp calibrated)", probs_cal, C_CAL, "-",  2.0),
        ("StatsBomb xG (industry ref.)",       probs_sb,  C_SB,  "-",  2.0),
    ]

    brier_vals = {}
    ece_vals   = {}
    for label, p, col, ls, lw in sources:
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins,
                                                strategy="uniform")
        ax_main.plot(mean_pred, frac_pos, color=col, linestyle=ls, linewidth=lw,
                     marker="o", markersize=5, label=label, zorder=3)
        brier_vals[label] = brier_score_loss(y_true, p)
        ece_vals[label]   = compute_ece(y_true, p, n_bins=n_bins)

    # Perfect calibration diagonal
    ax_main.plot([0, 1], [0, 1], color=C_DIAG, linestyle=":", linewidth=1.2,
                 label="Perfect calibration", zorder=1)

    # Shaded overconfident / underconfident regions
    ax_main.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="#16A34A",
                         label="_nolegend_")
    ax_main.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="#DC2626",
                         label="_nolegend_")
    ax_main.text(0.72, 0.58, "over\nconfident", fontsize=8, color="#94A3B8",
                 ha="center", style="italic")
    ax_main.text(0.28, 0.42, "under\nconfident", fontsize=8, color="#94A3B8",
                 ha="center", style="italic")

    ax_main.set_xlim(0, 1); ax_main.set_ylim(0, 1)
    ax_main.set_ylabel("Observed goal rate (fraction positive)", fontsize=11)
    ax_main.set_xticklabels([])

    # Metric annotations
    pad = 0.012
    y_pos = 0.975
    ann_cols = {
        "HybridGAT (raw)":                    C_RAW,
        "HybridGAT+T (per-comp calibrated)":  C_CAL,
        "StatsBomb xG (industry ref.)":        C_SB,
    }
    for name, col in ann_cols.items():
        b = brier_vals[name]; e = ece_vals[name]
        ax_main.annotate(
            f"Brier={b:.3f}  ECE={e:.3f}",
            xy=(0.98, y_pos), xycoords="axes fraction",
            fontsize=8.5, color=col, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=col, lw=0.9, alpha=0.85),
        )
        y_pos -= 0.095

    ax_main.legend(loc="upper left", fontsize=9.5, frameon=True,
                   framealpha=0.9, edgecolor="#CBD5E1")
    ax_main.set_title(
        "Reliability Diagram — xG Model Calibration\n"
        f"n={len(y_true):,} shots · {int(y_true.sum())} goals "
        f"({100*y_true.mean():.1f}%) · 7 competitions",
        fontsize=12, fontweight="bold", color="#1E293B",
    )
    ax_main.spines[["top", "right"]].set_visible(False)

    # ── Probability distribution histogram (bottom panel) ─────────────────
    bins = np.linspace(0, 1, n_bins + 1)
    for label, p, col, _, _ in sources:
        ax_hist.hist(p, bins=bins, density=True, histtype="step",
                     color=col, linewidth=1.5, alpha=0.9)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Mean predicted xG", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.set_yticks([])

    # Brier improvement annotation
    delta_b = brier_vals["HybridGAT (raw)"] - brier_vals["HybridGAT+T (per-comp calibrated)"]
    delta_e = ece_vals["HybridGAT (raw)"]   - ece_vals["HybridGAT+T (per-comp calibrated)"]
    ax_hist.text(
        0.98, 0.97,
        f"T-scaling: ΔBrier={delta_b:+.3f}  ΔECE={delta_e:+.3f}",
        transform=ax_hist.transAxes, fontsize=8.5, ha="right", va="top",
        color=C_CAL,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_CAL, lw=0.9),
    )

    path = ASSETS / "fig_reliability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")
    return brier_vals, ece_vals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading graphs …")
    all_g = load_graphs()
    _, _, test_g = stratified_split(all_g)
    y_test = np.array([g.y.item() for g in test_g])
    sb_xg  = np.array([g.sb_xg.item() for g in test_g])
    print(f"  test n={len(test_g)}  goals={int(y_test.sum())}")

    print("Loading HybridGAT model …")
    ckpt = torch.load(PROCESSED / "pool_7comp_hybrid_gat_xg.pt",
                      weights_only=True, map_location="cpu")
    pool_dim    = 32
    actual_meta = int(ckpt["head.0.weight"].shape[1]) - pool_dim
    edge_dim    = int(ckpt["convs.0.lin_edge.weight"].shape[1])
    model = HybridGATModel(node_in=9, edge_dim=edge_dim,
                           meta_dim=actual_meta, hidden=32, heads=4,
                           n_layers=3, dropout=0.0)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  meta_dim={actual_meta}  edge_dim={edge_dim}")

    # Load T
    T_global = float(torch.load(PROCESSED / "pool_7comp_gat_T.pt",
                                weights_only=True)["T"])
    T_map    = torch.load(PROCESSED / "pool_7comp_per_comp_T_gat.pt",
                          weights_only=True)
    print(f"  Global T={T_global:.4f}  per-comp T={T_map}")

    print("Running inference …")
    probs_raw, probs_cal = infer(model, test_g, T_map, T_global)

    print("Generating reliability diagram …")
    brier_vals, ece_vals = make_diagram(y_test, probs_raw, probs_cal, sb_xg)

    print("\nCalibration Summary:")
    print(f"  {'Model':<45}  {'Brier':>6}  {'ECE':>6}")
    print(f"  {'-'*62}")
    for name in brier_vals:
        print(f"  {name:<45}  {brier_vals[name]:.3f}  {ece_vals[name]:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
