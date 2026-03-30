#!/usr/bin/env python3
"""
mc_dropout_uncertainty.py
-------------------------
Monte Carlo Dropout uncertainty estimation for HybridGATv2 xG model.

Uses the existing dropout=0.3 layers as an approximate Bayesian method:
run N stochastic forward passes in train() mode → per-shot xG distribution.

Outputs
-------
  data/processed/mc_dropout_results.json   per-shot mean/std/cv + summary stats
  assets/fig_mc_dropout.png                4-panel uncertainty figure for paper

Usage
-----
  python scripts/mc_dropout_uncertainty.py
  python scripts/mc_dropout_uncertainty.py --n-samples 500 --batch 32
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.hybrid_gat import HybridGATModel

PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SEED       = 42
BATCH_SIZE = 64
META_DIM   = 27

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Data helpers  (mirrors stratified_split from train_xg_hybrid.py exactly)
# ---------------------------------------------------------------------------

def load_graphs() -> list:
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
        print(f"  {pt.name}: {len(gs)}")
    print(f"  Total: {len(graphs)}")
    return graphs


def stratified_split(graphs, train_frac=0.70, val_frac=0.15):
    rng = random.Random(SEED)
    goals    = [g for g in graphs if g.y.item() == 1]
    no_goals = [g for g in graphs if g.y.item() == 0]
    rng.shuffle(goals)
    rng.shuffle(no_goals)

    def split(lst):
        t = int(len(lst) * train_frac)
        v = int(len(lst) * (train_frac + val_frac))
        return lst[:t], lst[t:v], lst[v:]

    g_tr, g_va, g_te = split(goals)
    n_tr, n_va, n_te = split(no_goals)
    tr = g_tr + n_tr; rng.shuffle(tr)
    va = g_va + n_va; rng.shuffle(va)
    te = g_te + n_te; rng.shuffle(te)
    return tr, va, te


# ---------------------------------------------------------------------------
# 27-dim metadata tensor  (must match _metadata_tensor in train_xg_hybrid.py)
# ---------------------------------------------------------------------------

def build_meta(batch) -> torch.Tensor:
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
        batch.gk_perp_offset.squeeze(),
        batch.n_def_direct_line.squeeze(),
        batch.is_right_foot.squeeze(),
    ], dim=1)
    plc  = batch.shot_placement.view(-1, 9)
    return torch.cat([base, tech, gk, new, plc], dim=1)  # [n, 27]


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def infer_mc_dropout(
    model: HybridGATModel,
    graphs: list,
    n_samples: int = 200,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run N stochastic forward passes with dropout active.

    Parameters
    ----------
    model     : HybridGATModel (loaded weights, dropout=0.3)
    graphs    : list of PyG Data objects
    n_samples : number of MC samples per shot
    batch_size: mini-batch size for memory efficiency

    Returns
    -------
    means : (N,) mean predicted xG per shot
    stds  : (N,) std deviation of xG across samples
    """
    # activate dropout by staying in train() mode — DO NOT call model.eval()
    model.train()

    loader      = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_samples = []  # will be shape (n_samples, N_shots)

    with torch.no_grad():
        for s in range(n_samples):
            probs_s = []
            for batch in loader:
                meta    = build_meta(batch)
                ea      = batch.edge_attr if batch.edge_attr is not None else None
                logits  = model(batch.x, batch.edge_index, batch.batch,
                                meta, edge_attr=ea)
                probs_s.append(torch.sigmoid(logits).squeeze().cpu())
            all_samples.append(torch.cat(probs_s).numpy())

    samples = np.stack(all_samples, axis=0)  # (n_samples, N_shots)
    means   = samples.mean(axis=0)
    stds    = samples.std(axis=0)
    return means, stds


# ---------------------------------------------------------------------------
# Per-competition temperature calibration (apply after MC mean)
# ---------------------------------------------------------------------------

def load_per_comp_T() -> dict:
    path = PROCESSED / "pool_7comp_per_comp_T_gat.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    return {}


def apply_per_comp_T(means: np.ndarray, graphs: list, per_T: dict,
                     T_global: float) -> np.ndarray:
    """Re-calibrate mean predictions using per-competition temperature."""
    logits   = np.log(means / (1 - means + 1e-9) + 1e-9)
    calibrated = means.copy()
    for i, g in enumerate(graphs):
        cl = getattr(g, "comp_label", "unknown") or "unknown"
        T  = per_T.get(cl, T_global)
        calibrated[i] = 1 / (1 + np.exp(-logits[i] / T))
    return calibrated


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    y_true:    np.ndarray,
    means_cal: np.ndarray,
    stds:      np.ndarray,
    sb_xg:     np.ndarray,
) -> plt.Figure:
    """
    4-panel MC Dropout uncertainty figure.

    Panel 1: Histogram of prediction std by shot outcome (goal vs no-goal)
    Panel 2: xG_mean vs xG_std scatter (coloured by outcome)
    Panel 3: Calibration curve — MC mean vs StatsBomb reference
    Panel 4: Uncertainty-stratified Brier scores (low/mid/high uncertainty terciles)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("MC Dropout Uncertainty — HybridGATv2  (N=200 samples)",
                 fontsize=13, fontweight="bold")

    goals   = y_true == 1
    no_goal = y_true == 0
    cv      = stds / (means_cal + 1e-9)   # coefficient of variation

    # ── Panel 1: std distribution by outcome ─────────────────────────────────
    ax = axes[0, 0]
    bins = np.linspace(0, stds.max() * 1.05, 40)
    ax.hist(stds[no_goal], bins=bins, alpha=0.6, color="steelblue",
            label=f"No Goal  (n={no_goal.sum()})", density=True)
    ax.hist(stds[goals], bins=bins, alpha=0.7, color="tomato",
            label=f"Goal  (n={goals.sum()})", density=True)
    ax.axvline(stds.mean(), color="k", ls="--", lw=1.2,
               label=f"Mean σ = {stds.mean():.4f}")
    ax.set_xlabel("Prediction Std (σ)")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution by Outcome")
    ax.legend(fontsize=8)

    # ── Panel 2: xG mean vs std scatter ──────────────────────────────────────
    ax = axes[0, 1]
    sc = ax.scatter(means_cal[no_goal], stds[no_goal], s=4, alpha=0.3,
                    color="steelblue", label="No Goal", rasterized=True)
    ax.scatter(means_cal[goals], stds[goals], s=8, alpha=0.6,
               color="tomato", label="Goal", rasterized=True)
    ax.set_xlabel("xG Mean (calibrated)")
    ax.set_ylabel("xG Std (σ)")
    ax.set_title("Uncertainty vs Predicted xG")
    ax.legend(fontsize=8, markerscale=3)
    # annotate mean σ in each quintile
    q_edges = np.quantile(means_cal, [0, 0.25, 0.5, 0.75, 1.0])
    for lo, hi in zip(q_edges[:-1], q_edges[1:]):
        mask = (means_cal >= lo) & (means_cal < hi)
        if mask.sum() > 0:
            ax.text((lo + hi) / 2, stds[mask].mean(),
                    f"σ={stds[mask].mean():.3f}", fontsize=6.5,
                    ha="center", va="bottom", color="darkgreen")

    # ── Panel 3: Calibration — MC mean vs StatsBomb ───────────────────────────
    ax = axes[1, 0]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for probs, label, color, ls in [
        (means_cal, "MC Mean (calibrated)", "steelblue", "-"),
        (sb_xg,     "StatsBomb xG",         "tomato",    "--"),
    ]:
        frac_pos, mean_pred = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() >= 5:
                frac_pos.append(y_true[mask].mean())
                mean_pred.append(probs[mask].mean())
        ax.plot(mean_pred, frac_pos, color=color, lw=1.5,
                label=label, ls=ls, marker="o", ms=5)
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfect")
    ax.set_xlabel("Mean Predicted xG")
    ax.set_ylabel("Observed Goal Rate")
    ax.set_title("Calibration: MC Mean vs StatsBomb")
    ax.legend(fontsize=8)

    # ── Panel 4: Brier by uncertainty tercile ────────────────────────────────
    ax = axes[1, 1]
    terciles = np.quantile(stds, [1/3, 2/3])
    t_labels = ["Low σ\n(certain)", "Mid σ", "High σ\n(uncertain)"]
    t_masks  = [
        stds < terciles[0],
        (stds >= terciles[0]) & (stds < terciles[1]),
        stds >= terciles[1],
    ]
    briers_mc = [brier_score_loss(y_true[m], means_cal[m]) for m in t_masks]
    briers_sb = [brier_score_loss(y_true[m], sb_xg[m])     for m in t_masks]
    x = np.arange(3)
    w = 0.35
    bars_mc = ax.bar(x - w/2, briers_mc, w, color="steelblue",
                     alpha=0.8, label="MC Mean")
    bars_sb = ax.bar(x + w/2, briers_sb, w, color="tomato",
                     alpha=0.8, label="StatsBomb xG")
    for bars in [bars_mc, bars_sb]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(t_labels)
    ax.set_ylabel("Brier Score")
    ax.set_title("Brier Score by Uncertainty Tercile")
    ax.legend(fontsize=8)
    sizes = [m.sum() for m in t_masks]
    for i, s in enumerate(sizes):
        ax.text(i, 0.002, f"n={s}", ha="center", va="bottom",
                fontsize=7, color="gray")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=200,
                   help="Number of MC dropout forward passes (default: 200)")
    p.add_argument("--batch",     type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print(f"  MC Dropout Uncertainty  (N={args.n_samples} samples)")
    print("=" * 64)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n── Loading graphs ───────────────────────────────────────────")
    graphs = load_graphs()
    _, _, test_g = stratified_split(graphs)
    print(f"  Test set: {len(test_g)} shots")

    y_true = np.array([g.y.item() for g in test_g])
    sb_xg  = np.array([g.sb_xg.item() for g in test_g])

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = PROCESSED / "pool_7comp_hybrid_gat_xg.pt"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    sample_g  = test_g[0]
    in_ch     = sample_g.x.shape[1]
    edge_ch   = sample_g.edge_attr.shape[1] if sample_g.edge_attr is not None else 4

    model = HybridGATModel(
        node_in=in_ch, edge_dim=edge_ch,
        meta_dim=META_DIM, hidden=32, heads=4, n_layers=3, dropout=0.3,
    )
    state = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state)
    print(f"\n  Loaded: {ckpt_path.name}")

    # ── MC Dropout inference ──────────────────────────────────────────────────
    print(f"\n── Running {args.n_samples} stochastic forward passes …")
    means_raw, stds = infer_mc_dropout(model, test_g,
                                       n_samples=args.n_samples,
                                       batch_size=args.batch)
    print(f"  Done.  mean(σ)={stds.mean():.4f}  max(σ)={stds.max():.4f}")

    # ── Calibrate means with per-competition T ────────────────────────────────
    per_T    = load_per_comp_T()
    T_global_path = PROCESSED / "pool_7comp_gat_T.pt"
    T_global = 1.0
    if T_global_path.exists():
        d = torch.load(T_global_path, weights_only=False)
        T_global = float(d["T"]) if isinstance(d, dict) else float(d)
    means_cal = apply_per_comp_T(means_raw, test_g, per_T, T_global)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n── Metrics ──────────────────────────────────────────────────")
    cv  = stds / (means_cal + 1e-9)  # coefficient of variation

    auc_mc  = roc_auc_score(y_true, means_cal)
    auc_sb  = roc_auc_score(y_true, sb_xg)
    brier_mc = brier_score_loss(y_true, means_cal)
    brier_sb = brier_score_loss(y_true, sb_xg)

    # Uncertainty-stratified metrics
    terciles = np.quantile(stds, [1/3, 2/3])
    t_labels = ["low_sigma", "mid_sigma", "high_sigma"]
    t_masks  = [
        stds < terciles[0],
        (stds >= terciles[0]) & (stds < terciles[1]),
        stds >= terciles[1],
    ]

    print(f"\n  {'Model':<40} {'AUC':>6}  {'Brier':>7}")
    print(f"  {'-'*56}")
    print(f"  {'MC-Dropout Mean (calibrated)':<40} {auc_mc:.3f}  {brier_mc:.4f}")
    print(f"  {'StatsBomb xG (reference)':<40} {auc_sb:.3f}  {brier_sb:.4f}")

    print(f"\n  Uncertainty summary:")
    print(f"    Mean σ : {stds.mean():.4f}")
    print(f"    Median σ: {np.median(stds):.4f}")
    print(f"    Max σ  : {stds.max():.4f}")
    print(f"    Mean CV: {cv.mean():.4f}")

    print(f"\n  Brier by uncertainty tercile (σ threshold: "
          f"{terciles[0]:.4f} / {terciles[1]:.4f}):")
    strat_results = {}
    for lab, mask in zip(t_labels, t_masks):
        b_mc = brier_score_loss(y_true[mask], means_cal[mask])
        b_sb = brier_score_loss(y_true[mask], sb_xg[mask])
        strat_results[lab] = {
            "n":        int(mask.sum()),
            "brier_mc": round(float(b_mc), 4),
            "brier_sb": round(float(b_sb), 4),
            "mean_std": round(float(stds[mask].mean()), 4),
        }
        print(f"    {lab:15s}: n={mask.sum():4d}  "
              f"Brier_MC={b_mc:.4f}  Brier_SB={b_sb:.4f}  "
              f"mean_σ={stds[mask].mean():.4f}")

    # Most uncertain shots
    top_k = 10
    top_idx = np.argsort(stds)[::-1][:top_k]
    print(f"\n  Top-{top_k} most uncertain shots:")
    print(f"  {'#':>3}  {'xG_mean':>8}  {'σ':>8}  {'CV':>6}  {'outcome':>7}  "
          f"{'sb_xg':>7}")
    for rank, i in enumerate(top_idx):
        outcome = "GOAL" if y_true[i] == 1 else "miss"
        print(f"  {rank+1:>3}  {means_cal[i]:8.4f}  {stds[i]:8.4f}  "
              f"{cv[i]:6.3f}  {outcome:>7}  {sb_xg[i]:7.4f}")

    # Correlation: σ vs |error|
    abs_err = np.abs(y_true - means_cal)
    corr    = float(np.corrcoef(stds, abs_err)[0, 1])
    print(f"\n  Corr(σ, |error|) = {corr:.3f}  "
          f"({'positive: σ tracks error' if corr > 0 else 'negative'})")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "n_samples":      args.n_samples,
        "n_test_shots":   len(test_g),
        "auc_mc_cal":     round(float(auc_mc),  4),
        "auc_sb":         round(float(auc_sb),  4),
        "brier_mc_cal":   round(float(brier_mc), 4),
        "brier_sb":       round(float(brier_sb), 4),
        "sigma_mean":     round(float(stds.mean()),   4),
        "sigma_median":   round(float(np.median(stds)), 4),
        "sigma_max":      round(float(stds.max()),    4),
        "cv_mean":        round(float(cv.mean()),     4),
        "corr_sigma_abs_error": round(corr, 4),
        "brier_by_sigma_tercile": strat_results,
        "sigma_tercile_thresholds": [round(float(t), 4) for t in terciles],
    }

    out_path = PROCESSED / "mc_dropout_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("\n── Generating figure …")
    fig = make_figure(y_true, means_cal, stds, sb_xg)
    fig_path = ASSETS / "fig_mc_dropout.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
