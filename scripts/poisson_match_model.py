#!/usr/bin/env python3
"""
poisson_match_model.py
----------------------
Poisson distribution model for match scoreline prediction.

The industry-standard analytical approach: model team goals as independent
Poisson processes with λ = team xG sum, then derive scoreline probabilities
and match outcome probabilities.

Compares Poisson-derived outcomes against:
  - Monte Carlo Bernoulli simulation (match_outcome_simulation.py)
  - StatsBomb xG-based Poisson
  - Actual match results

Outputs
-------
  data/processed/poisson_match_results.json
  assets/fig_poisson_model.png

Usage
-----
  python scripts/poisson_match_model.py
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
from scipy.stats import poisson
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import brier_score_loss

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.hybrid_gat import HybridGATModel

PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SEED     = 42
META_DIM = 27
MAX_GOALS = 8  # max goals per team in scoreline matrix

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading — identical split
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
# Model inference (reused from match_outcome_simulation.py)
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
    return torch.cat([base, tech, gk, new, plc], dim=1)


def get_predictions(model, graphs, batch_size=64):
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            meta  = build_meta(batch)
            ea    = batch.edge_attr if batch.edge_attr is not None else None
            logit = model(batch.x, batch.edge_index, batch.batch, meta, edge_attr=ea)
            probs_all.append(torch.sigmoid(logit).squeeze().cpu())
    return torch.cat(probs_all).numpy()


def load_per_comp_T() -> dict:
    p = PROCESSED / "pool_7comp_per_comp_T_gat.pt"
    return torch.load(p, weights_only=False) if p.exists() else {}


def apply_per_comp_T(probs, graphs, per_T, T_global):
    logits = np.log(probs / (1 - probs + 1e-9) + 1e-9)
    cal    = probs.copy()
    for i, g in enumerate(graphs):
        cl    = getattr(g, "comp_label", "unknown") or "unknown"
        T     = per_T.get(cl, T_global)
        cal[i] = 1 / (1 + np.exp(-logits[i] / T))
    return cal


# ---------------------------------------------------------------------------
# Poisson model core
# ---------------------------------------------------------------------------

def poisson_scoreline_matrix(lambda_home: float, lambda_away: float,
                              max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Compute P(home=h, away=a) for h,a in [0..max_goals].
    Assumes goals are independent Poisson processes.
    Returns (max_goals+1, max_goals+1) matrix.
    """
    h_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    a_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    return np.outer(h_probs, a_probs)


def poisson_outcome_probs(matrix: np.ndarray) -> dict:
    """Derive P(home win), P(draw), P(away win) from scoreline matrix."""
    n = matrix.shape[0]
    p_home = sum(matrix[h, a] for h in range(n) for a in range(n) if h > a)
    p_draw = sum(matrix[h, h] for h in range(n))
    p_away = sum(matrix[h, a] for h in range(n) for a in range(n) if h < a)
    return {
        "p_home_win": float(p_home),
        "p_draw":     float(p_draw),
        "p_away_win": float(p_away),
    }


def most_likely_scoreline(matrix: np.ndarray) -> tuple[int, int, float]:
    """Return (home_goals, away_goals, probability) of the most likely scoreline."""
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(idx[0]), int(idx[1]), float(matrix[idx])


# ---------------------------------------------------------------------------
# Match grouping
# ---------------------------------------------------------------------------

def build_match_index(graphs, probs, sb_probs):
    matches = {}
    for i, g in enumerate(graphs):
        mid = g.match_id.item()
        if mid not in matches:
            matches[mid] = {
                "home_team":  g.home_team,
                "away_team":  "",
                "comp_label": getattr(g, "comp_label", "unknown"),
                "shots": [], "probs": [], "sb_probs": [], "y": [],
            }
        matches[mid]["shots"].append(g)
        matches[mid]["probs"].append(float(probs[i]))
        matches[mid]["sb_probs"].append(float(sb_probs[i]))
        matches[mid]["y"].append(int(g.y.item()))
    for mid, m in matches.items():
        teams = set(g.team_name for g in m["shots"])
        m["away_team"] = next((t for t in teams if t != m["home_team"]), m["home_team"])
    return matches


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(match_records, comp_stats):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Poisson Match Outcome Model — HybridGATv2 xG",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Scoreline heatmap (aggregated across all matches) ────────
    ax = axes[0, 0]
    n = MAX_GOALS + 1
    agg_matrix = np.zeros((n, n))
    for r in match_records:
        m = poisson_scoreline_matrix(r["lambda_home"], r["lambda_away"])
        agg_matrix += m
    agg_matrix /= len(match_records)

    im = ax.imshow(agg_matrix, origin="lower", cmap="YlOrRd", aspect="equal")
    for h in range(min(6, n)):
        for a in range(min(6, n)):
            val = agg_matrix[h, a]
            if val > 0.005:
                ax.text(a, h, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if val > agg_matrix.max() * 0.6 else "black")
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title("Average Scoreline Probability (Poisson)")
    ax.set_xticks(range(min(6, n)))
    ax.set_yticks(range(min(6, n)))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel 2: Poisson vs MC-sim outcome probabilities ──────────────────
    ax = axes[0, 1]
    mc_path = PROCESSED / "match_simulation_results.json"
    if mc_path.exists():
        mc_data = json.load(open(mc_path))
        mc_by_id = {m["match_id"]: m for m in mc_data.get("matches", [])}
        poisson_hw, mc_hw = [], []
        for r in match_records:
            mc_rec = mc_by_id.get(r["match_id"])
            if mc_rec:
                poisson_hw.append(r["p_home_win"])
                mc_hw.append(mc_rec["p_home_win"])
        if poisson_hw:
            ax.scatter(mc_hw, poisson_hw, s=12, alpha=0.6, color="steelblue")
            ax.plot([0, 1], [0, 1], "k:", lw=1)
            corr = np.corrcoef(mc_hw, poisson_hw)[0, 1]
            ax.text(0.05, 0.9, f"r = {corr:.4f}", transform=ax.transAxes,
                    fontsize=10, va="top")
    ax.set_xlabel("MC Simulation P(home win)")
    ax.set_ylabel("Poisson P(home win)")
    ax.set_title("Poisson vs Monte Carlo: P(home win)")

    # ── Panel 3: Win probability calibration ──────────────────────────────
    ax = axes[1, 0]
    p_hw_pois = np.array([r["p_home_win"] for r in match_records])
    p_hw_sb   = np.array([r["sb_p_home_win"] for r in match_records])
    y_hw = np.array([1 if r["actual_result"] == "home_win" else 0
                     for r in match_records])

    n_bins   = 8
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for p_pred, label, color in [
        (p_hw_pois, "Poisson (HybridGAT)", "steelblue"),
        (p_hw_sb,   "Poisson (StatsBomb)",  "tomato"),
    ]:
        frac, mp = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p_pred >= lo) & (p_pred < hi)
            if mask.sum() >= 3:
                frac.append(y_hw[mask].mean())
                mp.append(p_pred[mask].mean())
        if mp:
            ax.plot(mp, frac, "o-", color=color, lw=1.5, ms=6, label=label)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Poisson Win Probability Calibration")
    ax.legend(fontsize=9)

    # ── Panel 4: Accuracy by competition ──────────────────────────────────
    ax = axes[1, 1]
    comps   = sorted(comp_stats.keys())
    n_comps = len(comps)
    x       = np.arange(n_comps)
    w       = 0.35
    acc_pois = [comp_stats[c]["accuracy_poisson"] for c in comps]
    acc_sb   = [comp_stats[c]["accuracy_sb"]      for c in comps]
    b1 = ax.bar(x - w/2, acc_pois, w, color="steelblue", alpha=0.8, label="Poisson (HybridGAT)")
    b2 = ax.bar(x + w/2, acc_sb,   w, color="tomato",    alpha=0.8, label="Poisson (StatsBomb)")
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    short = [c.replace("bundesliga", "buli") for c in comps]
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("3-way accuracy")
    ax.set_title("Poisson Outcome Accuracy by Competition")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Poisson Match Outcome Model")
    print("=" * 64)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n── Loading graphs ──────────────────────────────────────────")
    graphs = load_graphs()
    _, _, test_g = stratified_split(graphs)
    print(f"  Test set: {len(test_g)} shots")

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt = PROCESSED / "pool_7comp_hybrid_gat_xg.pt"
    if not ckpt.exists():
        print(f"ERROR: {ckpt} not found"); sys.exit(1)

    g0     = test_g[0]
    in_ch  = g0.x.shape[1]
    edge_ch = g0.edge_attr.shape[1] if g0.edge_attr is not None else 4

    model = HybridGATModel(
        node_in=in_ch, edge_dim=edge_ch,
        meta_dim=META_DIM, hidden=32, heads=4, n_layers=3, dropout=0.3,
    )
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    print(f"  Loaded: {ckpt.name}")

    # ── Get per-shot xG predictions ────────────────────────────────────────
    print("\n── Inferring per-shot xG …")
    probs_raw = get_predictions(model, test_g)

    per_T = load_per_comp_T()
    T_global_path = PROCESSED / "pool_7comp_gat_T.pt"
    T_global = 1.0
    if T_global_path.exists():
        d = torch.load(T_global_path, weights_only=False)
        T_global = float(d["T"]) if isinstance(d, dict) else float(d)
    probs_cal = apply_per_comp_T(probs_raw, test_g, per_T, T_global)

    sb_xg = np.array([g.sb_xg.item() for g in test_g])

    # ── Group shots by match ───────────────────────────────────────────────
    match_index = build_match_index(test_g, probs_cal, sb_xg)
    print(f"  Unique matches: {len(match_index)}")

    # ── Poisson scoreline computation ─────────────────────────────────────
    print(f"\n── Computing Poisson scoreline probabilities …")

    match_records = []
    comp_buckets = defaultdict(list)

    for mid, m in sorted(match_index.items()):
        home  = m["home_team"]
        shots = m["shots"]

        home_idx = [i for i, g in enumerate(shots) if g.team_name == home]
        away_idx = [i for i, g in enumerate(shots) if g.team_name != home]

        # λ = sum of per-shot xG for each team
        lambda_home    = sum(m["probs"][i]    for i in home_idx) if home_idx else 0.01
        lambda_away    = sum(m["probs"][i]    for i in away_idx) if away_idx else 0.01
        lambda_home_sb = sum(m["sb_probs"][i] for i in home_idx) if home_idx else 0.01
        lambda_away_sb = sum(m["sb_probs"][i] for i in away_idx) if away_idx else 0.01

        # Poisson scoreline matrices
        matrix    = poisson_scoreline_matrix(lambda_home,    lambda_away)
        matrix_sb = poisson_scoreline_matrix(lambda_home_sb, lambda_away_sb)

        out_pois    = poisson_outcome_probs(matrix)
        out_pois_sb = poisson_outcome_probs(matrix_sb)

        ml_h, ml_a, ml_p = most_likely_scoreline(matrix)

        # Actual result
        actual_home = sum(m["y"][i] for i in home_idx)
        actual_away = sum(m["y"][i] for i in away_idx)
        actual_result = ("home_win" if actual_home > actual_away else
                         "draw"     if actual_home == actual_away else
                         "away_win")

        # Predicted result (argmax)
        opts = [("home_win", out_pois["p_home_win"]),
                ("draw",     out_pois["p_draw"]),
                ("away_win", out_pois["p_away_win"])]
        pred_result = max(opts, key=lambda x: x[1])[0]

        opts_sb = [("home_win", out_pois_sb["p_home_win"]),
                   ("draw",     out_pois_sb["p_draw"]),
                   ("away_win", out_pois_sb["p_away_win"])]
        pred_result_sb = max(opts_sb, key=lambda x: x[1])[0]

        rec = {
            "match_id":         mid,
            "home_team":        home,
            "away_team":        m["away_team"],
            "comp_label":       m["comp_label"],
            "lambda_home":      round(lambda_home, 4),
            "lambda_away":      round(lambda_away, 4),
            "lambda_home_sb":   round(lambda_home_sb, 4),
            "lambda_away_sb":   round(lambda_away_sb, 4),
            **out_pois,
            "sb_p_home_win":    out_pois_sb["p_home_win"],
            "sb_p_draw":        out_pois_sb["p_draw"],
            "sb_p_away_win":    out_pois_sb["p_away_win"],
            "most_likely_score": f"{ml_h}-{ml_a}",
            "most_likely_prob":  round(ml_p, 4),
            "predicted_result":    pred_result,
            "sb_predicted_result": pred_result_sb,
            "actual_home_goals":   actual_home,
            "actual_away_goals":   actual_away,
            "actual_result":       actual_result,
        }
        match_records.append(rec)
        comp_buckets[m["comp_label"]].append(rec)

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\n── Metrics ─────────────────────────────────────────────────")
    n = len(match_records)

    # 3-way accuracy
    n_pois = sum(r["predicted_result"] == r["actual_result"] for r in match_records)
    n_sb   = sum(r["sb_predicted_result"] == r["actual_result"] for r in match_records)
    print(f"\n  3-way outcome accuracy:")
    print(f"    Poisson (HybridGAT) : {n_pois}/{n}  ({100*n_pois/n:.1f}%)")
    print(f"    Poisson (StatsBomb) : {n_sb}/{n}  ({100*n_sb/n:.1f}%)")

    # Home-win Brier
    y_hw     = np.array([1 if r["actual_result"] == "home_win" else 0
                         for r in match_records])
    p_hw_pois = np.array([r["p_home_win"]    for r in match_records])
    p_hw_sb   = np.array([r["sb_p_home_win"] for r in match_records])
    b_pois = brier_score_loss(y_hw, p_hw_pois)
    b_sb   = brier_score_loss(y_hw, p_hw_sb)
    print(f"\n  Home-win Brier:")
    print(f"    Poisson (HybridGAT) : {b_pois:.4f}")
    print(f"    Poisson (StatsBomb) : {b_sb:.4f}")

    # λ MAE
    act_h = np.array([r["actual_home_goals"] for r in match_records])
    act_a = np.array([r["actual_away_goals"] for r in match_records])
    lam_h = np.array([r["lambda_home"] for r in match_records])
    lam_a = np.array([r["lambda_away"] for r in match_records])
    lam_h_sb = np.array([r["lambda_home_sb"] for r in match_records])
    lam_a_sb = np.array([r["lambda_away_sb"] for r in match_records])
    mae_pois = float(np.abs(np.concatenate([act_h - lam_h, act_a - lam_a])).mean())
    mae_sb   = float(np.abs(np.concatenate([act_h - lam_h_sb, act_a - lam_a_sb])).mean())
    print(f"\n  Team xG (λ) MAE vs actual goals:")
    print(f"    Poisson (HybridGAT) : {mae_pois:.3f}")
    print(f"    Poisson (StatsBomb) : {mae_sb:.3f}")

    # Most likely scoreline accuracy
    ml_correct = sum(
        1 for r in match_records
        if r["most_likely_score"] == f"{r['actual_home_goals']}-{r['actual_away_goals']}"
    )
    print(f"\n  Most likely scoreline accuracy: {ml_correct}/{n} ({100*ml_correct/n:.1f}%)")

    # Ranked Probability Score (RPS)
    rps_vals = []
    rps_sb_vals = []
    for r in match_records:
        actual_vec = np.zeros(3)
        if r["actual_result"] == "home_win":   actual_vec[0] = 1
        elif r["actual_result"] == "draw":     actual_vec[1] = 1
        else:                                  actual_vec[2] = 1
        cum_pred = np.cumsum([r["p_home_win"], r["p_draw"], r["p_away_win"]])
        cum_act  = np.cumsum(actual_vec)
        rps_vals.append(float(np.mean((cum_pred - cum_act) ** 2)))
        cum_sb = np.cumsum([r["sb_p_home_win"], r["sb_p_draw"], r["sb_p_away_win"]])
        rps_sb_vals.append(float(np.mean((cum_sb - cum_act) ** 2)))
    rps_mean = np.mean(rps_vals)
    rps_sb_mean = np.mean(rps_sb_vals)
    print(f"\n  Ranked Probability Score (RPS, lower is better):")
    print(f"    Poisson (HybridGAT) : {rps_mean:.4f}")
    print(f"    Poisson (StatsBomb) : {rps_sb_mean:.4f}")

    # Per-competition breakdown
    print(f"\n  Per-competition breakdown:")
    print(f"  {'Comp':<20} {'N':>4}  {'Acc_Pois':>8}  {'Acc_SB':>8}  "
          f"{'Brier_P':>8}  {'Brier_SB':>8}  {'RPS_P':>8}  {'RPS_SB':>8}")
    print(f"  {'-'*80}")
    comp_stats = {}
    for comp, recs in sorted(comp_buckets.items()):
        nc = len(recs)
        a_p  = sum(r["predicted_result"] == r["actual_result"] for r in recs) / nc
        a_sb = sum(r["sb_predicted_result"] == r["actual_result"] for r in recs) / nc
        yhw  = np.array([1 if r["actual_result"] == "home_win" else 0 for r in recs])
        phw_p  = np.array([r["p_home_win"]    for r in recs])
        phw_sb = np.array([r["sb_p_home_win"] for r in recs])
        bp  = brier_score_loss(yhw, phw_p)  if len(np.unique(yhw)) > 1 else float("nan")
        bsb = brier_score_loss(yhw, phw_sb) if len(np.unique(yhw)) > 1 else float("nan")
        c_rps = np.mean([rps_vals[i] for i, r in enumerate(match_records) if r["comp_label"] == comp])
        c_rps_sb = np.mean([rps_sb_vals[i] for i, r in enumerate(match_records) if r["comp_label"] == comp])
        print(f"  {comp:<20} {nc:>4}  {a_p:>8.1%}  {a_sb:>8.1%}  "
              f"{bp:>8.4f}  {bsb:>8.4f}  {c_rps:>8.4f}  {c_rps_sb:>8.4f}")
        comp_stats[comp] = {
            "n_matches": nc,
            "accuracy_poisson": round(a_p, 4),
            "accuracy_sb": round(a_sb, 4),
            "brier_hw_poisson": round(bp, 4) if not np.isnan(bp) else None,
            "brier_hw_sb": round(bsb, 4) if not np.isnan(bsb) else None,
            "rps_poisson": round(float(c_rps), 4),
            "rps_sb": round(float(c_rps_sb), 4),
        }

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "description": "Poisson scoreline model from team xG sums",
        "n_matches": n,
        "accuracy_poisson": round(n_pois / n, 4),
        "accuracy_sb": round(n_sb / n, 4),
        "brier_hw_poisson": round(b_pois, 4),
        "brier_hw_sb": round(b_sb, 4),
        "mae_poisson": round(mae_pois, 3),
        "mae_sb": round(mae_sb, 3),
        "rps_poisson": round(float(rps_mean), 4),
        "rps_sb": round(float(rps_sb_mean), 4),
        "scoreline_accuracy": round(ml_correct / n, 4),
        "per_competition": comp_stats,
        "matches": [
            {k: v for k, v in r.items()}
            for r in match_records
        ],
    }
    out_path = PROCESSED / "poisson_match_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Figure ────────────────────────────────────────────────────────────
    print("\n── Generating figure …")
    fig = make_figure(match_records, comp_stats)
    fig_path = ASSETS / "fig_poisson_model.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
