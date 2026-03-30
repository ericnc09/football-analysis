#!/usr/bin/env python3
"""
match_outcome_simulation.py
---------------------------
Monte Carlo match outcome simulation using per-shot HybridGATv2 xG predictions.

Method
------
For each match in the test set:
  1. Retrieve calibrated per-shot xG predictions from HybridGATv2.
  2. Run N_SIM simulations: for every shot, sample Bernoulli(xg) → goal/no-goal.
  3. Count home/away goals per simulation → simulated scoreline distribution.
  4. Aggregate → P(home win), P(draw), P(away win) per match.

Validation
----------
  - Binary outcome accuracy vs actual result
  - Win-probability calibration (Brier + reliability diagram)
  - Compare HybridGATv2 outcome predictions vs StatsBomb xG outcome predictions
  - Per-competition breakdown

Outputs
-------
  data/processed/match_simulation_results.json
  assets/fig_match_simulation.png

Usage
-----
  python scripts/match_outcome_simulation.py
  python scripts/match_outcome_simulation.py --n-sim 50000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.hybrid_gat import HybridGATModel

PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SEED    = 42
N_SIM   = 10_000
META_DIM = 27

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Data helpers
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
# Metadata tensor (mirrors _metadata_tensor in train_xg_hybrid.py)
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


# ---------------------------------------------------------------------------
# Model inference (deterministic, eval mode)
# ---------------------------------------------------------------------------

def get_predictions(
    model: HybridGATModel,
    graphs: list,
    batch_size: int = 64,
) -> np.ndarray:
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


def apply_per_comp_T(
    probs: np.ndarray,
    graphs: list,
    per_T: dict,
    T_global: float,
) -> np.ndarray:
    logits = np.log(probs / (1 - probs + 1e-9) + 1e-9)
    cal    = probs.copy()
    for i, g in enumerate(graphs):
        cl     = getattr(g, "comp_label", "unknown") or "unknown"
        T      = per_T.get(cl, T_global)
        cal[i] = 1 / (1 + np.exp(-logits[i] / T))
    return cal


# ---------------------------------------------------------------------------
# Match grouping
# ---------------------------------------------------------------------------

def build_match_index(graphs: list, probs: np.ndarray, sb_probs: np.ndarray) -> dict:
    """
    Returns {match_id: {home_team, away_team, comp_label, shots, probs, sb_probs, y}}.
    away_team = the other team in the match.
    """
    matches: dict[int, dict] = {}
    for i, g in enumerate(graphs):
        mid = g.match_id.item()
        if mid not in matches:
            matches[mid] = {
                "home_team":  g.home_team,
                "away_team":  "",       # filled below
                "comp_label": getattr(g, "comp_label", "unknown"),
                "shots":      [],
                "probs":      [],
                "sb_probs":   [],
                "y":          [],
            }
        matches[mid]["shots"].append(g)
        matches[mid]["probs"].append(float(probs[i]))
        matches[mid]["sb_probs"].append(float(sb_probs[i]))
        matches[mid]["y"].append(int(g.y.item()))

    # Fill away_team
    for mid, m in matches.items():
        teams = set(g.team_name for g in m["shots"])
        m["away_team"] = next(
            (t for t in teams if t != m["home_team"]), m["home_team"]
        )

    return matches


# ---------------------------------------------------------------------------
# Monte Carlo match outcome simulation
# ---------------------------------------------------------------------------

def simulate_match(
    home_xg: np.ndarray,
    away_xg: np.ndarray,
    n_sim: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns simulated home_goals and away_goals arrays of shape (n_sim,).
    Each shot is an independent Bernoulli draw.
    """
    home_goals = rng.binomial(1, home_xg[None, :], size=(n_sim, len(home_xg))).sum(axis=1)
    away_goals = rng.binomial(1, away_xg[None, :], size=(n_sim, len(away_xg))).sum(axis=1)
    return home_goals, away_goals


def outcome_probs(home_goals: np.ndarray, away_goals: np.ndarray) -> dict:
    p_home = float((home_goals > away_goals).mean())
    p_draw = float((home_goals == away_goals).mean())
    p_away = float((home_goals < away_goals).mean())
    exp_home = float(home_goals.mean())
    exp_away = float(away_goals.mean())
    return {
        "p_home_win": p_home,
        "p_draw":     p_draw,
        "p_away_win": p_away,
        "exp_home_goals": exp_home,
        "exp_away_goals": exp_away,
    }


def actual_outcome(match: dict) -> dict:
    """Compute actual home/away goals from the shot labels."""
    shots = match["shots"]
    y     = match["y"]
    home  = match["home_team"]

    home_goals = sum(yy for g, yy in zip(shots, y) if g.team_name == home)
    away_goals = sum(yy for g, yy in zip(shots, y) if g.team_name != home)
    result = "home_win" if home_goals > away_goals else (
             "draw"     if home_goals == away_goals else "away_win")
    return {
        "actual_home_goals": home_goals,
        "actual_away_goals": away_goals,
        "actual_result":     result,
    }


# ---------------------------------------------------------------------------
# Accuracy + calibration helpers
# ---------------------------------------------------------------------------

def pick_predicted_result(p: dict) -> str:
    """Argmax of (p_home_win, p_draw, p_away_win)."""
    opts = [("home_win", p["p_home_win"]),
            ("draw",     p["p_draw"]),
            ("away_win", p["p_away_win"])]
    return max(opts, key=lambda x: x[1])[0]


def win_brier(y_win: np.ndarray, p_win: np.ndarray) -> float:
    return float(brier_score_loss(y_win, p_win))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    match_records: list[dict],
    comp_stats:    dict,
) -> plt.Figure:
    """
    4-panel match outcome simulation figure.

    Panel 1: Calibration of P(home win) — MC sim vs StatsBomb
    Panel 2: Predicted vs actual goal totals (scatter)
    Panel 3: Outcome accuracy by competition
    Panel 4: xG-implied win% vs actual win rate (binned)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Match Outcome Simulation — HybridGATv2 vs StatsBomb xG",
        fontsize=13, fontweight="bold",
    )

    p_hw_mc  = np.array([r["p_home_win"]    for r in match_records])
    p_hw_sb  = np.array([r["sb_p_home_win"] for r in match_records])
    y_hw     = np.array([1 if r["actual_result"] == "home_win" else 0
                         for r in match_records])
    y_draw   = np.array([1 if r["actual_result"] == "draw" else 0
                         for r in match_records])

    # ── Panel 1: Win-probability calibration ──────────────────────────────────
    ax = axes[0, 0]
    n_bins   = 8
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for p_pred, label, color in [
        (p_hw_mc, "MC-Sim (HybridGAT)", "steelblue"),
        (p_hw_sb, "StatsBomb xG",       "tomato"),
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
    ax.set_title("Win Probability Calibration")
    ax.legend(fontsize=9)

    # ── Panel 2: Predicted vs actual total goals (per match) ──────────────────
    ax = axes[0, 1]
    exp_tot_mc = np.array([r["exp_home_goals"] + r["exp_away_goals"]
                            for r in match_records])
    exp_tot_sb = np.array([r["sb_exp_home"] + r["sb_exp_away"]
                            for r in match_records])
    act_tot    = np.array([r["actual_home_goals"] + r["actual_away_goals"]
                            for r in match_records])

    jitter = np.random.RandomState(SEED).uniform(-0.12, 0.12, len(act_tot))
    ax.scatter(act_tot + jitter, exp_tot_mc, s=18, alpha=0.6,
               color="steelblue", label="MC-Sim", zorder=3)
    ax.scatter(act_tot + jitter * 0.5, exp_tot_sb, s=18, alpha=0.5,
               color="tomato", label="StatsBomb", zorder=2, marker="^")
    mn = min(act_tot.min(), exp_tot_mc.min(), exp_tot_sb.min()) - 0.3
    mx = max(act_tot.max(), exp_tot_mc.max(), exp_tot_sb.max()) + 0.3
    ax.plot([mn, mx], [mn, mx], "k:", lw=1)
    ax.set_xlabel("Actual total goals")
    ax.set_ylabel("Expected total goals (xG sum)")
    ax.set_title("Predicted vs Actual Total Goals per Match")
    ax.legend(fontsize=9)

    # ── Panel 3: Outcome accuracy by competition ───────────────────────────────
    ax = axes[1, 0]
    comps   = sorted(comp_stats.keys())
    n_comps = len(comps)
    x       = np.arange(n_comps)
    w       = 0.35
    acc_mc  = [comp_stats[c]["accuracy_mc"]  for c in comps]
    acc_sb  = [comp_stats[c]["accuracy_sb"]  for c in comps]
    b1 = ax.bar(x - w/2, acc_mc, w, color="steelblue", alpha=0.8, label="MC-Sim")
    b2 = ax.bar(x + w/2, acc_sb, w, color="tomato",    alpha=0.8, label="StatsBomb")
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    short_labels = [c.replace("bundesliga", "buli") for c in comps]
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Outcome accuracy (argmax)")
    ax.set_title("3-Way Outcome Accuracy by Competition")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)

    # ── Panel 4: Goal-count distribution (simulated vs actual) ────────────────
    ax = axes[1, 1]
    act_home = np.array([r["actual_home_goals"] for r in match_records])
    act_away = np.array([r["actual_away_goals"] for r in match_records])
    act_all  = np.concatenate([act_home, act_away])

    sim_home_mean = np.array([r["exp_home_goals"] for r in match_records])
    sim_away_mean = np.array([r["exp_away_goals"] for r in match_records])
    sim_all  = np.concatenate([sim_home_mean, sim_away_mean])
    sb_all   = np.concatenate([
        np.array([r["sb_exp_home"] for r in match_records]),
        np.array([r["sb_exp_away"] for r in match_records]),
    ])

    max_g = int(act_all.max()) + 1
    bins  = np.arange(-0.5, max_g + 0.5, 1)
    ax.hist(act_all,  bins=bins, density=True, alpha=0.5,
            color="black",     label="Actual",     edgecolor="white", lw=0.3)
    ax.hist(sim_all,  bins=np.linspace(0, max_g, 30), density=True, alpha=0.55,
            color="steelblue", label="MC-Sim (mean xG/match)", histtype="stepfilled")
    ax.hist(sb_all,   bins=np.linspace(0, max_g, 30), density=True, alpha=0.45,
            color="tomato",    label="StatsBomb xG/match",     histtype="step", lw=2)
    ax.set_xlabel("Goals per team per match")
    ax.set_ylabel("Density")
    ax.set_title("Goal-Count Distribution: Actual vs xG Models")
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sim", type=int, default=N_SIM,
                   help=f"MC simulations per match (default: {N_SIM})")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print(f"  Match Outcome Simulation  (N={args.n_sim:,} sims/match)")
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

    g0    = test_g[0]
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
    print(f"  T_global={T_global:.4f}  per_T comps: {list(per_T.keys())}")

    # ── Group shots by match ───────────────────────────────────────────────
    match_index = build_match_index(test_g, probs_cal, sb_xg)
    print(f"  Unique matches in test set: {len(match_index)}")

    # ── Simulate outcomes ──────────────────────────────────────────────────
    print(f"\n── Simulating {args.n_sim:,} outcomes per match …")
    rng = np.random.RandomState(SEED)

    match_records = []
    comp_buckets: dict[str, list] = defaultdict(list)

    for mid, m in sorted(match_index.items()):
        home  = m["home_team"]
        shots = m["shots"]

        home_idx = [i for i, g in enumerate(shots) if g.team_name == home]
        away_idx = [i for i, g in enumerate(shots) if g.team_name != home]

        home_xg_mc = np.array([m["probs"][i]    for i in home_idx]) if home_idx else np.array([0.0])
        away_xg_mc = np.array([m["probs"][i]    for i in away_idx]) if away_idx else np.array([0.0])
        home_xg_sb = np.array([m["sb_probs"][i] for i in home_idx]) if home_idx else np.array([0.0])
        away_xg_sb = np.array([m["sb_probs"][i] for i in away_idx]) if away_idx else np.array([0.0])

        # MC simulation
        hg_mc, ag_mc = simulate_match(home_xg_mc, away_xg_mc, args.n_sim, rng)
        hg_sb, ag_sb = simulate_match(home_xg_sb, away_xg_sb, args.n_sim, rng)

        out_mc = outcome_probs(hg_mc, ag_mc)
        out_sb = outcome_probs(hg_sb, ag_sb)
        actual = actual_outcome(m)

        rec = {
            "match_id":          mid,
            "home_team":         home,
            "away_team":         m["away_team"],
            "comp_label":        m["comp_label"],
            **{f"{k}": v for k, v in out_mc.items()},
            "sb_p_home_win":     out_sb["p_home_win"],
            "sb_p_draw":         out_sb["p_draw"],
            "sb_p_away_win":     out_sb["p_away_win"],
            "sb_exp_home":       out_sb["exp_home_goals"],
            "sb_exp_away":       out_sb["exp_away_goals"],
            "predicted_result":  pick_predicted_result(out_mc),
            "sb_predicted_result": pick_predicted_result(out_sb),
            **actual,
        }
        match_records.append(rec)
        comp_buckets[m["comp_label"]].append(rec)

    # ── Metrics ────────────────────────────────────────────────────────────
    print("\n── Metrics ─────────────────────────────────────────────────")

    # 3-way accuracy
    n_mc = sum(r["predicted_result"] == r["actual_result"] for r in match_records)
    n_sb = sum(r["sb_predicted_result"] == r["actual_result"] for r in match_records)
    n    = len(match_records)
    print(f"\n  3-way outcome accuracy (argmax):")
    print(f"    MC-Sim (HybridGAT) : {n_mc}/{n}  ({100*n_mc/n:.1f}%)")
    print(f"    StatsBomb xG       : {n_sb}/{n}  ({100*n_sb/n:.1f}%)")

    # Home-win Brier (binary)
    y_hw     = np.array([1 if r["actual_result"] == "home_win" else 0
                         for r in match_records])
    p_hw_mc  = np.array([r["p_home_win"]    for r in match_records])
    p_hw_sb  = np.array([r["sb_p_home_win"] for r in match_records])
    b_hw_mc  = brier_score_loss(y_hw, p_hw_mc)
    b_hw_sb  = brier_score_loss(y_hw, p_hw_sb)
    print(f"\n  Home-win probability Brier score:")
    print(f"    MC-Sim (HybridGAT) : {b_hw_mc:.4f}")
    print(f"    StatsBomb xG       : {b_hw_sb:.4f}")

    # Expected goals MAE
    act_home = np.array([r["actual_home_goals"] for r in match_records])
    act_away = np.array([r["actual_away_goals"] for r in match_records])
    exp_home_mc = np.array([r["exp_home_goals"] for r in match_records])
    exp_away_mc = np.array([r["exp_away_goals"] for r in match_records])
    exp_home_sb = np.array([r["sb_exp_home"]    for r in match_records])
    exp_away_sb = np.array([r["sb_exp_away"]    for r in match_records])
    mae_mc = float(np.abs(np.concatenate([act_home - exp_home_mc, act_away - exp_away_mc])).mean())
    mae_sb = float(np.abs(np.concatenate([act_home - exp_home_sb, act_away - exp_away_sb])).mean())
    print(f"\n  Team xG MAE vs actual goals:")
    print(f"    MC-Sim (HybridGAT) : {mae_mc:.3f}")
    print(f"    StatsBomb xG       : {mae_sb:.3f}")

    # Per-competition breakdown
    print(f"\n  Per-competition breakdown:")
    print(f"  {'Comp':<20} {'Matches':>7}  {'Acc_MC':>7}  {'Acc_SB':>7}  "
          f"{'Brier_MC':>8}  {'Brier_SB':>8}")
    print(f"  {'-'*66}")
    comp_stats = {}
    for comp, recs in sorted(comp_buckets.items()):
        nc   = len(recs)
        a_mc = sum(r["predicted_result"] == r["actual_result"] for r in recs) / nc
        a_sb = sum(r["sb_predicted_result"] == r["actual_result"] for r in recs) / nc
        yhw  = np.array([1 if r["actual_result"] == "home_win" else 0 for r in recs])
        phw_mc = np.array([r["p_home_win"]    for r in recs])
        phw_sb = np.array([r["sb_p_home_win"] for r in recs])
        bmc = brier_score_loss(yhw, phw_mc) if len(np.unique(yhw)) > 1 else float("nan")
        bsb = brier_score_loss(yhw, phw_sb) if len(np.unique(yhw)) > 1 else float("nan")
        print(f"  {comp:<20} {nc:>7}  {a_mc:>7.3f}  {a_sb:>7.3f}  "
              f"{bmc:>8.4f}  {bsb:>8.4f}")
        comp_stats[comp] = {
            "n_matches":   nc,
            "accuracy_mc": round(a_mc, 4),
            "accuracy_sb": round(a_sb, 4),
            "brier_hw_mc": round(bmc, 4) if not np.isnan(bmc) else None,
            "brier_hw_sb": round(bsb, 4) if not np.isnan(bsb) else None,
        }

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "n_sim":             args.n_sim,
        "n_matches":         len(match_records),
        "accuracy_mc":       round(n_mc / n, 4),
        "accuracy_sb":       round(n_sb / n, 4),
        "brier_hw_mc":       round(b_hw_mc, 4),
        "brier_hw_sb":       round(b_hw_sb, 4),
        "mae_xg_mc":         round(mae_mc, 3),
        "mae_xg_sb":         round(mae_sb, 3),
        "per_competition":   comp_stats,
        "matches":           [
            {k: v for k, v in r.items() if k != "shots"}
            for r in match_records
        ],
    }
    out_path = PROCESSED / "match_simulation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Figure ─────────────────────────────────────────────────────────────
    print("\n── Generating figure …")
    fig = make_figure(match_records, comp_stats)
    fig_path = ASSETS / "fig_match_simulation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
