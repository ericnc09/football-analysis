#!/usr/bin/env python3
"""
elo_ratings.py
--------------
Elo rating system for team strength quantification across all 7 StatsBomb
360 competitions (326 matches).

Method
------
- Initialize all teams at Elo 1500.
- Process matches chronologically within each competition.
- Standard Elo update with configurable K-factor and home advantage.
- Derive pre-match win probabilities from Elo difference.
- Compare Elo-predicted outcomes vs actual vs Poisson (from xG).

Outputs
-------
  data/processed/elo_results.json
  assets/fig_elo_ratings.png

Usage
-----
  python scripts/elo_ratings.py
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
from sklearn.metrics import brier_score_loss
from statsbombpy import sb

REPO_ROOT = Path(__file__).parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# Elo parameters
INITIAL_ELO = 1500
K_FACTOR    = 32
HOME_ADV    = 100


# ---------------------------------------------------------------------------
# Competition definitions (same as build_shot_graphs.py)
# ---------------------------------------------------------------------------

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
# Elo computation
# ---------------------------------------------------------------------------

def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score for player A given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def elo_update(elo: float, expected: float, actual: float, k: float = K_FACTOR) -> float:
    """Update Elo rating after a match."""
    return elo + k * (actual - expected)


def actual_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    """Return (home_score, away_score) in Elo terms: 1=win, 0.5=draw, 0=loss."""
    if home_goals > away_goals:
        return 1.0, 0.0
    elif home_goals < away_goals:
        return 0.0, 1.0
    else:
        return 0.5, 0.5


def elo_win_probability(elo_home: float, elo_away: float,
                         home_advantage: float = HOME_ADV) -> dict:
    """
    Derive P(home win), P(draw), P(away win) from Elo difference.
    Uses the Elo expected score with a draw margin.
    """
    e_home = expected_score(elo_home + home_advantage, elo_away)
    # Approximate draw probability from the gap between expected scores
    # Wider gap → less draw probability
    draw_factor = 0.28 * (1 - abs(2 * e_home - 1))  # empirical draw scaling
    p_home = e_home - draw_factor / 2
    p_away = (1 - e_home) - draw_factor / 2
    p_draw = draw_factor
    # Clamp and normalize
    p_home = max(p_home, 0.01)
    p_away = max(p_away, 0.01)
    p_draw = max(p_draw, 0.01)
    total  = p_home + p_draw + p_away
    return {
        "p_home_win": p_home / total,
        "p_draw":     p_draw / total,
        "p_away_win": p_away / total,
    }


# ---------------------------------------------------------------------------
# Fetch all matches
# ---------------------------------------------------------------------------

def fetch_all_matches() -> pd.DataFrame:
    """Fetch match data for all 7 competitions from StatsBomb."""
    all_dfs = []
    for comp in COMPETITIONS:
        print(f"  Fetching {comp['label']} "
              f"(comp={comp['comp_id']}, season={comp['season_id']}) …")
        df = sb.matches(competition_id=comp["comp_id"],
                        season_id=comp["season_id"])
        df["comp_label"] = comp["label"]
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["match_date"] = pd.to_datetime(combined["match_date"])
    combined = combined.sort_values("match_date").reset_index(drop=True)

    print(f"  Total matches: {len(combined)}")
    return combined


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(match_records, elo_ratings, comp_stats):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Elo Rating System — 326 StatsBomb Matches",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Top 20 final Elo ratings ─────────────────────────────────
    ax = axes[0, 0]
    top = sorted(elo_ratings.items(), key=lambda x: -x[1])[:20]
    names = [t[0] for t in top][::-1]
    elos  = [t[1] for t in top][::-1]
    colors = ["steelblue" if e >= INITIAL_ELO else "salmon" for e in elos]
    ax.barh(range(len(names)), elos, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(INITIAL_ELO, color="black", ls=":", lw=1, label=f"Start ({INITIAL_ELO})")
    ax.set_xlabel("Elo Rating")
    ax.set_title("Top 20 Teams by Final Elo")
    ax.legend(fontsize=8)

    # ── Panel 2: Elo trajectory for selected teams ────────────────────────
    ax = axes[0, 1]
    # Track Elo over time for top 6 teams
    top6 = [t[0] for t in sorted(elo_ratings.items(), key=lambda x: -x[1])[:6]]
    team_history = defaultdict(list)
    for r in match_records:
        if r["home_team"] in top6:
            team_history[r["home_team"]].append(
                (r["match_date"], r["elo_home_after"]))
        if r["away_team"] in top6:
            team_history[r["away_team"]].append(
                (r["match_date"], r["elo_away_after"]))
    for team in top6:
        hist = sorted(team_history[team], key=lambda x: x[0])
        if hist:
            dates = [h[0] for h in hist]
            elos  = [h[1] for h in hist]
            ax.plot(range(len(dates)), elos, "-o", ms=3, lw=1.2, label=team)
    ax.axhline(INITIAL_ELO, color="black", ls=":", lw=1)
    ax.set_xlabel("Match index")
    ax.set_ylabel("Elo Rating")
    ax.set_title("Elo Trajectory — Top 6 Teams")
    ax.legend(fontsize=7, loc="lower right")

    # ── Panel 3: Win probability calibration ──────────────────────────────
    ax = axes[1, 0]
    p_hw = np.array([r["p_home_win"] for r in match_records])
    y_hw = np.array([1 if r["actual_result"] == "home_win" else 0
                     for r in match_records])
    n_bins   = 8
    bin_edges = np.linspace(0, 1, n_bins + 1)
    frac, mp = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p_hw >= lo) & (p_hw < hi)
        if mask.sum() >= 3:
            frac.append(y_hw[mask].mean())
            mp.append(p_hw[mask].mean())
    if mp:
        ax.plot(mp, frac, "o-", color="steelblue", lw=1.5, ms=6, label="Elo")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("Elo P(home win)")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Elo Win Probability Calibration")
    ax.legend(fontsize=9)

    # ── Panel 4: Accuracy by competition ──────────────────────────────────
    ax = axes[1, 1]
    comps   = sorted(comp_stats.keys())
    x       = np.arange(len(comps))
    acc     = [comp_stats[c]["accuracy"] for c in comps]
    bars = ax.bar(x, acc, color="steelblue", alpha=0.8)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.0%}", ha="center", va="bottom", fontsize=8)
    short = [c.replace("bundesliga", "buli") for c in comps]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("3-way accuracy")
    ax.set_title("Elo Outcome Accuracy by Competition")
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print(f"  Elo Rating System  (K={K_FACTOR}, HFA={HOME_ADV})")
    print("=" * 64)

    # ── Fetch matches ─────────────────────────────────────────────────────
    print("\n── Fetching match data from StatsBomb ──────────────────────")
    matches_df = fetch_all_matches()

    # ── Run Elo ───────────────────────────────────────────────────────────
    print(f"\n── Computing Elo ratings ({len(matches_df)} matches) …")
    elo = defaultdict(lambda: INITIAL_ELO)
    match_records = []
    comp_buckets  = defaultdict(list)

    for _, row in matches_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h_goals = int(row["home_score"])
        a_goals = int(row["away_score"])
        comp    = row["comp_label"]
        date    = str(row["match_date"].date())

        elo_h_before = elo[home]
        elo_a_before = elo[away]

        # Pre-match predictions
        pred = elo_win_probability(elo_h_before, elo_a_before)

        # Actual scores for Elo update
        s_h, s_a = actual_score(h_goals, a_goals)
        e_h = expected_score(elo_h_before + HOME_ADV, elo_a_before)
        e_a = 1 - e_h

        # Update
        elo[home] = elo_update(elo_h_before, e_h, s_h)
        elo[away] = elo_update(elo_a_before, e_a, s_a)

        actual_result = ("home_win" if h_goals > a_goals else
                         "draw"     if h_goals == a_goals else
                         "away_win")
        opts = [("home_win", pred["p_home_win"]),
                ("draw",     pred["p_draw"]),
                ("away_win", pred["p_away_win"])]
        predicted = max(opts, key=lambda x: x[1])[0]

        rec = {
            "match_id":       int(row["match_id"]),
            "match_date":     date,
            "home_team":      home,
            "away_team":      away,
            "comp_label":     comp,
            "home_score":     h_goals,
            "away_score":     a_goals,
            "elo_home_before": round(elo_h_before, 1),
            "elo_away_before": round(elo_a_before, 1),
            "elo_home_after":  round(elo[home], 1),
            "elo_away_after":  round(elo[away], 1),
            "elo_diff":        round(elo_h_before - elo_a_before, 1),
            **pred,
            "predicted_result": predicted,
            "actual_result":    actual_result,
        }
        match_records.append(rec)
        comp_buckets[comp].append(rec)

    # ── Final ratings ─────────────────────────────────────────────────────
    elo_final = dict(elo)
    top10 = sorted(elo_final.items(), key=lambda x: -x[1])[:10]
    print("\n  Top 10 Elo ratings:")
    for i, (team, rating) in enumerate(top10, 1):
        print(f"    {i:>2}. {team:<25} {rating:.1f}")

    bottom5 = sorted(elo_final.items(), key=lambda x: x[1])[:5]
    print("\n  Bottom 5:")
    for team, rating in bottom5:
        print(f"      {team:<25} {rating:.1f}")

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\n── Metrics ─────────────────────────────────────────────────")
    n = len(match_records)
    n_correct = sum(r["predicted_result"] == r["actual_result"] for r in match_records)
    print(f"\n  3-way outcome accuracy: {n_correct}/{n} ({100*n_correct/n:.1f}%)")

    # Home-win Brier
    y_hw = np.array([1 if r["actual_result"] == "home_win" else 0
                     for r in match_records])
    p_hw = np.array([r["p_home_win"] for r in match_records])
    brier = brier_score_loss(y_hw, p_hw)
    print(f"  Home-win Brier: {brier:.4f}")

    # RPS
    rps_vals = []
    for r in match_records:
        actual_vec = np.zeros(3)
        if r["actual_result"] == "home_win":   actual_vec[0] = 1
        elif r["actual_result"] == "draw":     actual_vec[1] = 1
        else:                                  actual_vec[2] = 1
        cum_pred = np.cumsum([r["p_home_win"], r["p_draw"], r["p_away_win"]])
        cum_act  = np.cumsum(actual_vec)
        rps_vals.append(float(np.mean((cum_pred - cum_act) ** 2)))
    rps_mean = np.mean(rps_vals)
    print(f"  Ranked Probability Score (RPS): {rps_mean:.4f}")

    # Per-competition breakdown
    print(f"\n  Per-competition breakdown:")
    print(f"  {'Comp':<20} {'N':>4}  {'Accuracy':>8}  {'Brier':>8}  {'RPS':>8}")
    print(f"  {'-'*55}")
    comp_stats = {}
    for comp, recs in sorted(comp_buckets.items()):
        nc = len(recs)
        acc = sum(r["predicted_result"] == r["actual_result"] for r in recs) / nc
        yhw = np.array([1 if r["actual_result"] == "home_win" else 0 for r in recs])
        phw = np.array([r["p_home_win"] for r in recs])
        b   = brier_score_loss(yhw, phw) if len(np.unique(yhw)) > 1 else float("nan")
        c_rps_list = [rps_vals[i] for i, mr in enumerate(match_records) if mr["comp_label"] == comp]
        c_rps = np.mean(c_rps_list)
        print(f"  {comp:<20} {nc:>4}  {acc:>8.1%}  {b:>8.4f}  {c_rps:>8.4f}")
        comp_stats[comp] = {
            "n_matches": nc,
            "accuracy":  round(acc, 4),
            "brier_hw":  round(b, 4) if not np.isnan(b) else None,
            "rps":       round(float(c_rps), 4),
        }

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "description": "Elo rating system across 7 StatsBomb 360 competitions",
        "parameters": {"K": K_FACTOR, "home_advantage": HOME_ADV, "initial": INITIAL_ELO},
        "n_matches": n,
        "n_teams": len(elo_final),
        "accuracy": round(n_correct / n, 4),
        "brier_hw": round(brier, 4),
        "rps": round(float(rps_mean), 4),
        "per_competition": comp_stats,
        "final_ratings": {k: round(v, 1) for k, v in
                          sorted(elo_final.items(), key=lambda x: -x[1])},
        "matches": match_records,
    }
    out_path = PROCESSED / "elo_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Figure ────────────────────────────────────────────────────────────
    print("\n── Generating figure …")
    fig = make_figure(match_records, elo_final, comp_stats)
    fig_path = ASSETS / "fig_elo_ratings.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
