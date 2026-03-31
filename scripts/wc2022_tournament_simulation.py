#!/usr/bin/env python3
"""
wc2022_tournament_simulation.py
--------------------------------
Full FIFA World Cup 2022 tournament simulation using HybridGATv2 per-shot xG.

Runs all 64 matches in correct bracket order (group stage → R16 → QF → SF → Final)
across N independent simulations to produce P(champion), P(advance) per team
and team-level validation metrics for the research paper.

Historical matches use real per-shot freeze-frame xG from our model.
Counterfactual knockout matches (when a different team advanced) use that
team's per-shot average xG drawn as Bernoulli across their typical shot volume.

Usage
-----
  python scripts/wc2022_tournament_simulation.py
  python scripts/wc2022_tournament_simulation.py --n-sim 100   # sanity check
  python scripts/wc2022_tournament_simulation.py --ko-draws extra_time
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.hybrid_gat import HybridGATModel

PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SEED      = 42
META_DIM  = 27
COMP_ID   = 43
SEASON_ID = 106

# WC2022 R16 bracket template: (group_of_home, rank, group_of_away, rank)
# rank 0 = group winner, rank 1 = group runner-up
R16_BRACKET = [
    ("A", 0, "B", 1),  # slot 0: 1A vs 2B  (Netherlands vs USA)
    ("C", 0, "D", 1),  # slot 1: 1C vs 2D  (Argentina vs Australia)
    ("B", 0, "A", 1),  # slot 2: 1B vs 2A  (England vs Senegal)
    ("D", 0, "C", 1),  # slot 3: 1D vs 2C  (France vs Poland)
    ("E", 0, "F", 1),  # slot 4: 1E vs 2F  (Japan vs Croatia)
    ("G", 0, "H", 1),  # slot 5: 1G vs 2H  (Brazil vs South Korea)
    ("F", 0, "E", 1),  # slot 6: 1F vs 2E  (Morocco vs Spain)
    ("H", 0, "G", 1),  # slot 7: 1H vs 2G  (Portugal vs Switzerland)
]
# QF: pairs of R16 slot indices that meet
# QF0: slots 0&1  QF1: slots 2&3  QF2: slots 4&5  QF3: slots 6&7
QF_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
# SF: QF0w vs QF2w  and  QF1w vs QF3w  (cross-bracket; actual WC2022 SFs)
# Argentina (QF0) vs Croatia (QF2); France (QF1) vs Morocco (QF3)
SF_PAIRS = [(0, 2), (1, 3)]

# Hardcoded WC2022 group assignments (StatsBomb team name spelling).
# Avoids date-sort ambiguity when two groups start on the same day.
WC2022_GROUPS = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "United States", "Wales", "Iran"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Germany", "Japan", "Costa Rica"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}


# ---------------------------------------------------------------------------
# Import shared helpers from match_outcome_simulation.py
# ---------------------------------------------------------------------------

def _import_helpers():
    spec = importlib.util.spec_from_file_location(
        "match_outcome_simulation",
        REPO_ROOT / "scripts" / "match_outcome_simulation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_h = _import_helpers()
get_predictions  = _h.get_predictions
apply_per_comp_T = _h.apply_per_comp_T
build_meta       = _h.build_meta


# ---------------------------------------------------------------------------
# Fixture loading + group reconstruction
# ---------------------------------------------------------------------------

def fetch_fixtures():
    try:
        from statsbombpy import sb
    except ImportError:
        print("ERROR: pip install statsbombpy"); sys.exit(1)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sb.matches(competition_id=COMP_ID, season_id=SEASON_ID)
    df = df.sort_values(["match_date", "kick_off"]).reset_index(drop=True)
    group_df  = df[df["competition_stage"] == "Group Stage"].copy()
    ko_df     = df[df["competition_stage"] != "Group Stage"].copy()
    return df, group_df, ko_df


def reconstruct_groups(group_df):
    """
    Infer group letters A-H from group-stage fixture connectivity.
    Connected components of the 'played against' undirected graph = groups.
    Groups sorted by earliest match date → assigned letters A-H.
    Returns {letter: [team0, team1, team2, team3]}.
    """
    from collections import deque

    adjacency: dict[str, set] = defaultdict(set)
    first_date: dict[str, str] = {}
    for _, row in group_df.iterrows():
        h, a = row["home_team"], row["away_team"]
        adjacency[h].add(a)
        adjacency[a].add(h)
        for t in [h, a]:
            if t not in first_date or row["match_date"] < first_date[t]:
                first_date[t] = row["match_date"]

    visited = set()
    components = []
    for start in sorted(adjacency.keys(), key=lambda t: first_date[t]):
        if start in visited:
            continue
        comp, queue = [], deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node); comp.append(node)
            queue.extend(adjacency[node] - visited)
        components.append((min(first_date[t] for t in comp), comp))

    components.sort(key=lambda x: x[0])
    letters = "ABCDEFGH"
    return {letters[i]: comp for i, (_, comp) in enumerate(components)}


# ---------------------------------------------------------------------------
# Model + data loading
# ---------------------------------------------------------------------------

def load_model():
    ckpt = PROCESSED / "pool_7comp_hybrid_gat_xg.pt"
    if not ckpt.exists():
        print(f"ERROR: {ckpt} not found"); sys.exit(1)

    wc_graphs = torch.load(
        PROCESSED / "statsbomb_wc2022_shot_graphs.pt", weights_only=False
    )
    g0     = wc_graphs[0]
    in_ch  = g0.x.shape[1]
    edge_ch = g0.edge_attr.shape[1] if g0.edge_attr is not None else 4

    model = HybridGATModel(
        node_in=in_ch, edge_dim=edge_ch,
        meta_dim=META_DIM, hidden=32, heads=4, n_layers=3, dropout=0.3,
    )
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    return model, wc_graphs


def build_match_xg_lookup(wc_graphs, probs_mc, probs_sb, group_df, ko_df):
    """
    Build {match_id: {home_team, away_team, home_xg_mc, away_xg_mc,
                       home_xg_sb, away_xg_sb, actual_home_goals, actual_away_goals}}.
    """
    match_xg: dict[int, dict] = {}

    # Group shots by match_id
    mid_to_shots: dict[int, list] = defaultdict(list)
    for i, g in enumerate(wc_graphs):
        mid_to_shots[g.match_id.item()].append((i, g))

    # Build actual scores from fixture dataframes
    all_df = group_df._append(ko_df)
    actual_scores = {}
    for _, row in all_df.iterrows():
        actual_scores[row["match_id"]] = {
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
        }

    for mid, items in mid_to_shots.items():
        info = actual_scores.get(mid, {})
        home_team = info.get("home_team", "")
        away_team = info.get("away_team", "")

        home_mc, away_mc, home_sb, away_sb = [], [], [], []
        for i, g in items:
            if g.team_name == home_team:
                home_mc.append(probs_mc[i])
                home_sb.append(probs_sb[i])
            else:
                away_mc.append(probs_mc[i])
                away_sb.append(probs_sb[i])

        match_xg[mid] = {
            "home_team":         home_team,
            "away_team":         away_team,
            "home_xg_mc":        np.array(home_mc, dtype=np.float32),
            "away_xg_mc":        np.array(away_mc, dtype=np.float32),
            "home_xg_sb":        np.array(home_sb, dtype=np.float32),
            "away_xg_sb":        np.array(away_sb, dtype=np.float32),
            "actual_home_goals": info.get("home_score", 0),
            "actual_away_goals": info.get("away_score", 0),
        }

    return match_xg


def build_team_fallback(match_xg, group_df):
    """
    Per-team average per-shot xG and median shot count per match.
    Used for counterfactual knockout matches not in historical data.
    """
    team_shots_mc: dict[str, list] = defaultdict(list)
    team_shots_per_match: dict[str, list] = defaultdict(list)

    group_mids = set(group_df["match_id"].tolist())
    for mid, m in match_xg.items():
        if mid not in group_mids:
            continue  # only group-stage data for unbiased fallback
        for team, xg_arr in [(m["home_team"], m["home_xg_mc"]),
                              (m["away_team"], m["away_xg_mc"])]:
            team_shots_mc[team].extend(xg_arr.tolist())
            team_shots_per_match[team].append(len(xg_arr))

    avg_per_shot = {
        t: float(np.mean(v)) if v else 0.10
        for t, v in team_shots_mc.items()
    }
    median_n_shots = {
        t: max(1, int(np.median(v))) if v else 7
        for t, v in team_shots_per_match.items()
    }
    global_avg = float(np.mean(list(avg_per_shot.values()))) if avg_per_shot else 0.10
    global_n   = 7
    return avg_per_shot, median_n_shots, global_avg, global_n


# ---------------------------------------------------------------------------
# Vectorised simulation helpers
# ---------------------------------------------------------------------------

def _sim_match_vectorised(
    team_h: str,
    team_a: str,
    pair_to_mid: dict,
    match_xg: dict,
    avg_per_shot: dict,
    med_shots: dict,
    global_avg: float,
    global_n: int,
    n_batch: int,
    use_sb: bool,
    rng: np.random.RandomState,
):
    """Vectorised Bernoulli simulation for n_batch sims of one matchup."""
    mid = pair_to_mid.get(frozenset({team_h, team_a}))
    xg_key_h = "home_xg_sb" if use_sb else "home_xg_mc"
    xg_key_a = "away_xg_sb" if use_sb else "away_xg_mc"

    if mid is not None:
        m = match_xg[mid]
        if m["home_team"] == team_h:
            hxg, axg = m[xg_key_h], m[xg_key_a]
        else:
            hxg, axg = m[xg_key_a], m[xg_key_h]
        # Clip to valid probability range
        hxg = np.clip(hxg, 0.0, 1.0)
        axg = np.clip(axg, 0.0, 1.0)
        if len(hxg) == 0:
            hg = np.zeros(n_batch, dtype=np.int32)
        else:
            hg = rng.binomial(1, hxg[None, :], size=(n_batch, len(hxg))).sum(1)
        if len(axg) == 0:
            ag = np.zeros(n_batch, dtype=np.int32)
        else:
            ag = rng.binomial(1, axg[None, :], size=(n_batch, len(axg))).sum(1)
    else:
        # Counterfactual: use team average per-shot xG × median shots
        ph = avg_per_shot.get(team_h, global_avg)
        pa = avg_per_shot.get(team_a, global_avg)
        nh = med_shots.get(team_h, global_n)
        na = med_shots.get(team_a, global_n)
        hg = rng.binomial(nh, ph, size=n_batch)
        ag = rng.binomial(na, pa, size=n_batch)

    return hg.astype(np.int32), ag.astype(np.int32)


def _resolve_ko_slot(
    teams_home: np.ndarray,  # (N,) object array of strings
    teams_away: np.ndarray,  # (N,) object array of strings
    pair_to_mid: dict,
    match_xg: dict,
    avg_per_shot: dict,
    med_shots: dict,
    global_avg: float,
    global_n: int,
    use_sb: bool,
    ko_draws: str,
    rng: np.random.RandomState,
):
    """
    Resolve one knockout slot across all N simulations.
    Groups sims by their (team_h, team_a) pair for vectorised batching.
    Returns winners (N,) and losers (N,) as object arrays of strings.
    """
    N = len(teams_home)
    winners = np.empty(N, dtype=object)
    losers  = np.empty(N, dtype=object)

    # Find unique matchups and batch-simulate
    pairs = list(zip(teams_home.tolist(), teams_away.tolist()))
    unique_pairs = list(set(pairs))

    for (th, ta) in unique_pairs:
        mask = np.array([p == (th, ta) for p in pairs], dtype=bool)
        n_batch = int(mask.sum())
        if n_batch == 0:
            continue

        hg, ag = _sim_match_vectorised(
            th, ta, pair_to_mid, match_xg,
            avg_per_shot, med_shots, global_avg, global_n,
            n_batch, use_sb, rng,
        )

        # Resolve draws
        draws_mask = hg == ag
        advance_home = hg > ag  # (n_batch,) bool

        if draws_mask.any():
            n_draws = int(draws_mask.sum())
            if ko_draws == "penalties":
                coin = rng.randint(0, 2, size=n_draws)
                advance_home[draws_mask] = coin == 0
            else:  # extra_time: Poisson resample
                ph = avg_per_shot.get(th, global_avg)
                pa = avg_per_shot.get(ta, global_avg)
                nh = max(1, med_shots.get(th, global_n) // 4)
                na = max(1, med_shots.get(ta, global_n) // 4)
                et_h = rng.binomial(nh, ph, size=n_draws)
                et_a = rng.binomial(na, pa, size=n_draws)
                # still a draw in ET → home advances (simplification)
                advance_home[draws_mask] = et_h >= et_a

        w = np.where(advance_home, th, ta)
        l = np.where(advance_home, ta, th)
        winners[mask] = w
        losers[mask]  = l

    return winners, losers


# ---------------------------------------------------------------------------
# Group stage (vectorised)
# ---------------------------------------------------------------------------

def simulate_group_stage(groups, group_df, match_xg, pair_to_mid,
                         avg_per_shot, med_shots, global_avg, global_n,
                         N, use_sb, rng):
    """
    Simulate all 48 group-stage matches across N sims.
    Returns:
      group_rank[letter][rank] = (N,) string array of team names
      e.g. group_rank["A"][0] = array of N group-A winners
    """
    team_to_group = {}
    for letter, teams in groups.items():
        for t in teams:
            team_to_group[t] = letter

    # Pre-simulate all 48 matches vectorised
    mid_goals: dict[int, tuple] = {}  # mid → (home_goals_N, away_goals_N)
    for mid, m in match_xg.items():
        if mid not in set(group_df["match_id"].tolist()):
            continue
        xk_h = "home_xg_sb" if use_sb else "home_xg_mc"
        xk_a = "away_xg_sb" if use_sb else "away_xg_mc"
        hxg = np.clip(m[xk_h], 0, 1)
        axg = np.clip(m[xk_a], 0, 1)
        hg = rng.binomial(1, hxg[None, :], size=(N, len(hxg))).sum(1) if len(hxg) else np.zeros(N, int)
        ag = rng.binomial(1, axg[None, :], size=(N, len(axg))).sum(1) if len(axg) else np.zeros(N, int)
        mid_goals[mid] = (hg.astype(np.int32), ag.astype(np.int32))

    group_mid_df = group_df[["match_id", "home_team", "away_team"]].copy()

    group_rank: dict[str, list] = {}  # letter → [winner_N, runner_N, 3rd_N, 4th_N]

    for letter, teams in groups.items():
        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        # Accumulate points, GD, GF — all shape (n_teams, N)
        pts = np.zeros((n_teams, N), dtype=np.int32)
        gd  = np.zeros((n_teams, N), dtype=np.int32)
        gf  = np.zeros((n_teams, N), dtype=np.int32)

        group_matches = group_mid_df[
            group_mid_df["home_team"].isin(teams) & group_mid_df["away_team"].isin(teams)
        ]

        for _, row in group_matches.iterrows():
            mid = row["match_id"]
            hi = team_idx[row["home_team"]]
            ai = team_idx[row["away_team"]]

            if mid in mid_goals:
                hg, ag = mid_goals[mid]
            else:
                # Fallback: no shots in data for this match
                ph = avg_per_shot.get(row["home_team"], global_avg)
                pa = avg_per_shot.get(row["away_team"], global_avg)
                nh = med_shots.get(row["home_team"], global_n)
                na = med_shots.get(row["away_team"], global_n)
                hg = rng.binomial(nh, ph, size=N)
                ag = rng.binomial(na, pa, size=N)

            home_win = hg > ag
            draw_    = hg == ag
            away_win = hg < ag

            pts[hi] += np.where(home_win, 3, np.where(draw_, 1, 0))
            pts[ai] += np.where(away_win, 3, np.where(draw_, 1, 0))
            gd[hi]  += hg - ag
            gd[ai]  += ag - hg
            gf[hi]  += hg
            gf[ai]  += ag

        # Composite sort key (descending): points → GD → GF
        # offset GD by 30 to ensure positive values (max GD = ±9)
        composite = pts * 10000 + (gd + 30) * 100 + gf  # (n_teams, N)
        order = np.argsort(-composite, axis=0)            # (n_teams, N)

        team_arr = np.array(teams)
        group_rank[letter] = [team_arr[order[r, :]] for r in range(n_teams)]

    return group_rank


# ---------------------------------------------------------------------------
# Knockout stage
# ---------------------------------------------------------------------------

def simulate_knockout(group_rank, pair_to_mid, match_xg,
                      avg_per_shot, med_shots, global_avg, global_n,
                      N, use_sb, ko_draws, rng):
    """
    Sequential knockout simulation. Vectorised by matchup batch within each slot.
    Returns per-team stage-count dicts (how many of N sims each team reached each stage).
    """
    # R16 bracket
    r16_winners = []
    r16_losers  = []
    for (gA, rA, gB, rB) in R16_BRACKET:
        th = group_rank[gA][rA]  # (N,) string array
        ta = group_rank[gB][rB]  # (N,)
        w, l = _resolve_ko_slot(
            th, ta, pair_to_mid, match_xg,
            avg_per_shot, med_shots, global_avg, global_n,
            use_sb, ko_draws, rng,
        )
        r16_winners.append(w)
        r16_losers.append(l)

    # QF
    qf_winners = []
    qf_losers  = []
    for (i, j) in QF_PAIRS:
        w, l = _resolve_ko_slot(
            r16_winners[i], r16_winners[j],
            pair_to_mid, match_xg,
            avg_per_shot, med_shots, global_avg, global_n,
            use_sb, ko_draws, rng,
        )
        qf_winners.append(w)
        qf_losers.append(l)

    # SF
    sf_winners = []
    sf_losers  = []
    for (i, j) in SF_PAIRS:
        w, l = _resolve_ko_slot(
            qf_winners[i], qf_winners[j],
            pair_to_mid, match_xg,
            avg_per_shot, med_shots, global_avg, global_n,
            use_sb, ko_draws, rng,
        )
        sf_winners.append(w)
        sf_losers.append(l)

    # 3rd place
    third_w, _ = _resolve_ko_slot(
        sf_losers[0], sf_losers[1],
        pair_to_mid, match_xg,
        avg_per_shot, med_shots, global_avg, global_n,
        use_sb, ko_draws, rng,
    )

    # Final
    final_w, final_l = _resolve_ko_slot(
        sf_winners[0], sf_winners[1],
        pair_to_mid, match_xg,
        avg_per_shot, med_shots, global_avg, global_n,
        use_sb, ko_draws, rng,
    )

    return {
        "r16_winners":  r16_winners,  # list of 8 (N,) arrays
        "r16_losers":   r16_losers,
        "qf_winners":   qf_winners,   # list of 4
        "qf_losers":    qf_losers,
        "sf_winners":   sf_winners,   # list of 2
        "sf_losers":    sf_losers,
        "third_w":      third_w,      # (N,)
        "final_w":      final_w,      # (N,) champion
        "final_l":      final_l,      # (N,) runner-up
    }


def tally_probabilities(group_rank, ko, N):
    """
    Aggregate stage counts → per-team probability dict.
    """
    counts: dict[str, dict] = defaultdict(lambda: {
        "n_advance": 0, "n_r16": 0, "n_qf": 0,
        "n_sf": 0, "n_finalist": 0, "n_champion": 0,
    })

    # Group advance
    for letter, rank_arrays in group_rank.items():
        for rank_arr in rank_arrays[:2]:  # top 2 advance
            for team in np.unique(rank_arr):
                counts[team]["n_advance"] += int((rank_arr == team).sum())

    # R16 advance (= reached QF)
    for arr in ko["r16_winners"]:
        for team in np.unique(arr):
            counts[team]["n_r16"] += int((arr == team).sum())

    # QF advance (= reached SF)
    for arr in ko["qf_winners"]:
        for team in np.unique(arr):
            counts[team]["n_qf"] += int((arr == team).sum())

    # SF advance (= reached Final)
    for arr in ko["sf_winners"]:
        for team in np.unique(arr):
            counts[team]["n_sf"] += int((arr == team).sum())

    # Finalist + champion
    for team in np.unique(ko["final_w"]):
        counts[team]["n_finalist"] += int((ko["final_w"] == team).sum())
        counts[team]["n_champion"] += int((ko["final_w"] == team).sum())
    for team in np.unique(ko["final_l"]):
        counts[team]["n_finalist"] += int((ko["final_l"] == team).sum())

    probs = {}
    for team, c in counts.items():
        probs[team] = {
            "p_advance":   c["n_advance"]  / N,
            "p_r16":       c["n_r16"]      / N,
            "p_qf":        c["n_qf"]       / N,
            "p_sf":        c["n_sf"]       / N,
            "p_finalist":  c["n_finalist"] / N,
            "p_champion":  c["n_champion"] / N,
        }
    return probs


# ---------------------------------------------------------------------------
# Actual finish derivation
# ---------------------------------------------------------------------------

def compute_actual_finishes(all_df, all_teams):
    finish = {t: "group" for t in all_teams}

    # Teams that appear in each knockout stage
    r16_teams  = set(all_df[all_df["competition_stage"] == "Round of 16"][["home_team","away_team"]].values.flatten())
    qf_teams   = set(all_df[all_df["competition_stage"] == "Quarter-finals"][["home_team","away_team"]].values.flatten())
    sf_teams   = set(all_df[all_df["competition_stage"] == "Semi-finals"][["home_team","away_team"]].values.flatten())
    third_row  = all_df[all_df["competition_stage"] == "3rd Place Final"].iloc[0]
    final_row  = all_df[all_df["competition_stage"] == "Final"].iloc[0]

    for t in r16_teams:  finish[t] = "r16"
    for t in qf_teams:   finish[t] = "quarter"
    for t in sf_teams:   finish[t] = "semi"

    # 3rd/4th: both played the 3rd place match
    finish[third_row["home_team"]] = "third_fourth"
    finish[third_row["away_team"]] = "third_fourth"

    # Finalist + champion (both finalists; need to know winner)
    finish[final_row["home_team"]] = "finalist"
    finish[final_row["away_team"]] = "finalist"

    # Argentina won on penalties (StatsBomb records 3-3 FT; we hardcode the winner)
    finish["Argentina"] = "champion"

    return finish


FINISH_ORDER = ["group", "r16", "quarter", "semi", "third_fourth", "finalist", "champion"]
FINISH_RANK  = {f: i for i, f in enumerate(FINISH_ORDER)}


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def compute_metrics(probs_mc, probs_sb, actual_finish, groups, match_xg, all_df):
    all_teams = sorted(set(actual_finish.keys()))

    # ── Group advancement Brier ───────────────────────────────────────────────
    y_advance   = np.array([1 if actual_finish.get(t, "group") != "group" else 0
                            for t in all_teams])
    p_adv_mc    = np.array([probs_mc.get(t, {}).get("p_advance", 0) for t in all_teams])
    p_adv_sb    = np.array([probs_sb.get(t, {}).get("p_advance", 0) for t in all_teams])
    brier_mc    = float(brier_score_loss(y_advance, p_adv_mc))
    brier_sb    = float(brier_score_loss(y_advance, p_adv_sb))

    # ── Top-pick (champion) ───────────────────────────────────────────────────
    top_mc = max(all_teams, key=lambda t: probs_mc.get(t, {}).get("p_champion", 0))
    top_sb = max(all_teams, key=lambda t: probs_sb.get(t, {}).get("p_champion", 0))

    # ── Group top-2 accuracy ──────────────────────────────────────────────────
    group_correct_both_mc = 0
    group_correct_one_mc  = 0
    group_correct_both_sb = 0
    group_correct_one_sb  = 0
    group_details = {}

    for letter, teams in groups.items():
        # Actual top-2 (from group_df fixture results)
        group_match_rows = all_df[
            (all_df["competition_stage"] == "Group Stage") &
            (all_df["home_team"].isin(teams) | all_df["away_team"].isin(teams))
        ]
        # actual top-2 = teams that appear in R16 and are in this group
        r16_teams = set(all_df[all_df["competition_stage"] == "Round of 16"][["home_team","away_team"]].values.flatten())
        actual_top2 = {t for t in teams if t in r16_teams}

        # Model's most likely top-2
        mc_top2 = set(sorted(teams, key=lambda t: -probs_mc.get(t, {}).get("p_advance", 0))[:2])
        sb_top2 = set(sorted(teams, key=lambda t: -probs_sb.get(t, {}).get("p_advance", 0))[:2])

        mc_overlap = len(mc_top2 & actual_top2)
        sb_overlap = len(sb_top2 & actual_top2)

        if mc_overlap == 2: group_correct_both_mc += 1
        elif mc_overlap == 1: group_correct_one_mc += 1
        if sb_overlap == 2: group_correct_both_sb += 1
        elif sb_overlap == 1: group_correct_one_sb += 1

        group_details[letter] = {
            "teams":       sorted(teams),
            "actual_top2": sorted(actual_top2),
            "mc_top2":     sorted(mc_top2),
            "sb_top2":     sorted(sb_top2),
        }

    # ── Spearman ρ vs actual finish rank ─────────────────────────────────────
    actual_ranks = np.array([FINISH_RANK.get(actual_finish.get(t, "group"), 0) for t in all_teams])
    mc_champ_p   = np.array([probs_mc.get(t, {}).get("p_champion", 0) for t in all_teams])
    sb_champ_p   = np.array([probs_sb.get(t, {}).get("p_champion", 0) for t in all_teams])
    rho_mc = float(spearmanr(mc_champ_p, actual_ranks).statistic)
    rho_sb = float(spearmanr(sb_champ_p, actual_ranks).statistic)

    # ── Historical 64-match 3-way accuracy ───────────────────────────────────
    n_correct_mc, n_correct_sb, n_total = 0, 0, 0
    for mid, m in match_xg.items():
        ah, aa = m["actual_home_goals"], m["actual_away_goals"]
        actual_res = "home" if ah > aa else ("draw" if ah == aa else "away")

        # Expected goals for 3-way prediction
        exp_h_mc = float(m["home_xg_mc"].sum()) if len(m["home_xg_mc"]) else 0
        exp_a_mc = float(m["away_xg_mc"].sum()) if len(m["away_xg_mc"]) else 0
        exp_h_sb = float(m["home_xg_sb"].sum()) if len(m["home_xg_sb"]) else 0
        exp_a_sb = float(m["away_xg_sb"].sum()) if len(m["away_xg_sb"]) else 0

        pred_mc = "home" if exp_h_mc > exp_a_mc else ("draw" if abs(exp_h_mc - exp_a_mc) < 0.15 else "away")
        pred_sb = "home" if exp_h_sb > exp_a_sb else ("draw" if abs(exp_h_sb - exp_a_sb) < 0.15 else "away")

        n_correct_mc += (pred_mc == actual_res)
        n_correct_sb += (pred_sb == actual_res)
        n_total += 1

    return {
        "brier_group_advance_mc": round(brier_mc, 4),
        "brier_group_advance_sb": round(brier_sb, 4),
        "champion_pick_mc":       top_mc,
        "champion_pick_sb":       top_sb,
        "champion_pick_correct_mc": (top_mc == "Argentina"),
        "champion_pick_correct_sb": (top_sb == "Argentina"),
        "group_top2_mc": {"correct_both": group_correct_both_mc, "correct_one": group_correct_one_mc,
                          "correct_none": 8 - group_correct_both_mc - group_correct_one_mc},
        "group_top2_sb": {"correct_both": group_correct_both_sb, "correct_one": group_correct_one_sb,
                          "correct_none": 8 - group_correct_both_sb - group_correct_one_sb},
        "spearman_rho_mc": round(rho_mc, 4),
        "spearman_rho_sb": round(rho_sb, 4),
        "historical_xg_accuracy_mc": round(n_correct_mc / max(n_total, 1), 4),
        "historical_xg_accuracy_sb": round(n_correct_sb / max(n_total, 1), 4),
        "n_historical_matches": n_total,
        "group_details": group_details,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(probs_mc, probs_sb, actual_finish, groups, N, metrics):
    all_teams = sorted(set(actual_finish.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"WC2022 Tournament Simulation — HybridGATv2 vs StatsBomb xG  (N={N:,})",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: P(win WC2022) — top 12 ──────────────────────────────────────
    ax = axes[0, 0]
    top12 = sorted(all_teams,
                   key=lambda t: -probs_mc.get(t, {}).get("p_champion", 0))[:12]
    x = np.arange(len(top12))
    w = 0.38
    mc_vals = [probs_mc.get(t, {}).get("p_champion", 0) for t in top12]
    sb_vals = [probs_sb.get(t, {}).get("p_champion", 0) for t in top12]
    bars1 = ax.bar(x - w/2, mc_vals, w, color="steelblue", alpha=0.85, label="HybridGAT MC")
    bars2 = ax.bar(x + w/2, sb_vals, w, color="tomato",    alpha=0.85, label="StatsBomb xG")
    # Annotate actual champion
    for i, t in enumerate(top12):
        if actual_finish.get(t) == "champion":
            ax.text(i, max(mc_vals[i], sb_vals[i]) + 0.005, "★", ha="center",
                    fontsize=12, color="gold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("United States", "USA").replace("Netherlands","NED")
                        for t in top12], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("P(champion)")
    ax.set_title("P(Win WC2022) — Top 12 Teams")
    ax.legend(fontsize=8)

    # ── Panel 2: P(advance from group) — all 32 ──────────────────────────────
    ax = axes[0, 1]
    sorted_teams = sorted(all_teams,
                          key=lambda t: -probs_mc.get(t, {}).get("p_advance", 0))
    mc_adv = [probs_mc.get(t, {}).get("p_advance", 0) for t in sorted_teams]
    sb_adv = [probs_sb.get(t, {}).get("p_advance", 0) for t in sorted_teams]
    y = np.arange(len(sorted_teams))
    ax.barh(y - 0.2, mc_adv, 0.35, color="steelblue", alpha=0.8, label="HybridGAT")
    ax.barh(y + 0.2, sb_adv, 0.35, color="tomato",    alpha=0.8, label="StatsBomb")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_teams, fontsize=6.5)
    # Mark teams that actually advanced
    for i, t in enumerate(sorted_teams):
        if actual_finish.get(t, "group") != "group":
            ax.text(1.01, i, "✓", va="center", fontsize=7, color="green",
                    transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.4)
    ax.set_xlabel("P(advance from group)")
    ax.set_title("Group Advancement Probability — All 32 Teams")
    ax.legend(fontsize=8, loc="lower right")

    # ── Panel 3: Simulated champion frequency (top 10) ────────────────────────
    ax = axes[1, 0]
    top10 = sorted(all_teams,
                   key=lambda t: -probs_mc.get(t, {}).get("p_champion", 0))[:10]
    mc_c  = [probs_mc.get(t, {}).get("p_champion", 0) * N for t in top10]
    colors = ["gold" if actual_finish.get(t) == "champion" else "steelblue" for t in top10]
    bars = ax.bar(range(len(top10)), mc_c, color=colors, alpha=0.85, edgecolor="white")
    for b, v in zip(bars, mc_c):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                f"{v/N:.1%}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(top10, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Simulated champion count (out of {N:,})")
    ax.set_title("Champion Frequency — Top 10 Teams\n(gold = actual WC2022 champion)")

    # ── Panel 4: Group advance calibration ────────────────────────────────────
    ax = axes[1, 1]
    y_adv = np.array([1 if actual_finish.get(t, "group") != "group" else 0
                      for t in all_teams])
    p_mc_all = np.array([probs_mc.get(t, {}).get("p_advance", 0) for t in all_teams])
    p_sb_all = np.array([probs_sb.get(t, {}).get("p_advance", 0) for t in all_teams])

    bins = np.linspace(0, 1, 7)
    for p_vals, label, color in [
        (p_mc_all, f"HybridGAT (Brier={metrics['brier_group_advance_mc']:.3f})", "steelblue"),
        (p_sb_all, f"StatsBomb  (Brier={metrics['brier_group_advance_sb']:.3f})", "tomato"),
    ]:
        frac, mp = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p_vals >= lo) & (p_vals < hi)
            if mask.sum() >= 2:
                frac.append(float(y_adv[mask].mean()))
                mp.append(float(p_vals[mask].mean()))
        if mp:
            ax.plot(mp, frac, "o-", color=color, lw=1.8, ms=7, label=label)
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfect")
    ax.set_xlabel("Predicted P(advance from group)")
    ax.set_ylabel("Actual advance rate")
    ax.set_title("Group Advancement Calibration")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sim",     type=int,  default=10_000)
    p.add_argument("--ko-draws",  type=str,  default="penalties",
                   choices=["penalties", "extra_time"])
    p.add_argument("--batch-size",type=int,  default=64)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.RandomState(SEED)
    N    = args.n_sim

    print("=" * 64)
    print(f"  WC2022 Tournament Simulation  (N={N:,}  ko={args.ko_draws})")
    print("=" * 64)

    # ── Fixtures ─────────────────────────────────────────────────────────────
    print("\n── Fetching WC2022 fixtures …")
    all_df, group_df, ko_df = fetch_fixtures()
    # Use hardcoded group assignments to avoid date-sort ambiguity (E/F both start Nov 23)
    groups = WC2022_GROUPS.copy()
    all_teams = sorted({t for grp in groups.values() for t in grp})
    print(f"  {len(group_df)} group matches · {len(ko_df)} knockout matches")
    for letter, teams in sorted(groups.items()):
        print(f"  Group {letter}: {', '.join(teams)}")

    # ── Model + inference ─────────────────────────────────────────────────────
    print("\n── Loading model and WC2022 shot graphs …")
    model, wc_graphs = load_model()
    print(f"  {len(wc_graphs)} shots loaded")

    per_T_path = PROCESSED / "pool_7comp_per_comp_T_gat.pt"
    per_T = torch.load(per_T_path, weights_only=False) if per_T_path.exists() else {}
    T_global_path = PROCESSED / "pool_7comp_gat_T.pt"
    T_global = 1.0
    if T_global_path.exists():
        d = torch.load(T_global_path, weights_only=False)
        T_global = float(d["T"]) if isinstance(d, dict) else float(d)

    probs_raw = get_predictions(model, wc_graphs, batch_size=args.batch_size)
    probs_mc  = apply_per_comp_T(probs_raw, wc_graphs, per_T, T_global)
    probs_sb  = np.array([g.sb_xg.item() for g in wc_graphs])
    print(f"  Inference done. T_global={T_global:.4f}")

    # ── Match xG lookup ───────────────────────────────────────────────────────
    print("\n── Building match xG lookup …")
    match_xg = build_match_xg_lookup(wc_graphs, probs_mc, probs_sb, group_df, ko_df)
    pair_to_mid = {frozenset({m["home_team"], m["away_team"]}): mid
                   for mid, m in match_xg.items()}
    print(f"  {len(match_xg)} matches with shot data")

    avg_per_shot, med_shots, global_avg, global_n = build_team_fallback(match_xg, group_df)

    # ── Actual finishes ───────────────────────────────────────────────────────
    actual_finish = compute_actual_finishes(all_df, all_teams)

    # ── Simulate: HybridGAT ───────────────────────────────────────────────────
    print(f"\n── Simulating {N:,} tournaments (HybridGAT xG) …")
    rng_mc = np.random.RandomState(SEED)
    group_rank_mc = simulate_group_stage(
        groups, group_df, match_xg, pair_to_mid,
        avg_per_shot, med_shots, global_avg, global_n,
        N, use_sb=False, rng=rng_mc,
    )
    ko_mc = simulate_knockout(
        group_rank_mc, pair_to_mid, match_xg,
        avg_per_shot, med_shots, global_avg, global_n,
        N, use_sb=False, ko_draws=args.ko_draws, rng=rng_mc,
    )
    probs_mc_team = tally_probabilities(group_rank_mc, ko_mc, N)

    # ── Simulate: StatsBomb ───────────────────────────────────────────────────
    print(f"── Simulating {N:,} tournaments (StatsBomb xG) …")
    rng_sb = np.random.RandomState(SEED)
    group_rank_sb = simulate_group_stage(
        groups, group_df, match_xg, pair_to_mid,
        avg_per_shot, med_shots, global_avg, global_n,
        N, use_sb=True, rng=rng_sb,
    )
    ko_sb = simulate_knockout(
        group_rank_sb, pair_to_mid, match_xg,
        avg_per_shot, med_shots, global_avg, global_n,
        N, use_sb=True, ko_draws=args.ko_draws, rng=rng_sb,
    )
    probs_sb_team = tally_probabilities(group_rank_sb, ko_sb, N)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n── Computing validation metrics …")
    metrics = compute_metrics(probs_mc_team, probs_sb_team, actual_finish,
                              groups, match_xg, all_df)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n  {'Metric':<45} {'HybridGAT':>10}  {'StatsBomb':>10}")
    print(f"  {'-'*68}")
    print(f"  {'Group advance Brier':<45} {metrics['brier_group_advance_mc']:>10.4f}  {metrics['brier_group_advance_sb']:>10.4f}")
    print(f"  {'Champion pick correct (Argentina?)':<45} {str(metrics['champion_pick_correct_mc']):>10}  {str(metrics['champion_pick_correct_sb']):>10}")
    print(f"  {'  Champion predicted':<45} {metrics['champion_pick_mc']:>10}  {metrics['champion_pick_sb']:>10}")
    print(f"  {'Group top-2 correct (both)':<45} {metrics['group_top2_mc']['correct_both']:>10}/8  {metrics['group_top2_sb']['correct_both']:>10}/8")
    print(f"  {'Group top-2 correct (one)':<45} {metrics['group_top2_mc']['correct_one']:>10}/8  {metrics['group_top2_sb']['correct_one']:>10}/8")
    print(f"  {'Spearman ρ vs actual finish':<45} {metrics['spearman_rho_mc']:>10.4f}  {metrics['spearman_rho_sb']:>10.4f}")
    print(f"  {'xG-implied 3-way match accuracy':<45} {metrics['historical_xg_accuracy_mc']:>10.1%}  {metrics['historical_xg_accuracy_sb']:>10.1%}")

    print(f"\n  P(champion) — top 10 teams:")
    top10 = sorted(all_teams,
                   key=lambda t: -probs_mc_team.get(t, {}).get("p_champion", 0))[:10]
    print(f"  {'Team':<25} {'P(champ) MC':>12}  {'P(champ) SB':>12}  {'Actual':>12}")
    for t in top10:
        print(f"  {t:<25} {probs_mc_team.get(t,{}).get('p_champion',0):>12.3f}  "
              f"{probs_sb_team.get(t,{}).get('p_champion',0):>12.3f}  "
              f"{actual_finish.get(t,'?'):>12}")

    print(f"\n  P(advance from group) — teams that actually advanced:")
    advanced = [t for t in all_teams if actual_finish.get(t, "group") != "group"]
    print(f"  {'Team':<25} {'P(adv) MC':>10}  {'P(adv) SB':>10}  {'Finish':>12}")
    for t in sorted(advanced, key=lambda t: -probs_mc_team.get(t,{}).get("p_advance",0)):
        print(f"  {t:<25} {probs_mc_team.get(t,{}).get('p_advance',0):>10.3f}  "
              f"{probs_sb_team.get(t,{}).get('p_advance',0):>10.3f}  "
              f"{actual_finish.get(t,'?'):>12}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "n_sim":     N,
        "ko_draws":  args.ko_draws,
        **metrics,
        "teams": {
            t: {
                "group":          next((l for l, ts in groups.items() if t in ts), "?"),
                "actual_finish":  actual_finish.get(t, "group"),
                **{f"{k}_mc": round(v, 4) for k, v in probs_mc_team.get(t, {}).items()},
                **{f"{k}_sb": round(v, 4) for k, v in probs_sb_team.get(t, {}).items()},
            }
            for t in sorted(all_teams)
        },
    }
    out_path = PROCESSED / "wc2022_tournament_sim_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("── Generating figure …")
    fig = make_figure(probs_mc_team, probs_sb_team, actual_finish, groups, N, metrics)
    fig_path = ASSETS / "fig_wc2022_tournament.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
