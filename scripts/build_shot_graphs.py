#!/usr/bin/env python3
"""
build_shot_graphs.py
---------------------
Build PyG graph datasets from StatsBomb 360 shot freeze-frames.

Task: xG (Expected Goals) — binary classification
  Label 0 = no goal, Label 1 = goal
  Also stores StatsBomb's own xG (shot_statsbomb_xg) on every graph
  so we can benchmark our model against the industry standard.

Node features (9):
  [x_m, y_m, teammate, actor (shooter), keeper,
   dist_to_atk_goal, dist_to_def_goal, angle_to_goal, pressure]

Graph-level metadata stored on each Data object:
  data.y           = tensor([1.0]) if goal else tensor([0.0])
  data.sb_xg       = tensor([float]) — StatsBomb's xG prediction
  data.shot_dist   = tensor([float]) — shooter distance to goal (metres)
  data.shot_angle  = tensor([float]) — shooter angle to goal (radians)
  data.is_header   = tensor([0/1])
  data.is_open_play = tensor([0/1])

Usage:
    # Build WC2022 shot graphs (~960 shots, ~100 goals)
    python scripts/build_shot_graphs.py --competition 43 --season 106 --label wc2022

    # Build WWC2023 for cross-competition test
    python scripts/build_shot_graphs.py --competition 72 --season 107 --label wwc2023

    # Pool multiple competitions into one file
    python scripts/build_shot_graphs.py --competition 43 72 --season 106 107 --label wc_pool
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import requests
from torch_geometric.data import Data
from statsbombpy import sb

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.graph_builder import (
    normalise_statsbomb,
    _edges_delaunay,
    _compute_edge_features,
    PITCH_LENGTH, PITCH_WIDTH,
)
from src.features import enrich_graph, encode_technique

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Goal centre in metres (StatsBomb pitch: goal at x=120, y=40 → metres)
GOAL_X = (120 / 120) * PITCH_LENGTH   # 105.0 m
GOAL_Y = (40  /  80) * PITCH_WIDTH    # 34.0 m

# Goal-post y-coordinates (goal width = 7.32 m, centred at y=34)
GOAL_POST_LEFT  = np.array([GOAL_X, GOAL_Y - 3.66], dtype=np.float32)
GOAL_POST_RIGHT = np.array([GOAL_X, GOAL_Y + 3.66], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_frames(match_id: int) -> dict:
    url = (
        f"https://raw.githubusercontent.com/statsbomb/open-data/master"
        f"/data/three-sixty/{match_id}.json"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return {rec["event_uuid"]: rec["freeze_frame"] for rec in r.json()}
        except Exception:
            time.sleep(2 ** attempt)
    return {}


def _point_in_triangle(p: np.ndarray, a: np.ndarray,
                       b: np.ndarray, c: np.ndarray) -> bool:
    """Return True if point p lies inside triangle abc (2-D)."""
    def _sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = _sign(p, a, b), _sign(p, b, c), _sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _gk_pressure_features(freeze_frame: list, shot_loc: list) -> dict:
    """
    Compute goalkeeper-pressure features directly from the StatsBomb freeze frame.

    Features
    --------
    gk_dist          : Euclidean distance (m) from shooter to GK.
                       Default 35.0 if no GK found.
    n_def_in_cone    : Count of outfield defenders inside the shooting cone
                       (triangle: shooter → left post → right post).
    gk_off_centre    : Absolute lateral displacement of GK from the goal
                       centre-line, normalised by half-goal-width (3.66 m).
                       0 = perfectly centred, 1 = on a post, >1 = off far post.
    gk_perp_offset   : Perpendicular distance (m) of the GK from the straight
                       line connecting the shooter to the goal centre.
                       0 = GK perfectly on the shot line (best position),
                       large = GK is way off the direct shot path (easy to score).
    n_def_direct_line: Count of defenders within a ≤3° half-angle cone pointing
                       from the shooter directly at goal centre, and between the
                       shooter and the goal. Stricter than n_def_in_cone; captures
                       defenders who are literally blocking the straight shot.
    """
    sx, sy = normalise_statsbomb(*shot_loc)
    shooter  = np.array([sx, sy], dtype=np.float32)
    shot_dir = np.array([GOAL_X - sx, GOAL_Y - sy], dtype=np.float32)  # shooter→goal
    shot_len = float(np.linalg.norm(shot_dir))

    gk_pos = None
    def_positions: list[np.ndarray] = []

    for p in freeze_frame:
        if p.get("actor", False):
            continue  # skip the shooter himself
        x_m, y_m = normalise_statsbomb(*p["location"])
        pos = np.array([x_m, y_m], dtype=np.float32)
        if p.get("keeper", False):
            gk_pos = pos
        elif not p.get("teammate", True):   # opposing outfield player
            def_positions.append(pos)

    # ── GK metrics ────────────────────────────────────────────────────────────
    if gk_pos is not None:
        gk_dist       = float(np.linalg.norm(shooter - gk_pos))
        gk_off_centre = float(abs(gk_pos[1] - GOAL_Y) / 3.66)
        # Perpendicular distance from GK to the shooter→goal-centre line
        # = |cross(shot_dir, shooter→gk)| / |shot_dir|
        if shot_len > 0.1:
            to_gk = gk_pos - shooter
            cross = shot_dir[0] * to_gk[1] - shot_dir[1] * to_gk[0]
            gk_perp_offset = float(abs(cross) / shot_len)
        else:
            gk_perp_offset = 0.0
    else:
        gk_dist        = 35.0   # sensible default when GK absent from freeze frame
        gk_off_centre  = 0.0
        gk_perp_offset = 3.0   # assume GK is off the line (penalty-area average)

    # ── Defenders inside wide shooting cone (triangle to posts) ───────────────
    n_def_in_cone = 0
    for dp in def_positions:
        if _point_in_triangle(dp, shooter, GOAL_POST_LEFT, GOAL_POST_RIGHT):
            n_def_in_cone += 1

    # ── Defenders in the narrow direct-shot line (≤3° half-angle) ────────────
    # A defender counts only if they are (a) between shooter and goal and
    # (b) within 3° of the straight shooter→goal-centre direction.
    n_def_direct_line = 0
    if shot_len > 0.1:
        for dp in def_positions:
            to_def     = dp - shooter
            dist_to_def = float(np.linalg.norm(to_def))
            if dist_to_def < 0.5:
                continue   # ignore players right on top of shooter
            # Projection scalar along the shot direction (0=shooter, 1=goal)
            proj = float(np.dot(to_def, shot_dir) / (shot_len ** 2))
            if proj <= 0.0 or proj > 1.1:
                continue   # behind shooter or well past goal
            # Angle between shot direction and direction to defender
            cos_a     = float(np.dot(to_def, shot_dir) / (dist_to_def * shot_len))
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            if angle_deg <= 3.0:
                n_def_direct_line += 1

    return {
        "gk_dist":           gk_dist,
        "n_def_in_cone":     float(n_def_in_cone),
        "gk_off_centre":     gk_off_centre,
        "gk_perp_offset":    gk_perp_offset,
        "n_def_direct_line": float(n_def_direct_line),
    }


def _shot_distance_angle(shot_loc: list) -> tuple[float, float]:
    """Return (distance_m, angle_rad) from shot location to goal."""
    x_m, y_m = normalise_statsbomb(shot_loc[0], shot_loc[1])
    dx = GOAL_X - x_m
    dy = GOAL_Y - y_m
    dist = np.hypot(dx, dy)
    # Angle = width of goal as seen from shooter position
    # goal posts at y=34±3.66 m → approximate with arctan of half-width / dist
    angle = np.arctan2(7.32, max(dist, 0.1))   # 7.32m = standard goal width
    return float(dist), float(angle)


def build_shot_graph(freeze_frame: list, label: float,
                     sb_xg: float, shot_loc: list,
                     is_header: bool, is_open_play: bool,
                     technique: str = "",
                     match_id: int = 0,
                     minute: int = 0,
                     player_name: str = "",
                     team_name: str = "",
                     home_team: str = "",
                     body_part: str = "") -> Data:
    """Convert a StatsBomb 360 shot freeze frame into a PyG Data object."""
    positions, teams = [], []
    for p in freeze_frame:
        x_m, y_m = normalise_statsbomb(*p["location"])
        positions.append([x_m, y_m])
        teams.append(0.0 if p.get("teammate", True) else 1.0)

    positions = np.array(positions, dtype=np.float32)
    teams     = np.array(teams,     dtype=np.float32)

    # Node features: [x, y, teammate, actor, keeper]
    x_feat = np.column_stack([
        positions,
        teams,
        [float(p.get("actor",  False)) for p in freeze_frame],
        [float(p.get("keeper", False)) for p in freeze_frame],
    ]).astype(np.float32)

    edge_index_np = _edges_delaunay(positions)
    edge_attr_np  = _compute_edge_features(positions, teams, edge_index_np)

    dist_m, angle_rad = _shot_distance_angle(shot_loc)
    gk_feats          = _gk_pressure_features(freeze_frame, shot_loc)

    # Body-part flag: 1 = right foot, 0 = left foot / header / other
    # Used as a weak-foot proxy: combined with spatial context the model
    # can learn that right-foot shots from the left channel (and vice-versa)
    # are harder — without needing explicit preferred-foot lookup.
    is_right_foot = 1.0 if "Right" in body_part else 0.0

    data = Data(
        x          = torch.tensor(x_feat,        dtype=torch.float),
        edge_index = torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr  = torch.tensor(edge_attr_np,  dtype=torch.float),
        y          = torch.tensor([label],        dtype=torch.float),
        # metadata for benchmarking / model input
        sb_xg        = torch.tensor([sb_xg],                dtype=torch.float),
        shot_dist    = torch.tensor([dist_m],               dtype=torch.float),
        shot_angle   = torch.tensor([angle_rad],            dtype=torch.float),
        is_header    = torch.tensor([float(is_header)],     dtype=torch.float),
        is_open_play = torch.tensor([float(is_open_play)],  dtype=torch.float),
        # shot technique (8-dim one-hot: 0=unknown, 1-7=named techniques)
        technique      = torch.tensor(encode_technique(technique), dtype=torch.float),
        # goalkeeper-pressure features — original 3
        gk_dist        = torch.tensor([gk_feats["gk_dist"]],       dtype=torch.float),
        n_def_in_cone  = torch.tensor([gk_feats["n_def_in_cone"]], dtype=torch.float),
        gk_off_centre  = torch.tensor([gk_feats["gk_off_centre"]], dtype=torch.float),
        # NEW: 3 additional precision features (META_DIM 15 → 18)
        gk_perp_offset    = torch.tensor([gk_feats["gk_perp_offset"]],    dtype=torch.float),
        n_def_direct_line = torch.tensor([gk_feats["n_def_direct_line"]], dtype=torch.float),
        is_right_foot     = torch.tensor([is_right_foot],                 dtype=torch.float),
        # match-level context (for per-match report)
        match_id     = torch.tensor([match_id],  dtype=torch.long),
        minute       = torch.tensor([minute],    dtype=torch.long),
        player_name  = player_name,
        team_name    = team_name,
        home_team    = home_team,
    )
    return data


# ---------------------------------------------------------------------------
# Per-match processing
# ---------------------------------------------------------------------------

def process_match(match_id: int, home_team: str = "") -> tuple[list, dict]:
    stats = {"total_shots": 0, "with_frame": 0, "goals": 0, "no_goals": 0,
             "skipped_small": 0, "graphs": 0}

    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        print(f"    ERROR loading events: {e}")
        return [], stats

    shots = events[events["type"] == "Shot"].copy()
    stats["total_shots"] = len(shots)

    frames = _fetch_frames(match_id)
    if not frames:
        return [], stats

    dataset = []
    for _, row in shots.iterrows():
        ff = frames.get(row["id"])
        if ff is None:
            continue
        stats["with_frame"] += 1

        # Label: Goal=1, everything else=0
        outcome = row.get("shot_outcome", "")
        label = 1.0 if outcome == "Goal" else 0.0
        if label == 1.0:
            stats["goals"] += 1
        else:
            stats["no_goals"] += 1

        sb_xg        = float(row.get("shot_statsbomb_xg", 0.0) or 0.0)
        shot_loc     = row.get("location", [60.0, 40.0])
        body_part    = str(row.get("shot_body_part", "") or "")
        shot_type    = str(row.get("shot_type", "") or "")
        is_header    = "Head" in body_part
        is_open_play = shot_type == "Open Play"
        technique    = str(row.get("shot_technique", "") or "")
        minute       = int(row.get("minute", 0) or 0)
        player_name  = str(row.get("player", "") or "")
        team_name    = str(row.get("team", "") or "")

        if len(ff) < 3:
            stats["skipped_small"] += 1
            continue

        try:
            graph = build_shot_graph(
                ff, label, sb_xg, shot_loc, is_header, is_open_play,
                technique=technique, match_id=match_id, minute=minute,
                player_name=player_name, team_name=team_name, home_team=home_team,
                body_part=body_part,
            )
            graph = enrich_graph(graph, attacking_right=True, pressure_radius=5.0)
            dataset.append(graph)
            stats["graphs"] += 1
        except Exception:
            continue

    return dataset, stats


# ---------------------------------------------------------------------------
# Competition-level builder
# ---------------------------------------------------------------------------

def build_competition_shots(competition_ids: list[int], season_ids: list[int],
                             label: str) -> Path:
    output_path = PROCESSED_DIR / f"statsbomb_{label}_shot_graphs.pt"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"StatsBomb xG — shot freeze frames  ({label})")
    print(f"{'='*70}")

    all_graphs = []
    total = {"total_shots": 0, "with_frame": 0, "goals": 0, "no_goals": 0, "graphs": 0}

    for comp_id, season_id in zip(competition_ids, season_ids):
        matches = sb.matches(competition_id=comp_id, season_id=season_id)
        match_ids = matches["match_id"].tolist()
        print(f"\nCompetition {comp_id}/{season_id}: {len(match_ids)} matches")

        for i, mid in enumerate(match_ids, 1):
            row  = matches[matches["match_id"] == mid].iloc[0]
            home = str(row.get("home_team", "") or "")
            away = str(row.get("away_team", "") or "")
            print(f"  [{i:2d}/{len(match_ids)}] {home} vs {away} ... ", end="", flush=True)

            graphs, stats = process_match(mid, home_team=home)
            all_graphs.extend(graphs)
            for k in total:
                total[k] += stats.get(k, 0)

            goal_pct = 100 * stats["goals"] / max(stats["graphs"], 1)
            print(f"{stats['graphs']} graphs  (⚽ {stats['goals']} goals  {goal_pct:.0f}%)")

    # Summary
    print(f"\n{'─'*70}")
    print(f"Total shot graphs : {total['graphs']}")
    print(f"  Goals           : {total['goals']}  ({100*total['goals']/max(total['graphs'],1):.1f}%)")
    print(f"  No goals        : {total['no_goals']}")
    print(f"  Saved to        : {output_path}")

    torch.save(all_graphs, output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build xG shot graphs from StatsBomb 360 open data"
    )
    parser.add_argument("--competition", type=int, nargs="+", default=[43],
                        help="Competition ID(s) (default: 43=WC)")
    parser.add_argument("--season", type=int, nargs="+", default=[106],
                        help="Season ID(s) (default: 106=WC2022)")
    parser.add_argument("--label", type=str, default="",
                        help="Output file label (auto-generated if empty)")
    args = parser.parse_args()

    if len(args.competition) != len(args.season):
        print("ERROR: --competition and --season must have the same number of values")
        sys.exit(1)

    label = args.label or f"comp{'_'.join(map(str, args.competition))}"
    build_competition_shots(args.competition, args.season, label)


if __name__ == "__main__":
    main()
