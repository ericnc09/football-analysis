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
                     home_team: str = "") -> Data:
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
        technique    = torch.tensor(encode_technique(technique), dtype=torch.float),
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
