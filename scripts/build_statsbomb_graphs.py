#!/usr/bin/env python3
"""
build_statsbomb_graphs.py
--------------------------
Build PyG graph datasets from StatsBomb 360 open data.

Task: pass completion prediction
  - Node features: [x, y, teammate, actor, keeper]  (freeze-frame players)
  - Edge strategy: Delaunay triangulation
  - Label: 0 = pass completed, 1 = pass failed (Incomplete / Out)

Usage:
    # Build 5 WC2022 matches (quick smoke test)
    python scripts/build_statsbomb_graphs.py --matches 5

    # Build full World Cup 2022 (64 matches, ~45K graphs)
    python scripts/build_statsbomb_graphs.py --competition 43 --season 106

    # Build Women's World Cup 2023
    python scripts/build_statsbomb_graphs.py --competition 72 --season 107

    # List available competitions with 360 data
    python scripts/build_statsbomb_graphs.py --list
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import requests
from torch_geometric.data import Data
from statsbombpy import sb

from src.graph_builder import (
    build_graph_from_freeze_frame,
    EdgeStrategy,
    normalise_statsbomb,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from src.features import enrich_graph

PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Competitions known to have 360 data (competition_id, season_id, label)
COMPETITIONS_360 = [
    (43, 106, "wc2022"),
    (72, 107, "wwc2023"),
    (55, 282, "euro2024"),   # UEFA Euro 2024
    (2, 282, "pl2015_16"),   # Premier League (check availability)
]


# ---------------------------------------------------------------------------
# StatsBomb raw JSON helpers (avoids sb.frames() pandas 3.0 bug)
# ---------------------------------------------------------------------------

def _fetch_json(url: str, retries: int = 3) -> list:
    """Fetch a JSON list from GitHub raw URL with simple retry."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to fetch {url}: {e}") from e
    return []


def load_freeze_frames(match_id: int) -> dict:
    """
    Download 360 data for a match and return a dict:
        { event_uuid: freeze_frame_list }
    """
    url = (
        f"https://raw.githubusercontent.com/statsbomb/open-data/master"
        f"/data/three-sixty/{match_id}.json"
    )
    try:
        records = _fetch_json(url)
        return {r["event_uuid"]: r["freeze_frame"] for r in records}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Graph builder for StatsBomb pass events
# ---------------------------------------------------------------------------

def _node_features_from_freeze_frame(freeze_frame: list) -> np.ndarray:
    """
    Build (N, 5) node feature matrix from a StatsBomb freeze frame.
    Features per player: [x_m, y_m, teammate, actor, keeper]
    x_m, y_m are scaled to metres (105×68).
    """
    rows = []
    for p in freeze_frame:
        x_m, y_m = normalise_statsbomb(*p["location"])
        rows.append([
            x_m,
            y_m,
            float(p.get("teammate", False)),
            float(p.get("actor", False)),
            float(p.get("keeper", False)),
        ])
    return np.array(rows, dtype=np.float32)


def build_pass_graph(freeze_frame: list, label: float) -> Data:
    """
    Convert a StatsBomb 360 freeze frame (at a pass event) to a PyG Data object.
    Uses Delaunay triangulation for edges.
    """
    positions = []
    teams = []
    for p in freeze_frame:
        x_m, y_m = normalise_statsbomb(*p["location"])
        positions.append([x_m, y_m])
        # teammate=True → same team as actor → team flag 0; opponent → 1
        teams.append(0.0 if p.get("teammate", True) else 1.0)

    positions = np.array(positions, dtype=np.float32)
    teams = np.array(teams, dtype=np.float32)

    # Node features: [x, y, teammate, actor, keeper]
    x_feat = _node_features_from_freeze_frame(freeze_frame)

    # Edges via Delaunay
    from src.graph_builder import _edges_delaunay, _compute_edge_features
    edge_index_np = _edges_delaunay(positions)
    edge_attr_np = _compute_edge_features(positions, teams, edge_index_np)

    data = Data(
        x=torch.tensor(x_feat, dtype=torch.float),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr_np, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.float),
    )
    return data


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_match(match_id: int) -> tuple[list, dict]:
    """
    Build all pass-completion graphs for one match.

    Returns:
        (list_of_Data_objects, stats_dict)
    """
    stats = {
        "total_passes": 0,
        "with_frame": 0,
        "complete": 0,
        "incomplete": 0,
        "skipped_small": 0,
        "graphs": 0,
    }

    # Load events
    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        print(f"    ERROR loading events for match {match_id}: {e}")
        return [], stats

    passes = events[events["type"] == "Pass"].copy()
    stats["total_passes"] = len(passes)

    # Load 360 freeze frames
    frames = load_freeze_frames(match_id)
    if not frames:
        return [], stats

    dataset = []
    for _, row in passes.iterrows():
        event_id = row["id"]
        ff = frames.get(event_id)
        if ff is None:
            continue
        stats["with_frame"] += 1

        # Label: NaN outcome = completed successfully
        outcome = row.get("pass_outcome", float("nan"))
        if isinstance(outcome, float) and np.isnan(outcome):
            label = 0.0  # complete
            stats["complete"] += 1
        elif outcome in ("Incomplete", "Out"):
            label = 1.0  # failed
            stats["incomplete"] += 1
        else:
            # Unknown / off-target / etc. — skip
            continue

        # Need at least 6 players visible for a meaningful graph
        if len(ff) < 6:
            stats["skipped_small"] += 1
            continue

        try:
            graph = build_pass_graph(ff, label)
            # Enrich: add goal distance, angle, pressure
            # attacking_right=True is an approximation (ignores period/direction)
            graph = enrich_graph(graph, attacking_right=True, pressure_radius=5.0)
            dataset.append(graph)
            stats["graphs"] += 1
        except Exception:
            continue

    return dataset, stats


def build_competition_graphs(
    competition_id: int,
    season_id: int,
    label: str,
    max_matches: int | None = None,
) -> Path:
    """
    Build graphs for all matches in a competition and save to disk.

    Returns path to saved .pt file.
    """
    output_path = PROCESSED_DIR / f"statsbomb_{label}_pass_graphs.pt"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"StatsBomb 360 — {label}  (comp={competition_id}, season={season_id})")
    print(f"{'='*70}")

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].tolist()

    if max_matches:
        match_ids = match_ids[:max_matches]

    print(f"Processing {len(match_ids)} matches...")

    all_graphs = []
    total_stats = {
        "total_passes": 0,
        "with_frame": 0,
        "complete": 0,
        "incomplete": 0,
        "skipped_small": 0,
        "graphs": 0,
    }

    for i, mid in enumerate(match_ids, 1):
        row = matches[matches["match_id"] == mid].iloc[0]
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        print(f"  [{i:2d}/{len(match_ids)}] Match {mid}: {home} vs {away}", end=" ... ", flush=True)

        graphs, stats = process_match(mid)
        all_graphs.extend(graphs)

        for k in total_stats:
            total_stats[k] += stats[k]

        complete_pct = (
            100 * stats["complete"] / (stats["complete"] + stats["incomplete"])
            if (stats["complete"] + stats["incomplete"]) > 0 else 0
        )
        print(
            f"{stats['graphs']} graphs  "
            f"({stats['complete']}✓ / {stats['incomplete']}✗  {complete_pct:.0f}% complete)"
        )

    # Summary
    print(f"\n{'─'*70}")
    print(f"Total graphs built : {total_stats['graphs']}")
    print(f"  Complete passes  : {total_stats['complete']}")
    print(f"  Failed passes    : {total_stats['incomplete']}")
    if total_stats["complete"] + total_stats["incomplete"] > 0:
        pct = 100 * total_stats["complete"] / (total_stats["complete"] + total_stats["incomplete"])
        print(f"  Completion rate  : {pct:.1f}%")
    print(f"  Saved to         : {output_path}")

    torch.save(all_graphs, output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_competitions():
    """Print competitions that have 360 data available."""
    comps = sb.competitions()
    print("\nAll StatsBomb open-data competitions:")
    print(comps[["competition_id", "season_id", "competition_name", "season_name"]].to_string(index=False))
    print("\nKnown competitions with 360 freeze frames:")
    for cid, sid, lbl in COMPETITIONS_360:
        print(f"  --competition {cid} --season {sid}  ({lbl})")


def main():
    parser = argparse.ArgumentParser(
        description="Build pass-completion GNN datasets from StatsBomb 360 open data"
    )
    parser.add_argument("--list", action="store_true", help="List available competitions")
    parser.add_argument("--competition", type=int, default=43, help="Competition ID (default: 43 = FIFA World Cup)")
    parser.add_argument("--season", type=int, default=106, help="Season ID (default: 106 = WC 2022)")
    parser.add_argument("--label", type=str, default="", help="Output file label (auto-generated if empty)")
    parser.add_argument("--matches", type=int, default=None, help="Max number of matches (default: all)")
    args = parser.parse_args()

    if args.list:
        list_competitions()
        return

    # Auto-generate label if not provided
    label = args.label or f"comp{args.competition}_s{args.season}"

    build_competition_graphs(
        competition_id=args.competition,
        season_id=args.season,
        label=label,
        max_matches=args.matches,
    )


if __name__ == "__main__":
    main()
