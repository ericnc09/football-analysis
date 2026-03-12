#!/usr/bin/env python3
"""
build_graphs.py
---------------
Builds PyTorch Geometric graph datasets from Metrica tracking data.

Usage:
    python build_graphs.py --game 1        # Build graphs for Game 1 only
    python build_graphs.py --game 1 2      # Build for Games 1 and 2
    python build_graphs.py --all           # Build all available games
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.graph_builder import build_graph_from_tracking_frame, EdgeStrategy
from src.features import enrich_graph


# =============================================================================
# Constants
# =============================================================================

METRICA_DATA_DIR = REPO_ROOT / "data" / "raw" / "metrica" / "data"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FPS = 25.0  # Metrica tracking frame rate
METRICA_PITCH_LENGTH = 105.0  # metres
METRICA_PITCH_WIDTH = 68.0    # metres


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_tracking(filepath: str, team: str) -> pd.DataFrame:
    """
    Parse a Metrica tracking CSV into a tidy DataFrame.
    
    Returns columns:
      period, frame, time, player, jersey, x, y
    where x,y are in metres (scaled from [0,1]).
    """
    raw = pd.read_csv(filepath, header=[0, 1, 2])
    
    # The first 3 columns are Period, Frame, Time
    meta_cols = raw.columns[:3]
    period = raw[meta_cols[0]].values.astype(int)
    frame = raw[meta_cols[1]].values.astype(int)
    time = raw[meta_cols[2]].values.astype(float)
    
    # Collect all player data columns
    col_list = raw.columns.tolist()
    player_data = []
    
    i = 3
    while i < len(col_list) - 1:  # -1 to ensure we have both x and y
        lvl0, lvl1, lvl2 = col_list[i]
        # Check if this is a player column (has team name and Player label)
        if team in str(lvl0) and "Player" in str(lvl2):
            jersey = str(lvl1)
            player = str(lvl2)
            # X values are in column i, Y values are in column i+1
            x_vals = raw.iloc[:, i].values.astype(float) * METRICA_PITCH_LENGTH
            y_vals = raw.iloc[:, i + 1].values.astype(float) * METRICA_PITCH_WIDTH
            
            # Filter out NaN values
            valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                player_data.append({
                    'period': period[valid_indices],
                    'frame': frame[valid_indices],
                    'time': time[valid_indices],
                    'player': player,
                    'jersey': jersey,
                    'x': x_vals[valid_indices],
                    'y': y_vals[valid_indices],
                })
            i += 2  # Skip the y column since we already processed it
        else:
            i += 1
    
    # Build DataFrame more efficiently
    if not player_data:
        return pd.DataFrame()
    
    all_records = []
    for player_dict in player_data:
        n_frames = len(player_dict['period'])
        for j in range(n_frames):
            all_records.append({
                'period': player_dict['period'][j],
                'frame': player_dict['frame'][j],
                'time': player_dict['time'][j],
                'player': player_dict['player'],
                'jersey': player_dict['jersey'],
                'x': player_dict['x'][j],
                'y': player_dict['y'][j],
            })
    
    return pd.DataFrame(all_records)


def load_events(filepath: str) -> pd.DataFrame:
    """Load Metrica RawEventsData CSV."""
    events = pd.read_csv(filepath)
    events.columns = [
        c.strip().lower().replace(" ", "_").replace("[", "").replace("]", "")
        for c in events.columns
    ]
    return events


def add_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-player velocity (metres/second) via finite differences.
    Fills NaN at sequence boundaries with 0.
    """
    df = df.sort_values(["player", "frame"]).copy()
    df["vx"] = df.groupby("player")["x"].diff() * FPS
    df["vy"] = df.groupby("player")["y"].diff() * FPS
    df[["vx", "vy"]] = df[["vx", "vy"]].fillna(0.0)
    return df


def get_frame(
    home_tracking: pd.DataFrame,
    away_tracking: pd.DataFrame,
    frame_id: int,
) -> tuple:
    """
    Extract home/away player positions and velocities at a given frame.
    
    Returns:
        (home_pos, away_pos, home_vel, away_vel, home_players, away_players)
    """
    h_frame = home_tracking[home_tracking["frame"] == frame_id]
    a_frame = away_tracking[away_tracking["frame"] == frame_id]
    
    if len(h_frame) == 0 or len(a_frame) == 0:
        raise ValueError(f"Frame {frame_id} not found in tracking data")
    
    h_pos = h_frame[["x", "y"]].values
    a_pos = a_frame[["x", "y"]].values
    h_vel = h_frame[["vx", "vy"]].values if "vx" in h_frame.columns else np.zeros((len(h_frame), 2))
    a_vel = a_frame[["vx", "vy"]].values if "vx" in a_frame.columns else np.zeros((len(a_frame), 2))
    
    # Player identifiers (jersey numbers as strings)
    h_players = h_frame["jersey"].values  # Use jersey instead of player name
    a_players = a_frame["jersey"].values
    
    return h_pos, a_pos, h_vel, a_vel, h_players, a_players


# =============================================================================
# Graph Building
# =============================================================================

def build_pass_graphs(
    game_dir: Path,
    game_name: str,
    output_path: Path,
) -> int:
    """
    Build a dataset of pass-event graphs from Metrica tracking data.
    
    Args:
        game_dir: Path to game data directory (contains _RawTrackingData_*.csv, _RawEventsData.csv)
        game_name: Friendly name (e.g., "Sample_Game_1")
        output_path: Where to save the torch dataset
    
    Returns:
        Number of graphs created
    """
    print(f"\n{'='*70}")
    print(f"Building graphs for {game_name}")
    print(f"{'='*70}")
    print(f"Game directory: {game_dir}")
    print(f"Directory exists: {game_dir.exists()}")
    
    # Load data
    print(f"Loading tracking data from {game_dir}...")
    
    # Convert to Path if needed
    game_dir = Path(game_dir)
    
    # Find tracking files
    home_file = None
    away_file = None
    events_file = None
    
    print("Files in directory:")
    for f in game_dir.iterdir():
        print(f"  {f.name}")
        if "RawTrackingData_Home_Team" in f.name:
            home_file = f
            print(f"    -> Found HOME file")
        elif "RawTrackingData_Away_Team" in f.name:
            away_file = f
            print(f"    -> Found AWAY file")
        elif "RawEventsData" in f.name:
            events_file = f
            print(f"    -> Found EVENTS file")
    
    if not all([home_file, away_file, events_file]):
        print(f"ERROR: Missing required files in {game_dir}")
        print(f"  Home: {home_file}")
        print(f"  Away: {away_file}")
        print(f"  Events: {events_file}")
        return 0
    
    print(f"Loading tracking data...")
    home_tracking = load_tracking(str(home_file), team="Home")
    away_tracking = load_tracking(str(away_file), team="Away")
    events = load_events(str(events_file))
    
    print(f"  Loaded {len(home_tracking)} home frames, {len(away_tracking)} away frames")
    print(f"  Loaded {len(events)} events")
    
    # Add velocities
    home_tracking = add_velocities(home_tracking)
    away_tracking = add_velocities(away_tracking)
    
    # Extract pass events
    pass_events = events[events["type"] == "PASS"].reset_index(drop=True)
    print(f"  Found {len(pass_events)} pass events")
    
    # Build graphs
    dataset: list[Data] = []
    skipped = 0
    skip_reasons = {"no_data": 0, "frame_not_found": 0, "incomplete_players": 0, "graph_build_failed": 0, "enrichment_failed": 0}
    
    for idx, row in pass_events.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"    Processing pass {idx + 1}/{len(pass_events)}...")
        
        fid = int(row["start_frame"])
        
        try:
            h_pos, a_pos, h_vel, a_vel, h_pl, a_pl = get_frame(
                home_tracking, away_tracking, fid
            )
        except (KeyError, ValueError) as e:
            skip_reasons["frame_not_found"] += 1
            skipped += 1
            if idx < 5:  # Debug first few
                print(f"      Frame {fid} not found: {e}")
            continue
        
        # Skip frames with incomplete data
        if len(h_pos) < 10 or len(a_pos) < 10:
            skip_reasons["incomplete_players"] += 1
            skipped += 1
            if idx < 5:
                print(f"      Frame {fid}: only {len(h_pos)} home, {len(a_pos)} away players")
            continue
        
        # Label: 1.0 = pass completed (has a 'to' player), 0.0 = intercepted/out
        label = 1.0 if pd.notna(row.get("to", None)) else 0.0
        
        # Build graph
        try:
            graph = build_graph_from_tracking_frame(
                home_positions=h_pos,
                away_positions=a_pos,
                home_velocities=h_vel,
                away_velocities=a_vel,
                coord_system="tracab",  # already in metres
                strategy=EdgeStrategy.DELAUNAY,
                label=label,
            )
        except Exception as e:
            skip_reasons["graph_build_failed"] += 1
            if idx < 5:
                print(f"      Frame {fid}: Graph build failed: {e}")
            skipped += 1
            continue
        
        # Enrich graph
        try:
            graph = enrich_graph(graph, attacking_right=True, pressure_radius=5.0)
        except Exception as e:
            skip_reasons["enrichment_failed"] += 1
            if idx < 5:
                print(f"      Frame {fid}: Enrichment failed: {e}")
            skipped += 1
            continue
        
        dataset.append(graph)
    
    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    
    print(f"  ✓ Created {len(dataset)} graphs (skipped {skipped})")
    print(f"  ✓ Skip breakdown: {skip_reasons}")
    print(f"  ✓ Saved to {output_path}")
    
    return len(dataset)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Build PyTorch Geometric graph datasets from football tracking data"
    )
    parser.add_argument(
        "--game",
        type=int,
        nargs="+",
        default=[1],
        help="Game number(s) to process (default: 1)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available games",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Output directory (default: {PROCESSED_DIR})",
    )
    
    args = parser.parse_args()
    
    # Determine which games to process
    games_to_process = []
    
    if args.all:
        # Find all Sample_Game_N directories
        if METRICA_DATA_DIR.exists():
            for game_dir in sorted(METRICA_DATA_DIR.glob("Sample_Game_*")):
                game_num = game_dir.name.split("_")[-1]
                games_to_process.append((int(game_num), game_dir, game_dir.name))
        else:
            # Fallback to looking in data/Sample_Game_N
            data_root = REPO_ROOT / "data"
            for game_dir in sorted(data_root.glob("Sample_Game_*")):
                game_num = game_dir.name.split("_")[-1]
                games_to_process.append((int(game_num), game_dir, game_dir.name))
    else:
        # Process specified games
        for game_num in args.game:
            # Try metrica/data first, then fall back to data/
            game_dir = METRICA_DATA_DIR / f"Sample_Game_{game_num}"
            if not game_dir.exists():
                game_dir = REPO_ROOT / "data" / f"Sample_Game_{game_num}"
            
            if game_dir.exists():
                games_to_process.append((game_num, game_dir, f"Sample_Game_{game_num}"))
            else:
                print(f"WARNING: Game {game_num} data not found at {game_dir}")
    
    if not games_to_process:
        print("ERROR: No games found to process")
        sys.exit(1)
    
    # Process games
    total_graphs = 0
    for game_num, game_dir, game_name in games_to_process:
        output_file = args.output / f"metrica_game{game_num}_pass_graphs.pt"
        count = build_pass_graphs(game_dir, game_name, output_file)
        total_graphs += count
    
    print(f"\n{'='*70}")
    print(f"✓ Complete! Built {total_graphs} total graphs")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
