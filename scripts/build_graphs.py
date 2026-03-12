"""
scripts/build_graphs.py
-----------------------
Reusable graph-building pipeline for any Metrica game.
Outputs a .pt file of PyG Data objects aligned to PASS events.

Usage:
    python scripts/build_graphs.py --game 1
    python scripts/build_graphs.py --game 2
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.graph_builder import (
    build_graph_from_tracking_frame,
    EdgeStrategy,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from src.features import enrich_graph

FPS = 25.0


def load_tracking(filepath: str, team: str) -> pd.DataFrame:
    raw = pd.read_csv(filepath, header=[0, 1, 2])
    meta = raw.columns[:3]
    period = raw[meta[0]].values.astype(int)
    frame  = raw[meta[1]].values.astype(int)
    time   = raw[meta[2]].values.astype(float)

    records = []
    col_list = raw.columns.tolist()
    i = 3
    while i < len(col_list):
        lvl0, lvl1, lvl2 = col_list[i]
        if team in str(lvl0) and "Player" in str(lvl2):
            x_vals = raw[col_list[i]].values.astype(float)
            y_vals = raw[col_list[i + 1]].values.astype(float)
            for f_idx in range(len(frame)):
                if not (np.isnan(x_vals[f_idx]) or np.isnan(y_vals[f_idx])):
                    records.append({
                        "period": period[f_idx],
                        "frame":  frame[f_idx],
                        "time":   time[f_idx],
                        "player": str(lvl2),
                        "jersey": str(lvl1),
                        "x":      x_vals[f_idx] * PITCH_LENGTH,
                        "y":      y_vals[f_idx] * PITCH_WIDTH,
                    })
            i += 2
        else:
            i += 1
    return pd.DataFrame(records)


def add_velocities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player", "frame"]).copy()
    df["vx"] = df.groupby("player")["x"].diff() * FPS
    df["vy"] = df.groupby("player")["y"].diff() * FPS
    df[["vx", "vy"]] = df[["vx", "vy"]].fillna(0.0)
    return df


def get_frame(home_df, away_df, frame_id):
    h = home_df[home_df["frame"] == frame_id].sort_values("jersey")
    a = away_df[away_df["frame"] == frame_id].sort_values("jersey")
    return (
        h[["x", "y"]].values,
        a[["x", "y"]].values,
        h[["vx", "vy"]].values,
        a[["vx", "vy"]].values,
    )


def build_game_graphs(game_num: int) -> list[Data]:
    game   = f"Sample_Game_{game_num}"
    data_dir = os.path.join(REPO_ROOT, "data", "raw", "metrica", "data", game)

    print(f"  Loading {game} tracking...")
    home_df = load_tracking(
        os.path.join(data_dir, f"{game}_RawTrackingData_Home_Team.csv"), "Home"
    )
    away_df = load_tracking(
        os.path.join(data_dir, f"{game}_RawTrackingData_Away_Team.csv"), "Away"
    )
    home_df = add_velocities(home_df)
    away_df = add_velocities(away_df)

    events = pd.read_csv(os.path.join(data_dir, f"{game}_RawEventsData.csv"))
    events.columns = [
        c.strip().lower().replace(" ", "_").replace("[", "").replace("]", "")
        for c in events.columns
    ]
    passes = events[events["type"] == "PASS"].reset_index(drop=True)
    print(f"  {len(passes)} PASS events  "
          f"({passes['team'].value_counts().to_dict()})")

    dataset, skipped = [], 0
    for _, row in passes.iterrows():
        fid = int(row["start_frame"])
        h_pos, a_pos, h_vel, a_vel = get_frame(home_df, away_df, fid)

        if len(h_pos) < 10 or len(a_pos) < 10:
            skipped += 1
            continue

        team_label = 0.0 if row["team"] == "Home" else 1.0
        g = build_graph_from_tracking_frame(
            h_pos, a_pos, h_vel, a_vel,
            coord_system="tracab",
            strategy=EdgeStrategy.DELAUNAY,
            label=team_label,
        )
        g = enrich_graph(g, attacking_right=(row["team"] == "Home"))
        # Drop team flag (col 4) — model must learn from spatial structure
        g.x = torch.cat([g.x[:, :4], g.x[:, 5:]], dim=1)
        g.game = game_num
        dataset.append(g)

    print(f"  Built {len(dataset)} graphs  ({skipped} skipped)")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, required=True, choices=[1, 2])
    args = parser.parse_args()

    print(f"\nBuilding graphs for Game {args.game}...")
    graphs = build_game_graphs(args.game)

    save_dir  = os.path.join(REPO_ROOT, "data", "processed")
    save_path = os.path.join(save_dir, f"metrica_game{args.game}_pass_graphs.pt")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(graphs, save_path)
    print(f"Saved {len(graphs)} graphs → {save_path}\n")


if __name__ == "__main__":
    main()
