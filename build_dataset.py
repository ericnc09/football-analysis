#!/usr/bin/env python3
"""Generate and save the graph dataset from Metrica Sports tracking data."""

import sys
import os
sys.path.insert(0, os.path.expanduser("~/projects/football-analysis"))
os.chdir(os.path.expanduser("~/projects/football-analysis"))

import numpy as np
import pandas as pd
import torch
from src.graph_builder import build_graph_from_tracking_frame, EdgeStrategy, PITCH_LENGTH, PITCH_WIDTH
from src.features import enrich_graph


def load_tracking(filepath: str, team: str) -> pd.DataFrame:
    """Parse a Metrica tracking CSV into a tidy DataFrame."""
    raw = pd.read_csv(filepath, header=[0, 1, 2])
    meta_cols = raw.columns[:3]
    period = raw[meta_cols[0]].values.astype(int)
    frame = raw[meta_cols[1]].values.astype(int)
    time = raw[meta_cols[2]].values.astype(float)

    records = []
    col_list = raw.columns.tolist()
    i = 3
    while i < len(col_list):
        lvl0, lvl1, lvl2 = col_list[i]
        if team in str(lvl0) and "Player" in str(lvl2):
            jersey = str(lvl1)
            player = str(lvl2)
            x_vals = raw[col_list[i]].values.astype(float)
            y_vals = raw[col_list[i + 1]].values.astype(float)
            for f_idx in range(len(frame)):
                if not (np.isnan(x_vals[f_idx]) or np.isnan(y_vals[f_idx])):
                    records.append({
                        "period": period[f_idx], "frame": frame[f_idx], "time": time[f_idx],
                        "player": player, "jersey": jersey,
                        "x": x_vals[f_idx] * PITCH_LENGTH,
                        "y": y_vals[f_idx] * PITCH_WIDTH,
                    })
            i += 2
        else:
            i += 1
    return pd.DataFrame(records)


def get_frame(home_tracking, away_tracking, fid):
    h = home_tracking[home_tracking["frame"] == fid]
    a = away_tracking[away_tracking["frame"] == fid]
    h_vel = np.zeros_like(h[["x", "y"]].values)
    a_vel = np.zeros_like(a[["x", "y"]].values)
    return (h[["x", "y"]].values, a[["x", "y"]].values, h_vel, a_vel, h["player"].values, a["player"].values)


print(f"Working directory: {os.getcwd()}\n")

# Load data
print("1. Loading tracking data...")
home_tracking = load_tracking("data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv", "Home")
away_tracking = load_tracking("data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv", "Away")

print("2. Loading events data...")
events = pd.read_csv("data/Sample_Game_1/Sample_Game_1_RawEventsData.csv")
# Clean column names (lowercase, replace spaces with underscores)
events.columns = [c.strip().lower().replace(" ", "_").replace("[", "").replace("]", "") for c in events.columns]

print("3. Extracting pass events...")
pass_events = events[events["type"] == "PASS"].reset_index(drop=True)
print(f"   Found {len(pass_events)} passes\n")

print("4. Building graphs...")
dataset = []
skipped = 0

for idx, (_, row) in enumerate(pass_events.iterrows()):
    if (idx + 1) % 100 == 0:
        print(f"   {idx + 1}/{len(pass_events)}")
    
    fid = int(row["start_frame"])
    try:
        h_pos, a_pos, h_vel, a_vel, h_pl, a_pl = get_frame(home_tracking, away_tracking, fid)
        if len(h_pos) < 10 or len(a_pos) < 10:
            skipped += 1
            continue
        
        label = 0.0 if pd.isna(row.get("to", float("nan"))) else 1.0
        g = build_graph_from_tracking_frame(h_pos, a_pos, h_vel, a_vel, coord_system="tracab", strategy=EdgeStrategy.DELAUNAY, label=label)
        g = enrich_graph(g, attacking_right=(row["team"] == "Home"))
        dataset.append(g)
    except Exception as e:
        skipped += 1

print(f"\n5. Dataset stats:")
print(f"   Graphs: {len(dataset)}, Skipped: {skipped}")
print(f"   Passes: {sum(int(g.y.item()) for g in dataset)} successful\n")

print("6. Saving...")
os.makedirs("data/processed", exist_ok=True)
save_path = os.path.abspath("data/processed/metrica_game1_pass_graphs.pt")
torch.save(dataset, save_path)

if os.path.exists(save_path):
    print(f"   ✓ Saved to {save_path}")
    print(f"   ✓ Size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
else:
    print(f"   ✗ Failed to save!")
    sys.exit(1)
