import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/eric/projects/football-analysis')

filepath = "/Users/eric/projects/football-analysis/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
team = "Home"
METRICA_PITCH_LENGTH = 105.0
METRICA_PITCH_WIDTH = 68.0

raw = pd.read_csv(filepath, header=[0, 1, 2])

# The first 3 columns are Period, Frame, Time
meta_cols = raw.columns[:3]
period = raw[meta_cols[0]].values.astype(int)
frame = raw[meta_cols[1]].values.astype(int)
time = raw[meta_cols[2]].values.astype(float)

records = []
# Player columns come in x/y pairs: (Team, Jersey, PlayerN) then (Unnamed, ..., Unnamed)
col_list = raw.columns.tolist()
i = 3
while i < len(col_list) - 1:  # -1 to ensure we have both x and y
    lvl0, lvl1, lvl2 = col_list[i]
    # Check if this is a player column (has team name and Player label)
    if team in str(lvl0) and "Player" in str(lvl2):
        jersey = str(lvl1)
        player = str(lvl2)
        # X values are in column i, Y values are in column i+1
        x_vals = raw.iloc[:, i].values.astype(float)
        y_vals = raw.iloc[:, i + 1].values.astype(float)
        print(f"Found player: {player} (jersey={jersey})")
        for f_idx in range(len(frame)):
            if not (np.isnan(x_vals[f_idx]) or np.isnan(y_vals[f_idx])):
                records.append({
                    "period": period[f_idx],
                    "frame": frame[f_idx],
                    "time": time[f_idx],
                    "player": player,
                    "jersey": jersey,
                    "x": x_vals[f_idx] * METRICA_PITCH_LENGTH,
                    "y": y_vals[f_idx] * METRICA_PITCH_WIDTH,
                })
        i += 2  # Skip the y column since we already processed it
    else:
        i += 1

df = pd.DataFrame(records)
print(f"\nLoaded {len(df)} records")
print(f"Players: {df['player'].nunique()}")
print(f"Frames: {df['frame'].nunique():,}")
print(f"\nSample records:")
print(df.head(10))
