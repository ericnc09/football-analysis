# football-analysis

Graph Neural Network (GNN) research applied to football (soccer). Players are modeled as nodes, their spatial interactions and passes as edges — enabling models to reason about team shape, pressing traps, and off-ball movement as relational structures.

## Goal

Build a pipeline from raw football data → spatial graphs → GNN models that can:
- Classify team formations from player position snapshots
- Predict possession outcomes using pass network topology
- Detect pressing traps as subgraph patterns
- Value off-ball movement through graph attention

## Repo Structure

```
football-analysis/
├── data/
│   ├── raw/           # Downloaded datasets (gitignored)
│   └── processed/     # Graph objects (.pt, .pkl) (gitignored)
│
├── notebooks/
│   ├── 01_eda/                  # Data exploration
│   ├── 02_graph_construction/   # Events/tracking → graph objects
│   └── 03_gnn_experiments/      # Model training & evaluation
│
├── src/
│   ├── graph_builder.py   # Core: events/tracking → PyG Data objects
│   ├── features.py        # Node and edge feature engineering
│   └── models/            # GNN architectures (GCN, GAT, etc.)
│
└── legacy/            # Earlier visualization and scraping work
    ├── statsbomb/
    ├── fbref/
    └── devinpueler/
```

## Data Sources

| Dataset | Type | License | Link |
|---|---|---|---|
| StatsBomb Open Data | Events + 360 freeze-frames | Non-commercial | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) |
| Metrica Sports | Full optical tracking (25Hz) | MIT | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| Wyscout (Pappalardo) | Events, 1,941 matches | CC BY 4.0 | [figshare collection](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) |
| SkillCorner Open | Broadcast tracking (~10Hz) | CC BY-SA 4.0 | [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata) |

## GNN Formulation

```
G = (V, E, X_node, X_edge)

V  players on pitch at moment t
   node features: [x, y, vx, vy, team, role]

E  spatial proximity (<N meters), Delaunay triangulation,
   or directed pass connections
   edge features: [distance, angle, pass_success]

y  possession outcome, formation class, pressing trigger, xG
```

## Stack

- **PyTorch Geometric** — GNN framework
- **statsbombpy** — StatsBomb event data
- **kloppy** — Unified tracking data loader
- **socceraction / SPADL** — Standardized action format for event data
- **mplsoccer** — Pitch visualization

## Getting Started

```bash
pip install torch torch-geometric statsbombpy kloppy socceraction mplsoccer
```

Download StatsBomb open data:
```bash
git clone https://github.com/statsbomb/open-data.git data/raw/statsbomb
```

Download Metrica tracking data:
```bash
git clone https://github.com/metrica-sports/sample-data.git data/raw/metrica
```
