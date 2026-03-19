# football-analysis

Graph Neural Network (GNN) research applied to football (soccer). Players are modeled as nodes, their spatial interactions and passes as edges — enabling models to reason about team shape, pressing traps, and off-ball movement as relational structures.

## Goal

Build a pipeline from raw football data → spatial graphs → GNN models that can:
- Predict pass completion from spatial freeze-frames
- Classify which team is in possession from formation shape alone
- Detect pressing traps as subgraph patterns
- Value off-ball movement through graph attention

## Repo Structure

```
football-analysis/
├── data/
│   ├── raw/           # Downloaded datasets (gitignored)
│   └── processed/     # Graph objects (.pt) + model weights + plots (gitignored)
│
├── notebooks/
│   ├── 01_eda/                  # Data exploration
│   ├── 02_graph_construction/   # Events/tracking → graph objects
│   └── 03_gnn_experiments/      # Model training & evaluation
│
├── scripts/
│   ├── download_data.py            # Download Metrica CSV files from GitHub
│   ├── build_graphs.py             # Metrica tracking → PyG graph datasets
│   ├── build_statsbomb_graphs.py   # StatsBomb 360 → pass-completion graphs
│   ├── build_shot_graphs.py        # StatsBomb 360 → xG shot graphs
│   ├── train_team_classifier.py    # Experiment 1: which team is passing?
│   ├── train_statsbomb_classifier.py  # Experiment 3 & 4: pass completion
│   └── train_xg_model.py           # Experiment 5 & 6: xG vs StatsBomb baseline
│
├── src/
│   ├── graph_builder.py   # Core: events/tracking → PyG Data objects
│   ├── features.py        # Node and edge feature engineering
│   └── models/            # GNN architectures (GCN, GAT)
│
└── legacy/            # Earlier visualization and scraping work
    ├── statsbomb/
    ├── fbref/
    └── devinpueler/
```

## Data Sources

| Dataset | Type | Graphs Built | License | Link |
|---|---|---|---|---|
| StatsBomb Open Data | Events + 360 freeze-frames | ~107K (128 matches) | Non-commercial | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) |
| Metrica Sports | Full optical tracking (25Hz) | 1,763 (2 matches) | MIT | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| SkillCorner Open | Broadcast tracking (~10Hz) | — | CC BY-SA 4.0 | [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata) |

## GNN Formulation

```
G = (V, E, X_node, X_edge)

V  players visible on pitch at moment t (freeze-frame or tracking snapshot)
   node features: [x, y, teammate, actor, keeper,
                   dist_atk_goal, dist_def_goal, angle_atk, pressure]

E  Delaunay triangulation of player positions
   edge features: [distance, Δx, Δy, same_team, pass_angle, vel_alignment]

y  pass_completion (0=complete, 1=failed)  ← StatsBomb experiments
   possession_team (0=home, 1=away)        ← Metrica experiments
```

## Results

### Experiment 1 — Team Classifier (Metrica, in-game)

**Task:** Predict which team is making the pass (Home / Away) from spatial structure alone — team identity deliberately excluded from node features.

**Data:** 799 pass graphs · Metrica Sample Game 1 · 437 Home / 362 Away · 70/15/15 chronological split

| Model | Test Acc | Test AUC | Params |
|---|---|---|---|
| GCN (3-layer, hidden=64) | 0.868 | 0.998 | 11,009 |
| **GAT (3-layer, 4 heads, edge features)** | **1.000** | **1.000** | 46,433 |

**Takeaway:** GAT perfectly separates formations through edge attention (velocity alignment, pass direction). GCN reaches 86.8% with graph topology alone — both well above the 54.7% majority baseline.

---

### Experiment 2 — Cross-Match Generalization (Metrica Game 1 → Game 2)

**Task:** Same team classifier, but trained on Game 1 and tested on a completely held-out Game 2.

**Data:** Train 799 graphs (Game 1) · Test 964 graphs (Game 2)

| Model | Test Acc | Test AUC |
|---|---|---|
| **GCN** | **0.992** | **1.000** |
| GAT | 0.940 | 1.000 |

**Takeaway:** GCN generalizes *better* than GAT across matches — a classic bias-variance reversal. GAT's edge attention overfits to Game 1's specific patterns; GCN's simpler aggregation transfers more robustly.

---

### Experiment 3 — Pass Completion Classifier (StatsBomb WC2022, in-competition)

**Task:** Given the spatial freeze-frame at a pass event, predict whether the pass succeeds (label 0) or fails (label 1). 80/20 class imbalance handled with weighted BCE loss.

**Data:** 4,152 graphs · 5 WC2022 matches · 79.9% complete · 70/15/15 chronological split

| Model | Test Acc | Test AUC | Macro F1 |
|---|---|---|---|
| **GCN** | 0.685 | **0.609** | **0.602** |
| GAT | 0.737 | 0.597 | 0.563 |

**Takeaway:** GCN is better calibrated for the minority class (failed passes) at this scale. GAT inflates accuracy by defaulting to the majority class. AUC ~0.61 is consistent with professional expected-pass (xP) models — the task is genuinely hard.

---

### Experiment 4 — Cross-Competition Generalization (WC2022 → WWC2023)

**Task:** Train pass completion model on FIFA World Cup 2022 (men's), test on FIFA Women's World Cup 2023. Models never see women's football during training.

**Data:**
- Train: **56,298 graphs** · 64 WC2022 matches · 83.2% complete
- Test: **50,609 graphs** · 64 WWC2023 matches · 74.6% complete

| Model | Test Acc | Test AUC | Macro F1 | Complete F1 | Incomplete F1 |
|---|---|---|---|---|---|
| GCN | 0.647 | 0.610 | 0.570 | 0.75 | 0.39 |
| **GAT** | 0.484 | **0.672** | 0.483 | 0.51 | 0.46 |

**Takeaway:** GAT achieves the higher AUC (0.672 vs 0.610) cross-competition, but with a very different decision profile — it aggressively flags passes as risky (86% recall on failed passes) which fits the higher failure rate in WWC2023. GCN is more conservative and balanced. Crucially, **AUC is maintained across the men's → women's domain shift**, demonstrating that spatial pass-completion geometry is universal.

---

### Experiment 5 — xG Model (WC2022, in-competition)

**Task:** Predict whether a shot results in a goal from the 360° freeze-frame. Benchmarked against StatsBomb's published xG, logistic regression on shot geometry, and a majority-class baseline.

**Data:** 1,412 shot graphs · 64 WC2022 matches · 13.1% goals · 70/15/15 chronological split

| Model | AUC | Avg Precision | Brier |
|---|---|---|---|
| **StatsBomb xG** | **0.822** | **0.396** | **0.099** |
| LogReg (dist+angle) | 0.799 | 0.355 | 0.105 |
| GAT | 0.593 | 0.167 | 0.129 |
| GCN | 0.555 | 0.154 | 0.130 |
| Majority baseline | 0.500 | 0.131 | 0.114 |

**Takeaway:** With ~1,400 shots and only 185 goals, the GNNs underfit. Shot distance and angle dominate the signal, and a graph model needs 10K+ samples to learn nuanced blocker positioning from freeze frames. A 2-feature logistic regression (0.799 AUC) already captures most of the geometry — StatsBomb's benchmark (0.822) incorporates additional features including shot technique, body position, and historical context.

---

### Experiment 6 — xG Cross-Competition (WC2022 → WWC2023)

**Task:** Train the xG model on WC2022 shots, test on WWC2023. Domain shift: men's → women's football, different shot profiles and goal rates.

**Data:**
- Train: **1,412 graphs** · WC2022 · 13.1% goals
- Test: **1,589 graphs** · WWC2023 · 11.1% goals

| Model | AUC | Avg Precision | Brier |
|---|---|---|---|
| **StatsBomb xG** | **0.818** | **0.354** | **0.088** |
| LogReg (dist+angle) | 0.764 | 0.294 | 0.095 |
| GCN | 0.603 | 0.167 | 0.114 |
| GAT | 0.560 | 0.148 | 0.118 |

**Takeaway:** GCN (0.603) outperforms GAT (0.560) cross-competition — consistent with the bias-variance pattern seen in Experiment 2. The spatial geometry of shot situations transfers across men's and women's football, but GNNs still lag the logistic baseline at this data scale. The natural next step is a **hybrid model**: GCN graph embedding concatenated with shot metadata (distance, angle, body part) — the approach used by commercial xG providers.

---

## Summary Table — All Experiments

| # | Task | Train Data | Test Data | Best GNN AUC | Baseline AUC | Notes |
|---|---|---|---|---|---|---|
| 1 | Team classifier (in-game) | Metrica G1 | Metrica G1 | **1.000** (GAT) | 0.547 maj. | Perfect separation |
| 2 | Team classifier (cross-match) | Metrica G1 | Metrica G2 | **1.000** (GCN) | 0.547 maj. | GCN generalizes better |
| 3 | Pass completion (in-comp) | WC2022 5 matches | WC2022 5 matches | **0.609** (GCN) | ~0.5 | AUC ~ professional xP models |
| 4 | Pass completion (cross-comp) | WC2022 64 matches | WWC2023 64 matches | **0.672** (GAT) | ~0.5 | AUC maintained across domain shift |
| 5 | xG (in-competition) | WC2022 shots | WC2022 shots | 0.593 (GAT) | **0.799** LogReg | Small data; distance dominates |
| 6 | xG (cross-competition) | WC2022 shots | WWC2023 shots | 0.603 (GCN) | **0.764** LogReg | GCN beats GAT cross-domain |

---

## Next Steps

- [x] Cross-match generalization (Metrica)
- [x] StatsBomb 360 pipeline
- [x] Scale to full tournaments (64 matches each)
- [x] Cross-competition generalization (WC2022 → WWC2023)
- [x] **xG model** — benchmark GNN vs StatsBomb xG and logistic regression on shot freeze-frames
- [ ] **Hybrid xG model** — GCN graph embedding + shot metadata (dist, angle, body part) → joint head
- [ ] **GAT attention visualization** — plot which player pairs the model attends to for predicted failures
- [ ] **Node-level prediction** — predict pass destination (which player receives), not just outcome
- [ ] **Temporal GNNs** — use sequence of 5 frames before a shot/pass to capture player momentum
- [ ] **Formation classifier** — cluster position snapshots into tactical shapes (4-4-2, 4-3-3, etc.)

---

## Reproducing the Results

```bash
# 1. Install dependencies
pip install torch torch-geometric statsbombpy kloppy mplsoccer scikit-learn

# 2. Download Metrica tracking data
python scripts/download_data.py --metrica --game 1 2

# 3. Build Metrica graphs
python scripts/build_graphs.py --game 1 2

# 4. Train Metrica team classifier (Experiments 1 & 2)
python scripts/train_team_classifier.py

# 5. Build StatsBomb pass graphs (uses statsbombpy, no pre-download needed)
python scripts/build_statsbomb_graphs.py --competition 43 --season 106 --label wc2022   # WC2022
python scripts/build_statsbomb_graphs.py --competition 72 --season 107 --label wwc2023  # WWC2023

# 6. Train pass completion — in-competition (Experiment 3)
python scripts/train_statsbomb_classifier.py --data data/processed/statsbomb_wc2022_pass_graphs.pt

# 7. Train pass completion — cross-competition (Experiment 4)
python scripts/train_statsbomb_classifier.py \
  --train data/processed/statsbomb_wc2022_pass_graphs.pt \
  --test  data/processed/statsbomb_wwc2023_pass_graphs.pt

# 8. Build xG shot graphs
python scripts/build_shot_graphs.py --competition 43 --season 106 --label wc2022   # WC2022
python scripts/build_shot_graphs.py --competition 72 --season 107 --label wwc2023  # WWC2023

# 9. Train xG benchmark — in-competition (Experiment 5)
python scripts/train_xg_model.py --data data/processed/statsbomb_wc2022_shot_graphs.pt

# 10. Train xG benchmark — cross-competition (Experiment 6)
python scripts/train_xg_model.py \
  --train data/processed/statsbomb_wc2022_shot_graphs.pt \
  --test  data/processed/statsbomb_wwc2023_shot_graphs.pt
```

## Stack

| Library | Role |
|---|---|
| **PyTorch Geometric** | GNN layers (GCNConv, GATv2Conv), DataLoader |
| **statsbombpy** | StatsBomb event + 360 freeze-frame data |
| **kloppy** | Unified tracking data loader (Metrica Game 3 / EPTS) |
| **mplsoccer** | Pitch visualization |
| **scikit-learn** | AUC-ROC, classification reports |
| **scipy** | Delaunay triangulation for edge construction |
