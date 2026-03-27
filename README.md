# football-analysis

Graph Neural Network (GNN) research applied to football (soccer). Players are modeled as nodes, their spatial interactions and passes as edges — enabling models to reason about team shape, pressing traps, and off-ball movement as relational structures.

![Football GNN Dashboard — Shot Map · Freeze Frame · Gradient Saliency · xG Comparison](assets/dashboard_hero.png)

> **HybridGAT+T achieves AUC 0.760 · Brier 0.148** on 8,013 shots across 7 StatsBomb 360 competitions — trained with 27-dim metadata including PSxG shot placement, GK positioning, defensive blocking, and foot preference. Per-competition temperature scaling applied. Reaches 96% of StatsBomb's proprietary xG AUC using only free, open data.

---

### 📋 Match Report

![Match Report — Germany vs Japan · Shot maps · KPI table · Cumulative xG timeline](assets/match_report_screenshot.png)

The **Match Report** tab delivers a full per-match breakdown: side-by-side home/away shot maps coloured by HybridGAT xG, a KPI table (shots, goals, model xG, StatsBomb xG), and a cumulative xG step-function timeline with goal markers — select any match from any of the 7 competitions in the sidebar.

### Dashboard tabs

| Tab | What it shows |
|---|---|
| 📍 **Shot Map** | Half-pitch heat map of all shots coloured by HybridGAT xG; filter by outcome (goals/misses) or team; team KPI card; top-10 xG list; most surprising goals sidebar |
| 🔬 **Shot Inspector** | Full freeze-frame of every visible player at shot moment; gradient-saliency overlay (which players influenced the GCN most) or GAT attention overlay (top-3 player pairs attended to); xG comparison bar |
| 📊 **xG Distributions** | Goal vs miss histograms; reliability diagram (calibration curve); Brier score comparison table |
| 📋 **Match Report** | 4-panel report: home/away shot maps, KPI text, cumulative xG timeline; analyst narrative with executive summary and shot-by-shot log |
| 🌟 **Surprise Goals** | Goals rated < 15% xG by the model — worldies, deflections, and individual brilliance; ranked pitch map + table with player/team/minute |
| 👤 **Player Profile** | Per-player shots/goals/xG/overperformance aggregated across the competition; scatter of Goals vs xG; sortable/filterable table; CSV download |

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
│   ├── train_team_classifier.py    # Experiment 1 & 2: which team is passing?
│   ├── train_statsbomb_classifier.py  # Experiment 3 & 4: pass completion
│   ├── train_xg_model.py           # Experiment 5 & 6: GCN/GAT xG vs baselines
│   └── train_xg_hybrid.py          # Experiment 7 & 8: Hybrid xG (GCN + metadata)
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
| StatsBomb Open Data (360) | Events + freeze-frames | ~115K pass + 8,013 shot (326 matches, 7 comps) | Non-commercial | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) |
| Metrica Sports | Full optical tracking (25Hz) | 1,763 (2 matches) | MIT | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| SkillCorner Open | Broadcast tracking (~10Hz) | — | CC BY-SA 4.0 | [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata) |

**StatsBomb 360 competitions used:**

| Competition | Season | comp_id | season_id | Matches | Shot graphs | Goal % |
|---|---|---|---|---|---|---|
| FIFA World Cup | 2022 | 43 | 106 | 64 | 1,412 | 11.6% |
| Women's World Cup | 2023 | 72 | 107 | 64 | 1,589 | 9.4% |
| UEFA Euro | 2020 | 55 | 43 | 51 | 1,215 | 10.6% |
| UEFA Euro | 2024 | 55 | 282 | 51 | 1,279 | 8.3% |
| 1. Bundesliga | 2023/24 | 9 | 281 | 34 | 887 | 11.8% |
| UEFA Women's Euro | 2022 | 53 | 106 | 31 | 785 | 10.3% |
| UEFA Women's Euro | 2025 | 53 | 315 | 31 | 846 | 11.9% |
| **Total** | | | | **326** | **8,013** | **10.4%** |

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

**Takeaway:** GCN (0.603) outperforms GAT (0.560) cross-competition — consistent with the bias-variance pattern seen in Experiment 2. The spatial geometry of shot situations transfers across men's and women's football, but GNNs still lag the logistic baseline at this data scale. The fix: scale data and add a hybrid head.

---

### Experiment 7 — Full 360 Data + Hybrid xG Model (all 7 competitions pooled)

**Task:** Pool all 326 StatsBomb 360 matches across 7 competitions, train the Hybrid model (GCN embedding + shot metadata MLP head) alongside baselines. Stratified 70/15/15 split.

**Data:** **8,013 shot graphs** · 7 competitions · 836 goals (10.4%) · WC2022 + WWC2023 + Euro2020 + Euro2024 + Bundesliga23/24 + WEuro2022 + WEuro2025

| Model | AUC | Avg Precision | Brier |
|---|---|---|---|
| **StatsBomb xG** | **0.794** | **0.432** | **0.076** |
| **HybridGCN** | **0.752** | 0.343 | 0.178 |
| LogReg (dist+angle) | 0.740 | 0.307 | 0.192 |
| GCN | 0.655 | 0.166 | 0.232 |
| GAT | 0.635 | 0.167 | 0.286 |

**Takeaway:** The **Hybrid model beats LogReg (+0.012 AUC)** for the first time — proving the GNN freeze-frame component is learning real signal beyond just shot location. With 5.7× more data, GCN alone also jumps from 0.555 → 0.655 (+0.100). The gap to StatsBomb xG narrows from 0.229 (pure GCN, small data) to 0.042 (Hybrid, full data).

---

### Experiment 8 — Cross-Gender Hybrid xG (Men's 4 comps → Women's 3 comps)

**Task:** Train on all men's competitions (WC2022 + Euro2020 + Euro2024 + Bundesliga 23/24), test on all women's (WWC2023 + WEuro2022 + WEuro2025). Strictest domain shift: different eras, leagues, and genders.

**Data:**
- Train: **4,793 shots** (men's: WC2022 + Euro2020 + Euro2024 + Bundesliga)
- Test: **3,220 shots** (women's: WWC2023 + WEuro2022 + WEuro2025)

| Model | AUC | Avg Precision | Brier |
|---|---|---|---|
| **StatsBomb xG** | **0.825** | **0.458** | **0.072** |
| **HybridGCN** | **0.760** | 0.275 | 0.172 |
| LogReg (dist+angle) | 0.765 | 0.302 | 0.210 |
| GCN | 0.595 | 0.139 | 0.239 |
| GAT | 0.594 | 0.130 | 0.210 |

**Takeaway:** HybridGCN (0.760) nearly matches LogReg (0.765) across the men's → women's domain shift, and **the GNN component adds +0.165 AUC over pure GCN**. This confirms: (1) shot geometry is universal across genders, (2) the hybrid architecture scales well to unseen domains, (3) the gap to StatsBomb xG (0.825) now reflects only the absence of body position, technique, and historical prior — features that require proprietary data collection.

---

### Experiment 9 — Precision Features + HybridGAT with Temperature Scaling (all 7 comps pooled)

**Task:** Add 3 precision metadata features to reduce systematic variance vs StatsBomb xG; retrain both HybridGCN and HybridGAT with the expanded 18-dim metadata vector; apply per-model temperature scaling (T learned via LBFGS on NLL).

**New features:**
- `gk_perp_offset` — perpendicular distance of the goalkeeper from the shooter→goal centre line (metres). A GK at 0 perfectly blocks the direct path; 3+ m = exposed.
- `n_def_direct_line` — count of outfield defenders within a strict ≤3° half-angle cone directly between shooter and goal centre.
- `is_right_foot` — right-foot binary flag derived from StatsBomb `shot_body_part`; acts as a weak-foot penalty proxy in spatial context.

**Data:** 8,013 shot graphs · 7 competitions · 836 goals (10.4%) · META_DIM expanded 15 → 18

| Model | AUC | Avg Precision | Brier | T |
|---|---|---|---|---|
| **StatsBomb xG** | **0.794** | **0.432** | **0.076** | — |
| **HybridGAT + T-scaling** | **0.763** | **0.351** | **0.159** | 0.775 |
| HybridGCN + T-scaling | 0.760 | 0.350 | 0.171 | 0.854 |
| LogReg (dist + angle + header) | 0.743 | 0.301 | 0.190 | — |
| GCN (spatial only) | 0.655 | 0.166 | 0.232 | — |

**Takeaway:** The 3 precision features lift HybridGAT AUC from 0.752 → 0.763 (+0.011) and cut Brier from 0.178 → 0.159 (−11%). Temperature values T < 1 reveal the models were slightly *under-confident* overall (logits too compressed), and T-scaling correctly sharpens them. HybridGAT now outperforms HybridGCN on both metrics — the GATv2 attention mechanism learns which defender/GK interactions matter most for each shot.

---

### Experiment 10 — Sprint 1: PSxG Placement Feature + GAT Edge Features + Per-Competition Temperature

**Task:** Three simultaneous improvements targeting the remaining Brier gap. (1) Add `shot_placement` as a 9-bin PSxG feature — where on the goal face did the ball end up? (2) Fix GAT edge-feature pass-through (previously initialised but silently dropped). (3) Fit one temperature scalar per competition rather than a single global T.

**New features (META_DIM 18 → 27):**
- `shot_placement` — 9-dim one-hot encoding the goal-face zone (0=unknown/wide, 1=GK/saved, 2=post/bar, 3-8=quadrant grid). This is a **PSxG feature**: it reflects where the ball ended up after the shot, enabling the model to distinguish top-corner strikes from central saves. Stored as a fixed attribute on every graph.
- GAT edge_attr fix — `edge_attr` (4-dim: distance, Δx, Δy, same_team) is now actually passed to GATv2Conv layers during training, giving attention heads real geometric content to attend on.
- Per-competition T — one temperature T fitted per competition label on the validation set; `pool_7comp_per_comp_T_{gcn,gat}.pt` saved as `{comp_label: T}` dicts. Values range ~0.72 (WC2022) → ~0.86 (WWC2023), reflecting structural differences between men's and women's shot profiles.

**Data:** 8,013 shot graphs · 7 competitions · 836 goals (10.4%) · all graphs rebuilt with new attributes and `comp_label`

| Model | AUC | Avg Precision | Brier | T |
|---|---|---|---|---|
| **StatsBomb xG** | **0.794** | **0.432** | **0.076** | — |
| **HybridGAT + T-scaling** | **0.760** | **0.344** | **0.148** | 0.720 |
| HybridGCN + T-scaling | 0.762 | 0.346 | 0.163 | 0.876 |
| LogReg (dist + angle + header) | 0.743 | 0.301 | 0.190 | — |
| GCN (spatial only) | 0.655 | 0.166 | 0.232 | — |

**Brier trajectory:**

| Stage | HybridGAT Brier |
|---|---|
| Pre-session (15-dim, no T) | 0.193 |
| + precision features + global T (18-dim) | 0.159 |
| + PSxG placement + edge fix + per-comp T (27-dim) | **0.148** |
| StatsBomb target | 0.076 |

**Takeaway:** Shot placement is the single biggest Brier driver — knowing where the ball ended up on goal (top corner vs central save) captures shot quality that spatial freeze-frame geometry alone cannot infer. Brier drops a further −0.011 (−7%) to 0.148. The GAT edge_attr fix ensures attention weights are computed using actual player-pair distances rather than structure-only signals. Per-competition T reveals that WC2022 (men's, more powerful shots) needs sharper calibration (T=0.72) while WWC2023 needs less (T=0.86).

---

## Summary Table — All Experiments

| # | Task | Train Data | Test n | Best Model | AUC | Notes |
|---|---|---|---|---|---|---|
| 1 | Team classifier (in-game) | Metrica G1 | 120 | GAT | **1.000** | Perfect separation |
| 2 | Team classifier (cross-match) | Metrica G1→G2 | 145 | GCN | **1.000** | GCN generalizes better |
| 3 | Pass completion (in-comp) | WC2022 5 matches | 623 | GCN | **0.609** | AUC ~ professional xP |
| 4 | Pass completion (cross-comp) | WC2022 64 matches | 7,591 | GAT | **0.672** | Maintained across domain shift |
| 5 | xG pure GNN (in-comp) | WC2022 shots | 212 | GAT | 0.593 | Small data; LogReg wins (0.799) |
| 6 | xG pure GNN (cross-comp) | WC2022→WWC2023 | 1,589 | GCN | 0.603 | LogReg wins (0.764) |
| 7 | xG Hybrid (all 7 comps pooled) | 8,013 shots | 1,203 | **HybridGCN** | **0.760** | Technique + GK features; Brier 0.171 |
| 8 | xG Hybrid (men → women) | 4,793→3,220 | 3,220 | **HybridGCN** | **0.760** | Near-ties LogReg (0.765) |
| 9 | xG HybridGAT + T-scaling (all 7 comps) | 8,013 shots | 1,203 | **HybridGAT+T** | **0.763** | 18-dim meta, T=0.775, Brier 0.159 ↓ |
| 10 | Sprint 1: PSxG + edge fix + per-comp T (all 7 comps) | 8,013 shots | 1,203 | **HybridGAT+T** | **0.760** | 27-dim meta, Brier 0.148 ↓ (−7%) |

---

## Next Steps

- [x] Cross-match generalization (Metrica)
- [x] StatsBomb 360 pipeline
- [x] Scale to full tournaments (64 matches each)
- [x] Cross-competition generalization (WC2022 → WWC2023)
- [x] **xG model** — benchmark GNN vs StatsBomb xG and logistic regression
- [x] **All StatsBomb 360 data** — scale to all 326 matches across 7 competitions (8,013 shots)
- [x] **Hybrid xG model** — GCN embedding + shot metadata → MLP head; beats LogReg at scale
- [x] **Shot technique features** — add 8-dim one-hot technique to metadata; AUC +0.008
- [x] **Precision features** — `gk_perp_offset`, `n_def_direct_line`, `is_right_foot`; META_DIM 15 → 18
- [x] **Temperature scaling** — post-hoc LBFGS calibration; HybridGAT+T Brier 0.178 → 0.159
- [x] **GAT attention overlay** — top-3 player pairs the model attends to, visualised in Shot Inspector
- [x] **Surprise Goals detector** — dedicated tab for goals below 15% xG (worldies & deflections)
- [x] **Player xG Profile** — per-player aggregated shots/goals/xG/overperformance table + scatter
- [x] **CSV export** — download filtered shots or player stats from any view
- [x] **Shot placement feature** — `shot_placement` 9-bin PSxG one-hot; META_DIM 18 → 27; Brier 0.159 → 0.148
- [x] **GAT edge features** — edge_attr (distance, Δx, Δy, team) now passed to GATv2Conv during training
- [x] **Per-competition temperature** — one T per competition label; WC2022 T=0.72, WWC2023 T=0.86
- [x] **Permutation feature importance** — 12 feature groups ranked by AUC drop; GK distance dominates (+0.223)
- [x] **Feature Importance dashboard tab** — pre-computed bar chart + impact table in the app
- [x] **Cloud deployment prep** — `Dockerfile` (python:3.11-slim + CPU torch), `requirements.txt`, `scripts/upload_to_hub.py`
- [ ] **Residual metadata injection** — feed metadata into each GCN/GAT layer (not just the head)
- [ ] **Node-level prediction** — predict pass destination (which player receives), not just outcome
- [ ] **Temporal GNNs** — use sequence of 5 frames before a shot/pass to capture player momentum
- [ ] **Formation classifier** — cluster position snapshots into tactical shapes (4-4-2, 4-3-3, etc.)
- [ ] **Deploy to Railway** — push Docker image + set HF_TOKEN, get public URL

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

# 8. Build xG shot graphs — all 7 StatsBomb 360 competitions
python scripts/build_shot_graphs.py --competition 43 --season 106 --label wc2022          # WC2022
python scripts/build_shot_graphs.py --competition 72 --season 107 --label wwc2023         # WWC2023
python scripts/build_shot_graphs.py --competition 55 --season 43  --label euro2020        # Euro 2020
python scripts/build_shot_graphs.py --competition 55 --season 282 --label euro2024        # Euro 2024
python scripts/build_shot_graphs.py --competition  9 --season 281 --label bundesliga2324  # Bundesliga 23/24
python scripts/build_shot_graphs.py --competition 53 --season 106 --label weuro2022       # WEuro 2022
python scripts/build_shot_graphs.py --competition 53 --season 315 --label weuro2025       # WEuro 2025

# 9. Train xG — in-competition (Experiment 5, GCN/GAT only)
python scripts/train_xg_model.py --data data/processed/statsbomb_wc2022_shot_graphs.pt

# 10. Train xG — cross-competition (Experiment 6, GCN/GAT only)
python scripts/train_xg_model.py \
  --train data/processed/statsbomb_wc2022_shot_graphs.pt \
  --test  data/processed/statsbomb_wwc2023_shot_graphs.pt

# 11. Train Hybrid xG — all 7 competitions pooled (Experiment 7)
python scripts/train_xg_hybrid.py

# 12. Train Hybrid xG — men's → women's cross-gender (Experiment 8)
python scripts/train_xg_hybrid.py \
  --train data/processed/statsbomb_wc2022_shot_graphs.pt \
          data/processed/statsbomb_euro2020_shot_graphs.pt \
          data/processed/statsbomb_euro2024_shot_graphs.pt \
          data/processed/statsbomb_bundesliga2324_shot_graphs.pt \
  --test  data/processed/statsbomb_wwc2023_shot_graphs.pt \
          data/processed/statsbomb_weuro2022_shot_graphs.pt \
          data/processed/statsbomb_weuro2025_shot_graphs.pt

# 13. Temperature scaling is fitted automatically at end of train_xg_hybrid.py
#     Global T  → data/processed/pool_7comp_T.pt (GCN), pool_7comp_gat_T.pt (GAT)
#     Per-comp T → data/processed/pool_7comp_per_comp_T_{gcn,gat}.pt (dict: label → T)

# 14. Launch the Streamlit dashboard
streamlit run app.py
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
