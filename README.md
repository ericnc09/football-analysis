# football-analysis

Graph Neural Network (GNN) research applied to football (soccer). Players are modeled as nodes, their spatial interactions and passes as edges — enabling models to reason about team shape, pressing traps, and off-ball movement as relational structures.

## Goal

Build a pipeline from raw football data → spatial graphs → GNN models that can:
- Classify team formations from player position snapshots
- Predict possession outcomes using pass network topology
- Detect pressing traps as subgraph patterns
- Value off-ball movement through graph attention

## Product Vision

### The Problem
Football clubs spend millions on tactical analysis, but insights are locked in 
manual video review and static reports. Coaches need real-time, data-driven 
insights into team shape, pressing patterns, and player positioning — but 
current tools are either too complex (require data science teams) or too 
simplistic (basic stats).

### The Solution
An AI-powered tactical intelligence platform that:
- Automatically classifies team formations from any match footage
- Predicts possession loss risk in real-time during matches
- Identifies pressing trap patterns that lead to turnovers
- Values off-ball movement that traditional stats miss

### Target Users
1. **Professional Club Analysts** (Primary)
   - Need: Faster post-match analysis, pattern detection
   - Pain: Manual video review takes 8+ hours per match
   
2. **Coaching Staff** (Secondary)
   - Need: Tactical insights opponents' weaknesses
   - Pain: Can't process enough data during match preparation

3. **Performance Analysts** (Tertiary)
   - Need: Player evaluation beyond traditional stats
   - Pain: Off-ball movement is invisible to standard metrics

## Key Use Cases

### 1. Real-Time Formation Detection
**User Story:** As a tactical analyst, I want to identify opponent formation 
shifts during a match so I can alert the coaching staff to adjust tactics.

**Current Solution:** Manual observation (slow, subjective, error-prone)
**Our Solution:** GNN model processes player positions → formation classification in <1 second
**Value:** 10x faster than manual, 95%+ accuracy, objective

**Success Metrics:**
- Time to detection: <1 second (vs 30+ seconds manual)
- Accuracy: >95%
- Coach satisfaction: 4.5/5 (qualitative feedback)

### 2. Pressing Trap Prediction
**User Story:** As a coach, I want to know when opponent is vulnerable to 
pressing so I can trigger high-press tactics.

**Current Solution:** Coach intuition + video review
**Our Solution:** GNN detects spatial patterns indicating pressing opportunities
**Value:** Data-driven pressing triggers → 15% more turnovers in final third

**Success Metrics:**
- Precision: >80% (few false alarms)
- Recall: >70% (catch most opportunities)
- Turnovers created: +15% in testing

### 3. Off-Ball Movement Valuation
**User Story:** As a performance analyst, I want to quantify player value 
beyond goals/assists to identify undervalued talent.

**Current Solution:** Traditional stats (goals, assists, passes)
**Our Solution:** Graph attention reveals which players create space for teammates
**Value:** Identify undervalued players, improve scouting ROI

**Success Metrics:**
- Correlation with team performance: r > 0.7
- Scout adoption: 60% use in player evaluations
- Transfer value accuracy: ±10% vs market

## Product Roadmap

### Phase 1: MVP - Formation Classifier (Current)
**Timeline:** Q1 2025
**Goal:** Prove GNN approach works for tactical analysis
**Deliverables:**
- ✅ Team classifier (100% accuracy - DONE)
- 🏗️ Formation classifier (4-4-2, 4-3-3, etc.)
- 🏗️ Basic visualization dashboard
**Success Criteria:** Achieve >90% formation classification accuracy

### Phase 2: Beta - Pressing Detection (Next 3 months)
**Timeline:** Q2 2025
**Goal:** Add actionable tactical insights
**Deliverables:**
- Pressing trap detection model
- Real-time prediction pipeline
- Integration with broadcast tracking data
- Analyst feedback loop (5 beta testers)
**Success Criteria:** 
- 3 professional clubs piloting
- >80% pressing prediction precision
- <2 second latency for real-time use

### Phase 3: v1.0 - Multi-Club Platform (6 months)
**Timeline:** Q3 2025
**Goal:** Scale to multiple clubs, prove ROI
**Deliverables:**
- Multi-match, multi-team support
- API for third-party integrations
- Dashboard for non-technical users
- Automated match report generation
**Success Criteria:**
- 10 paying clubs
- $50K ARR
- 4.5/5 customer satisfaction

### Phase 4: Advanced Features (Long-term)
- Player value estimation
- Transfer market recommendations
- Youth academy talent identification
- Injury risk prediction from movement patterns

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

## Results

### Experiment 1 — Team Classifier (Metrica Sample Game 1)

**Task:** Given a snapshot of all 22 players at the moment a pass occurs, predict which team is making the pass (Home=0, Away=1) — using spatial structure alone, no team identity in node features.

**Data:** 799 pass-event graphs from Metrica Sample Game 1 · 437 Home / 362 Away · chronological 70/15/15 split

**Node features:** `[x, y, vx, vy, dist_atk_goal, dist_def_goal, angle_atk, pressure]`
**Edge features:** `[distance, Δx, Δy, same_team, pass_angle, vel_alignment]` · Delaunay triangulation

| Model | Test Acc | Test AUC | Params |
|---|---|---|---|
| GCN (3-layer, hidden=64) | 0.868 | 0.998 | 11,009 |
| **GAT (3-layer, hidden=32×4 heads, edge features)** | **1.000** | **1.000** | 46,433 |

**Takeaway:** GCN learns formation shape and field position well enough for 86.8% accuracy without any team identifier. GAT achieves perfect separation by attending over edge features (pass direction, velocity alignment) — direct evidence that edge-level attention captures tactically meaningful player relationships.

## Next Steps

- [ ] **Cross-match generalization** — train on Game 1, test on Game 2 (same teams, different game)
- [ ] **Harder label: possession loss** — predict whether possession is lost within the next 3 events (~94% base rate, requires oversampling)
- [ ] **Attention weight visualization** — which player pairs does the GAT focus on for each prediction?
- [ ] **StatsBomb 360 pipeline** — replicate with freeze-frame event graphs for broader match coverage
- [ ] **Formation classifier** — cluster player position snapshots into tactical shapes (4-4-2, 4-3-3, etc.)

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
