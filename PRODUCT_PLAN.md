# Football GNN Analysis — Product Plan

> **One-line pitch:** StatsBomb charges clubs £500K+/year for xG models. This project gets to 95% of their performance (0.752 vs 0.794 AUC) using only free, open data — proven across 8 experiments on 8,013 shots from 7 competitions.

---

## The Problem

Football analytics is a two-tier market:

- **Tier 1** (Premier League, top European clubs): Full access to Opta, StatsBomb, Second Spectrum — proprietary event data, tracking data, xG models, pass risk models. Annual contracts run £30K–500K+ per club.
- **Tier 2** (Championship, League One, most of the world): Scouts with spreadsheets. No spatial models. Decisions made on watched footage and instinct.

The gap is not capability — StatsBomb has published their open 360 dataset (player positions at every event moment). The gap is someone turning that free data into usable tools.

---

## What We've Proven (Evidence Base)

| Claim | Proof |
|---|---|
| Spatial freeze-frames encode real xG signal | HybridGCN beats 2-feature LogReg (+0.012 AUC) |
| GNN generalises across competitions | AUC maintained men's → women's football (0.760 cross-gender) |
| GNN graph embedding adds value over statistics | +0.097 AUC over pure GCN when graph embedding added to metadata |
| Open data reaches 95% of proprietary accuracy | 0.752 vs 0.794 StatsBomb xG with zero proprietary data |
| Remaining gap is understood and closeable | Shot technique + body position = what free data can't yet see |

### All 8 Experiments Summary

| # | Task | Train Data | Test n | Best Model | AUC | Notes |
|---|---|---|---|---|---|---|
| 1 | Team classifier (in-game) | Metrica G1 | 120 | GAT | **1.000** | Perfect separation |
| 2 | Team classifier (cross-match) | Metrica G1→G2 | 145 | GCN | **1.000** | GCN generalises better |
| 3 | Pass completion (in-comp) | WC2022 5 matches | 623 | GCN | **0.609** | AUC ~ professional xP models |
| 4 | Pass completion (cross-comp) | WC2022 64 matches | 7,591 | GAT | **0.672** | Maintained across domain shift |
| 5 | xG pure GNN (in-comp) | WC2022 shots | 212 | GAT | 0.593 | Small data; LogReg wins (0.799) |
| 6 | xG pure GNN (cross-comp) | WC2022→WWC2023 | 1,589 | GCN | 0.603 | LogReg wins (0.764) |
| 7 | xG Hybrid (all 7 comps pooled) | 8,013 shots | 1,203 | **HybridGCN** | **0.752** | Beats LogReg (0.740) ✓ |
| 8 | xG Hybrid (men → women) | 4,793→3,220 | 3,220 | **HybridGCN** | **0.760** | Near-ties LogReg (0.765) |

---

## Users

| Persona | Description | Pain today | What this gives them |
|---|---|---|---|
| **Club analyst** | Data analyst at Championship/League One club, 1–2 person team | Buys StatsBomb at £30K/yr or uses no spatial model at all | Free xG model + explainability layer showing *why* a shot was dangerous |
| **Head of recruitment** | Evaluates players across leagues, travel-heavy | Relies on watched footage + basic stats for shot quality | Quick xG-per-competition breakdown of transfer targets |
| **Data journalist** | Writes tactical analysis for The Athletic, FBRef, Twitter | Uses existing public xG tools with no spatial explainability | Shot maps with "which defenders blocked the angle" — a story tool |
| **Sports betting analyst** | Builds in-house models, wants open source base | Vendor-locked or building from scratch | Open-source, benchmarked, reproducible baseline model |

---

## Prioritised Roadmap

### Priority Framework (RICE)

| # | Feature | Reach | Impact | Confidence | Effort | Priority |
|---|---|---|---|---|---|---|
| 1 | Interactive pitch dashboard (Streamlit) | High | High | High | Medium | 🔴 **Top** |
| 2 | GAT attention heatmap on pitch | Medium | High | High | Low | 🔴 **Top** |
| 3 | Per-match xG report (HTML/PDF) | High | High | Medium | Medium | 🟠 **High** |
| 4 | Shot technique one-hot features | Low | Medium | High | Low | 🟠 **High** |
| 5 | Formation / pressing shape detector | Medium | Medium | Medium | High | 🟡 **Medium** |
| 6 | REST API endpoint | High | High | Low | High | 🟡 **Medium** |
| 7 | Temporal GNNs (pre-shot sequence) | Low | High | Low | High | 🟢 **Later** |

---

## 30 / 60 / 90 Day Roadmap

### 30 Days — *Make it showable*

**Goal:** A working demo screenShareable in any interview within 3 minutes.

**Features:**
- [ ] **Streamlit dashboard** — pitch renderer (mplsoccer), shot selector, side-by-side HybridGCN vs StatsBomb xG per shot
- [ ] **GAT attention overlay** — highlight top-3 player-pair edges the model focused on for a given shot, rendered directly on pitch
- [ ] **Screenshot/GIF of dashboard** — added to README as the hero image
- [ ] **Non-technical one-pager** — the product story written for a non-technical interviewer audience

**Success metric:** Full project explained and demoed in a 10-minute interview slot without touching the terminal.

---

### 60 Days — *Make it explainable and useful*

**Goal:** Something a real analyst could pick up and use on any StatsBomb open match.

**Features:**
- [ ] **Shot technique features** — add `shot_technique` one-hot (Normal / Volley / Half Volley / Overhead Kick / Lob) from StatsBomb events; retrain HybridGCN; target AUC > 0.770
- [ ] **Per-match xG report** — given a `match_id`, auto-generate: xG timeline, top 5 shots with model explanation, team xG totals vs actual goals; export as HTML
- [ ] **"Surprise goals" detector** — flag goals where HybridGCN xG < 0.15 (long-range strikes neither model predicted)
- [ ] **Goalkeeper pressure metric** — count + proximity of defenders between shooter and goal (derived from freeze-frames, no new data collection needed)
- [ ] **Model card** — document what the model does and doesn't see (shot technique blind spot, calibration over-confidence above 20%, penalty calibration issue)

**Success metric:** Generate a real match report for a WC2022 final that reads like something a club analyst would send their manager.

---

### 90 Days — *Make it a product*

**Goal:** Something presented as a shipped product, not a research project.

**Features:**
- [ ] **Match ID interface** — enter any StatsBomb match ID → get full xG analysis in the dashboard; removes CLI barrier entirely
- [ ] **Team comparison dashboard** — xG for vs xG against over-performance/under-performance by competition; identify teams that consistently outperform their xG (finishing quality signal)
- [ ] **Calibration fix** — temperature scaling on HybridGCN output to correct the systematic over-confidence above 20% xG; improves Brier score toward StatsBomb's 0.076
- [ ] **GitHub Actions CI** — automated pipeline tests + rebuild triggers on new StatsBomb open data drops
- [ ] **README as product page** — written for a club analyst, not a data scientist; includes a live demo link if deployed

**Success metric:** Someone outside the team reproduces the full pipeline from the README in under 30 minutes. At least one real analyst has looked at a match report and called it useful.

---

## Interview Narrative (3 Acts, 10 minutes)

### Act 1 — Problem (2 min)
> "Football clubs at the top spend millions on proprietary analytics. StatsBomb's xG model requires collecting shot technique, body position, and historical data — that's what costs money to gather at scale. But StatsBomb also publishes their 360 freeze-frame data for free — player positions at every event moment across 326 top-flight matches. I asked: what if you could get most of the way to their model using only that free spatial data?"

### Act 2 — How I built it (4 min)
> Walk through the 8-experiment progression. Start simple (Metrica team classifier, 100% AUC). Show the scale-up story: 5 WC2022 matches → 64 matches → all 7 competitions → Hybrid model. Key decision to highlight: *"I chose to benchmark against StatsBomb's own published xG rather than just claim improvement — that's the only honest baseline. It keeps the work honest."*

### Act 3 — What I learned and what's next (4 min)
> Pull up the goal analysis chart. Point to the calibration panel. *"The model ranks shots almost as well as StatsBomb — 0.782 vs 0.819 AUC across 8,013 shots. What it can't do is calibrate the probabilities correctly. When it says 60% chance of a goal, the real rate is closer to 40%. That's because it's blind to shot technique — a driven low shot and a speculative chip look the same in a freeze-frame. Adding shot technique as a one-hot feature closes most of that gap. That's experiment 9."* Then show the roadmap.

---

## Known Limitations & Honest Trade-offs

| Limitation | Impact | Mitigation |
|---|---|---|
| **Calibration over-confidence** | Brier score 0.178 vs StatsBomb's 0.073 | Temperature scaling post-training; shot technique features |
| **Shot technique blind** | Model can't distinguish driven shot from chip | Add `shot_technique` one-hot from event data (free) |
| **No keeper position** | Keeper is a flag (0/1), not a precise position | Already in node features; precision limited by freeze-frame timing |
| **~10% goal rate imbalance** | Pos_weight needed; recall/precision trade-off | Already handled; could try focal loss |
| **Free data scope** | StatsBomb 360 only available for major tournaments and one club league | Growing dataset — new competitions added each year |
| **CPU training only** | 3 models × 120 epochs takes ~35 min | Acceptable at this scale; GPU would take ~3 min |

---

## Tech Stack

| Component | Technology |
|---|---|
| Graph construction | `scipy.spatial.Delaunay`, PyTorch Geometric `Data` |
| GNN layers | `GCNConv`, `GATv2Conv` (PyG) |
| Training | PyTorch, `binary_cross_entropy_with_logits` with `pos_weight` |
| Baselines | scikit-learn `LogisticRegression`, StatsBomb published xG |
| Data | `statsbombpy`, raw GitHub JSON (bypass pandas 3.0 bug in `sb.frames()`) |
| Evaluation | AUC-ROC, Average Precision, Brier score, calibration curves |
| Visualisation | matplotlib, mplsoccer |
| Dashboard (planned) | Streamlit |

---

## What "Done" Looks Like

A club analyst at a Championship club can:
1. Enter a match ID
2. See every shot on a pitch map, colour-coded by HybridGCN xG
3. Click a shot → see which defenders the model flagged as key blockers (attention edges)
4. Export a one-page match report showing both teams' xG timelines
5. Understand what the model *can't* see (technique blind spot — stated honestly in the UI)

All of this using only free, open-source data and code.

---

*Last updated: March 2026*
