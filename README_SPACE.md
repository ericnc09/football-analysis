---
title: Football GNN xG Dashboard
emoji: ⚽
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.55.0
app_file: app.py
pinned: false
license: mit
---

# Football GNN · xG Dashboard

Interactive dashboard for exploring **HybridGATv2 expected goals (xG)** predictions versus StatsBomb's industry model across 8,013 shots from 7 competitions.

**AUC 0.760 · Brier 0.148 · 95.7% of StatsBomb AUC — using only free, open data.**

## What's inside

- **xG Overview** — goal/miss distributions, reliability diagram, per-competition breakdown
- **Match Report** — shot map, timeline, and xG comparison for individual matches
- **Shot Explorer** — freeze-frame graph viewer with GATv2 attention weights and node saliency
- **Calibration** — reliability diagram before/after temperature scaling vs StatsBomb
- **Feature Importance** — permutation importance across all 27 metadata dimensions

## Model

HybridGATv2 — freeze-frame spatial graph (GATv2Conv × 3, edge features) concatenated with 27-dim shot metadata MLP, per-competition temperature scaling.

Source: [github.com/ericnc09/football-analysis](https://github.com/ericnc09/football-analysis)
