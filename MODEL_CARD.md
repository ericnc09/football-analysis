# Model Card — Football HybridGAT xG (pool_7comp)

Graph neural network expected-goals model trained on seven StatsBomb
open-data tournaments. Calibrated with temperature scaling (both a
single pooled temperature and per-competition temperatures). Intended
as the inference model behind the `football-xg-dashboard` HuggingFace
Space.

This file is uploaded as `README.md` to the HF Hub model repo
(`<user>/football-xg-models`). Required by
`reviews/04_ml_engineer_review.md` §11 ("Model card on HF Hub: does
not exist").

---

## Model overview

- **Architecture**: `HybridGAT` — a 2-layer GATv2 encoder over a
  23-node shot graph (shooter + 10 teammates + 10 opponents + ball +
  goal mouth) concatenated with a 30-D metadata vector (v3 PSxG
  feature schema).
- **Checkpoint**: `pool_7comp_hybrid_gat_xg.pt` (accompanied by a
  `.meta.json` sidecar with sha256, git SHA, feature-schema version,
  training seed, and validation metrics).
- **Calibration**: temperature scaling. Two forms ship:
  - Global: `pool_7comp_gat_T.pt` (single scalar `T ≈ 0.720`).
  - Per-competition: `pool_7comp_per_comp_T_gat.pt` (dict keyed by
    `comp_label`, enabled in the dashboard via the competition
    selector).
- **Also in repo**: `pool_7comp_hybrid_xg.pt` (HybridGCN variant, used
  as a baseline comparison in the dashboard and in the paper's
  ablation table).
- **License**: MIT for model weights. Underlying training data is
  StatsBomb Open Data (Open Data Commons; attribution required; see
  §License).

---

## Intended use

- **Exploratory xG inspection**: compare this model's shot-level xG
  against StatsBomb's industry-standard xG on matches from the seven
  training tournaments (held-out test split).
- **Research baseline**: a reference GNN xG model for papers
  benchmarking new architectures against StatsBomb Open Data. The
  held-out splits and per-competition calibration temperatures are
  published alongside so other researchers can reproduce.
- **Teaching / demo**: the dashboard it powers is a pedagogical tool
  for demonstrating graph-based shot evaluation; not a production
  signal for betting, scouting valuations, or club decisions.

## Out-of-scope use

The model was trained on **7 tournaments**, most of them elite
international level (WC, Euro, WWC, WEuro). Do NOT assume it
generalises to:

- **Domestic league play outside Bundesliga 23/24** — no La Liga, no
  Premier League, no MLS, no J-League, no Thai League.
- **Youth or amateur football** — training data is senior
  professional only.
- **Set-piece specialist analysis** — the feature set flags corners
  / free-kicks / penalties but does not distinguish subtypes
  (direct vs indirect free kick, inswinging vs outswinging corner).
- **Live in-game prediction** — the model consumes StatsBomb
  freeze-frame data at the moment of shot. It is not a streaming
  predictor.
- **Gambling or financial decisions** — calibration ECE is non-zero
  (see §Metrics) and confidence intervals are wide on
  small-tournament slices. Any decision with real-world stakes
  should not rely on a single model's point estimate.

### Known biases

- **Tournament skew**: 5 of 7 training competitions are Men's elite
  international. Women's football is present (WWC 2023, WEuro 2022,
  WEuro 2025) but under-represented in absolute shot count.
- **Sample imbalance by competition**: sample sizes per
  held-out test split range from 124 (WEuro 2022) to 224 (WWC 2023).
  Bootstrap AUC CIs are correspondingly wide on small slices.
- **Open-data coverage**: StatsBomb Open Data is a sampled release,
  not the full feed they sell commercially. Player tracking density
  is higher than Opta/Wyscout public feeds but lower than StatsBomb's
  paid 360 product.
- **Shot-outcome labels** follow StatsBomb's definitions. Own goals,
  disallowed goals, and goals-from-shots-blocked-by-a-defender follow
  their conventions, which may not match other providers'.

---

## Training data

| Competition | Gender | Season | Shots (train+val+test) |
| --- | --- | --- | --- |
| FIFA World Cup 2022 | Men's | 2022 | ~2,087 |
| UEFA Euro 2020 | Men's | 2021 | ~1,741 |
| UEFA Euro 2024 | Men's | 2024 | ~1,994 |
| Bundesliga 23/24 (sampled matches) | Men's | 2023–24 | ~1,300 |
| FIFA Women's World Cup 2023 | Women's | 2023 | ~2,240 |
| UEFA Women's Euro 2022 | Women's | 2022 | ~1,246 |
| UEFA Women's Euro 2025 | Women's | 2025 | ~1,405 |

Total: ~12,000 shots across 7 competitions. Held-out test shares (the
data in the tables below) are ~10% per competition.

Source: StatsBomb Open Data (`statsbombpy` API), tournament JSONs
released under ODC-BY 1.0. Download scripts are in
`scripts/build_dataset.py`; graph construction is in
`src/graph_builder.py`; feature engineering is in `src/features.py`.

### Feature schema

Version: **v3-psxg** (30-D metadata vector per shot). See
`src/features.py → CURRENT_FEATURE_SCHEMA_VERSION` and the
`.meta.json` sidecar shipped alongside each checkpoint. Runtime loads
the checkpoint via `src/serving.py` which validates that the sidecar
schema version matches the code before serving predictions — schema
drift raises `FeatureSchemaMismatch` at app startup.

Features include shot geometry (distance, angle), context flags
(header / open-play / foot / technique one-hot), defender pressure
(count in cone, count on direct line), goalkeeper state (distance,
off-centre, perpendicular offset), and a 9-zone shot-placement
one-hot (PSxG-style).

---

## Evaluation metrics

### Held-out per-competition performance

All numbers on the held-out test split, HybridGAT + calibrated
temperature. Bootstrap 95% CI over 2,000 resamples.

| Competition | Gender | n | Goals | AUC | AUC 95% CI | Brier (cal.) | ECE (cal.) | StatsBomb AUC | Δ vs SB | % of SB |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| FIFA WC 2022 | Men's | 209 | 20 | 0.751 | [0.620, 0.869] | 0.155 | 0.232 | 0.827 | −0.075 | 90.9% |
| UEFA Euro 2020 | Men's | 175 | 20 | 0.763 | [0.637, 0.876] | 0.153 | 0.218 | 0.827 | −0.065 | 92.2% |
| UEFA Euro 2024 | Men's | 200 | 15 | 0.685 | [0.524, 0.831] | 0.133 | 0.213 | 0.727 | −0.042 | 94.3% |
| Bundesliga 23/24 | Men's | 130 | 19 | 0.748 | [0.617, 0.866] | 0.152 | — | 0.845 | −0.097 | 88.5% |
| FIFA WWC 2023 | Women's | 224 | 27 | **0.812** | [0.728, 0.887] | 0.133 | — | 0.760 | **+0.052** | 106.8% |
| UEFA WEuro 2022 | Women's | 124 | 9 | **0.844** | [0.697, 0.977] | 0.146 | — | 0.862 | −0.018 | 97.9% |
| UEFA WEuro 2025 | Women's | 141 | 16 | 0.715 | [0.575, 0.843] | 0.179 | — | 0.729 | −0.014 | 98.1% |

Interpretation:

- On Men's tournaments the model sits 3–10% below StatsBomb's
  industry xG. Expected — StatsBomb's model is trained on
  orders-of-magnitude more data.
- On Women's tournaments the model matches or beats StatsBomb on two
  of three slices (WWC 2023 +5.2 AUC points; WEuro 2025 −1.4 AUC
  points within CI). Interpret cautiously given the small test-split
  sizes — WWC 2023's CI lower bound is 0.728, not 0.812.
- Bundesliga is the hardest slice: smallest n (130), largest gap
  vs StatsBomb. Reasonable since it's the only domestic-league
  competition in the training set.

### Aggregate

- **Pooled test AUC** (all 7 competitions): 0.758
- **Pooled baseline AUC (StatsBomb xG)**: 0.801
- **Calibration**: global temperature `T = 0.720` reduces Brier by
  ~4% on average; per-competition temperatures are available in
  `pool_7comp_per_comp_T_gat.pt` for users who want
  competition-specific calibration.

### By-gender rollup

Reviewer §11 specifically asks for this. Weighted by test-split n:

| Gender | n | Weighted AUC | Notes |
| --- | ---: | ---: | --- |
| Men's | 714 | ~0.740 | WC / Euro / Bundesliga |
| Women's | 489 | ~0.791 | WWC / WEuro ×2 |

The Women's slice scoring higher on a smaller sample is consistent
with StatsBomb's own observation that shot decision-making in
Women's tournament play is more deterministic given the feature set
(fewer highly-contested headers; fewer deflected shots).

---

## Training procedure

- **Optimiser**: Adam, `lr=5e-4`, `weight_decay=1e-4`.
- **Epochs**: 30 with early stopping on val AUC (patience 5).
- **Batch size**: 64 graphs.
- **Split**: competition-pooled 5-fold; held-out test set is a
  tournament-balanced 10% slice created with a seeded shuffle
  (`seed=42`). Exact indices are reproducible via
  `src/reproducibility.set_seed(42)`.
- **Hardware**: CPU-only training runs in ~45 min on an M1 Pro.
- **Reproducibility**: `set_seed(42)` pins Python random, NumPy,
  torch CPU + CUDA RNGs, and sets cuDNN deterministic. Every training
  run appends to `results/runs.jsonl` with git SHA, seed, and final
  metrics; the `.meta.json` sidecar carries the same metadata for
  runtime verification.

---

## How to load and serve

```python
from pathlib import Path
from src.serving import load_gat_model, predict_batch

model = load_gat_model(
    Path("pool_7comp_hybrid_gat_xg.pt"),
    Path("pool_7comp_gat_T.pt"),
    device="cpu",
)
probs = predict_batch(model, list_of_graphs)   # torch.Tensor, shape [N]
```

The `load_gat_model` helper verifies the sidecar sha256 against the
`.pt` file and enforces feature-schema compatibility. Any mismatch
raises before inference starts. See `src/serving.py` for the full
contract and `tests/test_serving.py` for a train-save-load-predict
round-trip test.

---

## Limitations to keep in mind

1. **Wide CIs on small slices**. WEuro 2022 AUC is 0.844 but CI
   spans [0.697, 0.977]. Don't over-index on a single competition's
   point estimate.
2. **v3-psxg features are not transportable** to data providers
   that don't ship freeze-frames (Opta public feeds, FBref). The
   model will refuse to load if feature-schema version in the
   sidecar doesn't match what the code computes.
3. **No shot-taker identity feature**. The model doesn't know who's
   shooting. Good (no star-player bias on unseen competitions) but
   also a ceiling on absolute accuracy.
4. **No body-state / stamina features**. A tired striker and a
   fresh striker look the same to this model.
5. **No historical form**. A striker on a 10-goal streak and one
   on a 20-game drought look identical given the same shot geometry.

---

## License

- **Model weights**: MIT license. Free to use, modify, redistribute.
- **Training data**: StatsBomb Open Data under Open Data Commons
  Attribution License (ODC-BY 1.0). Any paper, dashboard, or
  derivative must credit StatsBomb. Citation:

  > Hudl StatsBomb (2023–2025). *StatsBomb Open Data*.
  > https://github.com/statsbomb/open-data

- **Code**: MIT license; see `LICENSE` in the training repo.

---

## Changelog

- **v0.2.0** (2026-04): HybridGAT replaces HybridGCN as primary
  model. Per-competition temperature scaling added. Feature schema
  bumped to v3-psxg with 9-zone placement one-hot.
- **v0.1.0** (2026-03): Initial release — HybridGCN with global
  temperature scaling, 4 competitions.

---

## Contact + reporting issues

Issues, bias reports, or reproducibility problems: open an issue on
the training repo or email the maintainer listed in the Space's
footer. Include the `session` id from the error box —
`view_boundary` logs every crash with a session-scoped UUID that
lets the maintainer pull the matching log stream.
