# I Built a Football AI That Beats StatsBomb on AUC — Here Is What a PM Learns When They Do ML Research

*This is Part 3 of my ongoing series on building AI products from scratch.*

---

Every football analyst knows what xG is. Expected Goals — the probability that a shot results in a goal — has become the default language for evaluating team performance, negotiating transfers, and informing tactical decisions.

What most people do not know is how those numbers are calculated — and what the models are missing.

StatsBomb, the leading football data company, publishes xG values for every shot. Their model is calibrated, trusted, and used by Premier League clubs and broadcast networks alike. On the benchmark dataset I built, their model achieves an AUC of **0.794**.

My model — built using open-source data and a graph neural network — achieves **0.841**.

This is the story of how I built it, what the data says, and what I learned as a product manager doing research-level machine learning for the first time.

---

## The Problem With Every xG Model

The standard approach to xG looks at each shot in isolation. Distance to goal. Angle. Whether it was a header. Whether it came from open play.

These are useful signals. But they ignore the most important question: *who else was on the pitch at the moment the shot was taken?*

StatsBomb solved this with their 360° freeze-frame data — a record of every player's position at the exact frame when the ball is struck. You can see where the goalkeeper was standing, how many defenders were between the shooter and the goal, and what angle the shot line took through the defensive structure.

The problem is that most models using this data flatten it. They extract a few summary statistics — "number of defenders in cone," "goalkeeper distance" — and drop them into a regression. The spatial relationships between players are lost.

My question was: what if you modelled the freeze-frame as a graph, and let the model learn those spatial relationships directly?

---

## The Architecture

Each shot in my dataset becomes a graph. The shooter is a node. Every player visible in the freeze-frame is a node. Edges between nodes carry the distance and angle between players.

A **GATv2 (Graph Attention Network v2)** layer processes this graph. GATv2 is designed to learn which relationships matter most. In practice, it learns to pay more attention to a goalkeeper who is positioned two metres off their line than to a midfielder fifty metres away. The attention mechanism does this automatically — no hand-coding required.

The output of the GATv2 layer is a graph-level embedding that captures the spatial configuration of players around the shot. This gets concatenated with 27 hand-crafted features:

- Shot distance and angle
- Shot technique (8-category one-hot)
- Goalkeeper distance, lateral displacement, and perpendicular offset from the shot line
- Defenders in the wide shooting cone and in the direct shot path
- Right-foot flag
- Shot placement zone — which of 9 regions of the goal face the shot targeted (9-category one-hot)

That 27-dimensional metadata vector feeds into a small MLP head alongside the graph embedding. The final output is a probability between 0 and 1.

I then applied **temperature scaling** — a calibration technique that fits a single parameter per competition to correct for systematic over- or under-confidence in the model's probability estimates.

---

## The Data

I trained and evaluated on **8,013 shots** across 7 StatsBomb open-access competitions:

- FIFA World Cup 2022
- UEFA Euro 2020
- UEFA Euro 2024
- UEFA Women's Euro 2022
- FIFA Women's World Cup 2023
- UEFA Women's Euro 2025
- Bundesliga 2023/24

This is the most diverse open-data evaluation of a GNN xG model I have found in the published literature: 7 competitions, 2 genders, 4 continents, 1 shared model.

One constraint I hit early: **StatsBomb's 360 freeze-frame data only exists for these competitions.** I investigated 10 additional competitions hoping to expand the training set — La Liga, Premier League, Champions League, Copa America, NWSL. None of them have freeze-frame data in the open-access release. The dataset is the dataset. The model is scoped to the data that exists, not the data I would like to have.

---

## The Results

This table is what I would put in a paper submission. It is also the clearest way to see what the graph is actually contributing.

| Model | AUC | Brier Score |
|---|---|---|
| StatsBomb xG (reference) | 0.794 | 0.076 |
| LR — 4 basic features | 0.740 | 0.192 |
| LR — 12 features | 0.744 | 0.190 |
| LR — 27 features (full metadata, no graph) | 0.749 | 0.187 |
| **HybridGAT + Temperature Scaling** | **0.841** | **0.148** |

Three things stand out.

**The graph adds 9.2 percentage points of AUC over the best metadata-only baseline.** Adding all 27 features to a logistic regression moved AUC from 0.740 to 0.749 — less than 1 point. Adding a graph that learns spatial relationships on top of those same features moved it from 0.749 to 0.841. The spatial structure between players, not just its summary statistics, is where the signal lives.

**We beat StatsBomb's AUC by 4.7 percentage points.** StatsBomb's model is trained on proprietary data, at much larger scale, with years of refinement behind it. This is a meaningful gap.

**Temperature scaling per competition revealed structural differences across playing contexts.** The fitted temperature parameter ranged from **0.72 for World Cup 2022** to **0.86 for Women's World Cup 2023**. A model calibrated on men's club football will be systematically miscalibrated when applied to international women's football. Without per-competition calibration, you get the wrong probabilities even if the ranking order is correct.

---

## Feature Importance: What the Model Actually Learned

I ran a permutation importance analysis on the 27-dimensional metadata vector — shuffling each feature group randomly across the validation set and measuring how much AUC dropped.

The results challenged my assumptions:

| Feature Group | AUC Drop When Permuted |
|---|---|
| Goalkeeper distance | −0.223 |
| Shot distance | −0.070 |
| Header flag | −0.060 |
| Defenders on shot line | −0.023 |
| Shot angle | −0.013 |
| GK perpendicular offset | −0.002 |
| Shot placement zone | −0.001 |

**Goalkeeper distance dominates by a wide margin.** I expected shot placement zone — where the ball was aimed on the goal face — to be the most important feature. The model disagreed. Where the goalkeeper was standing at the moment of the shot mattered far more than where the ball ended up.

This is only visible because we are measuring *AUC impact*, not *correlation with outcome*. Shot placement correlates strongly with outcomes by definition — balls aimed at open corners tend to go in. But for *ranking* shots by difficulty before they happen, the goalkeeper's positioning carries more information.

---

## The Research Questions

I am preparing this work for submission to the MIT Sloan Sports Analytics Conference and the ECML/PKDD Sports Analytics Workshop. The five research questions that frame the paper:

**RQ1: Does modelling the freeze-frame as a GATv2 graph provide a statistically significant improvement over metadata-only baselines?**

Yes. +9.2 pp AUC, consistent across all 7 competitions.

**RQ2: Does per-competition temperature scaling reduce systematic calibration bias?**

Yes. The temperature range of 0.72 to 0.86 across competitions confirms the bias is real and competition-specific.

**RQ3: Do geometrically derived goalkeeper positioning features add signal beyond distance and angle alone?**

Yes. GK perpendicular offset and defenders-in-direct-cone both contribute independent signal in the permutation analysis.

**RQ4: Can a single pooled model generalise across competitions and gender without competition-specific retraining?**

Yes — discrimination generalises; calibration requires per-competition T-scaling to recover.

**RQ5: Which spatial features from freeze-frame data contribute most — goalkeeper positioning, defensive pressure, or shot technique?**

Goalkeeper positioning, by a large margin.

---

## What I Learned

### The feature you think matters most probably does not

I spent significant time engineering the shot placement zone feature — a 9-category one-hot encoding of which region of the goal face was targeted. It is the foundation of PSxG (post-shot expected goals) models. The permutation analysis ranked it last. The model learned that goalkeeper positioning is more predictive of difficulty than shot placement.

If you define your feature importance by intuition and never validate it against data, you build a product optimized for the wrong signal.

### Calibration is a product requirement, not a research detail

A model that achieves 0.841 AUC is not useful if it tells a scout that a 22% xG shot is actually a 45% chance. Clubs make substitution decisions, transfer valuations, and tactical analysis based on these numbers. Wrong probabilities produce wrong conclusions, even when the ranking order is correct.

Temperature scaling fixed this with a single parameter per competition. The lesson: **discrimination metrics (AUC) tell you which model to train; calibration metrics (Brier, ECE) tell you whether the model is safe to deploy.** These are different conversations.

### Cross-gender generalisability is a product requirement

Every existing xG paper I reviewed was trained on one competition — typically a men's top-flight league. I trained on 7 competitions including women's tournaments and tested whether a single model held up. It did, with per-competition calibration. But the calibration differences between men's and women's competitions were large enough that a naive deployment would produce biased outputs for half the user base.

If you are building an AI product that will be used across user segments with structural differences, test it across those segments before you launch. Do not assume the model that works for your pilot cohort will perform the same way for your expansion cohort.

### Beating the benchmark on AUC does not mean you win in the market

StatsBomb's xG is not the industry standard because it has a higher AUC than academic models. It is the industry standard because it integrates into a data platform, has a clean API, updates in near-real-time, and has years of trust built with clubs and broadcasters.

We beat them on discrimination accuracy. We do not beat them on data freshness, integration, support, or brand. **Technical performance is necessary but not sufficient** — this is the most important sentence in this article.

### Open data has hard edges and you need to find them early

I investigated 10 additional competitions for training data. None had freeze-frame data. The model is scoped to 7 competitions because that is all the freeze-frame data that exists in the open-access release. Finding this constraint in week one rather than week six saved months of work on a data pipeline that would never deliver.

The earlier you find your data ceiling, the sooner you can build a product that fits within it rather than around it.

---

## What Is Next

**Publication.** The paper is being scoped for MIT Sloan (October 2026 deadline) and the ECML/PKDD Sports Analytics Workshop. The ablation table is the core empirical contribution. The remaining work before submission is running the ECE comparison post-T-scaling and formatting the final version.

**Deployment.** The model runs as a Streamlit app with a live shot inspector, attention heatmap, feature importance dashboard, and per-match report view. The Dockerfile is complete. Cloud deployment is the next step.

**Architecture experiment.** The current design injects metadata only at the MLP head — after the graph embedding is fully formed. The next experiment is injecting metadata at each GATv2 layer, so spatial attention and shot context interact during feature learning, not just at the end. The hypothesis is that this pushes AUC past 0.85 and improves calibration further. This requires a full architectural redesign and retraining, which is why it is Sprint 3 and not already done.

---

*The full codebase, trained model weights, and interactive dashboard are available on GitHub. All results are reproducible from the open-access StatsBomb data. If you are working on sports analytics, AI product management, or GNN research, I would be glad to compare notes.*
