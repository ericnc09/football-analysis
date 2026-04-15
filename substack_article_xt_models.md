# Beyond xG: Building the Full Mathematical Toolkit for Football Analytics

*This is Part 4 of my ongoing series on building AI products from scratch.*

---

An xG model tells you how good a shot was. It does not tell you how the team got there.

In Parts 1–3, I built a graph neural network that predicts shot quality from freeze-frame data — player positions at the exact moment a shot is taken. The model achieves AUC 0.760 on 8,013 shots across 7 competitions, reaching 95.7% of StatsBomb's proprietary benchmark using only free, open data.

That is a useful number. But if you are a club analyst trying to evaluate a player, a coach trying to change a tactical approach, or a recruiter trying to decide whether a midfielder creates danger — xG is not enough.

xG answers: *how good was the shot?*

It does not answer:
- *How did the ball get to that position?* (action valuation)
- *Who controls the space on the pitch?* (spatial dominance)
- *What is the most likely final score?* (match prediction)
- *How strong is each team coming into this match?* (team strength)
- *What types of shots does this player take?* (player profiling)

These are different questions, and they require different mathematical models. Over the past sprint, I built all of them — six classical models that complete the analytics toolkit alongside the GNN core. This article is about the two I found most interesting: Expected Threat (xT) and the Poisson match model — and what the process taught me about building a complete analytical product.

---

## Expected Threat: Valuing Every Action, Not Just Shots

xG has a fundamental blind spot: it only activates when someone shoots. A midfielder who plays 70 progressive passes per match, consistently moving the ball from low-danger zones into high-danger zones, generates zero xG. A striker who takes a speculative 30-yard shot generates more xG than the midfielder, even though the midfielder's contribution was more valuable.

Expected Threat (xT) fixes this. It assigns a value to every location on the pitch based on how likely a possession that reaches that location is to eventually produce a goal. A pass from your own half into the opposition penalty area is worth something — and xT quantifies exactly how much.

### How It Works

The model is a Markov chain. I divided the pitch into a 12×8 grid (96 zones) and processed every pass, carry, and dribble from all 326 matches in my dataset — **619,407 actions** in total.

For each zone, I computed three quantities:
- **s(z)**: the probability that an action from zone z is a shot
- **g(z)**: the probability that a shot from zone z results in a goal
- **T(z→z')**: the probability that a non-shot action from zone z moves the ball to zone z'

Then I solved for xT iteratively:

```
xT(z) = s(z) · g(z) + (1 - s(z)) · Σ T(z→z') · xT(z')
```

In plain language: the threat from any zone equals the chance of scoring from there directly, plus the weighted average threat of everywhere the ball might go next. The equation converges in about 20 iterations.

### What the Numbers Say

The xT grid tells you what every football fan intuitively knows — but now with a number attached.

| Zone | xT Value | Interpretation |
|---|---|---|
| Central penalty area | **0.199** | Highest threat — one in five possessions reaching here produce a goal |
| Wide penalty area (flanks) | 0.134 | Crosses and cutbacks — valuable but lower conversion than central |
| Edge of the box | 0.095–0.110 | Transition zone — shots are attempted but conversion drops fast |
| Own half (any zone) | 0.085–0.093 | Baseline threat — nearly uniform, reflecting the long journey to goal |

The gradient from 0.085 to 0.199 is the entire story of attacking football. Every pass, every carry, every dribble either moves you up that gradient (positive ΔxT) or down it (negative ΔxT).

### The Leaderboard Tells You Something xG Cannot

Here is where it gets interesting. When you sum ΔxT across all of a player's actions, you get a measure of *how much threat they generated* — independent of whether they ever took a shot.

**Top 10 players by total xT generated (ΣΔxT):**

| Rank | Player | ΣΔxT | Actions | Role |
|---|---|---|---|---|
| 1 | Alejandro Grimaldo | **+12.19** | 4,008 | Left-back, Bayer Leverkusen |
| 2 | Granit Xhaka | +9.39 | 7,778 | Midfielder, Bayer Leverkusen |
| 3 | Jeremie Frimpong | +7.09 | 2,117 | Right-back, Bayer Leverkusen |
| 4 | Florian Wirtz | +7.05 | 3,980 | Attacking mid, Bayer Leverkusen |
| 5 | Alex Greenwood | +7.01 | 2,252 | Centre-back, England Women |
| 6 | Lucy Bronze | +6.98 | 2,141 | Right-back, England Women |
| 7 | Kosovare Asllani | +6.16 | 986 | Forward, Sweden Women |
| 8 | Lauren Hemp | +6.12 | 1,119 | Winger, England Women |
| 9 | Jonas Hofmann | +5.93 | 2,767 | Wing-back, Bayer Leverkusen |
| 10 | Antoine Griezmann | +5.71 | 1,240 | Forward, France |

Three things jump out.

**Grimaldo, a left-back, generates more threat than any forward in the dataset.** This makes perfect sense if you watched Bayer Leverkusen's unbeaten Bundesliga season. Grimaldo's overlapping runs and crosses from the left were the primary mechanism for penetrating defences. xG would rank him as a peripheral figure. xT correctly identifies him as the engine of Leverkusen's attack.

**Xhaka appears second — through volume.** His 7,778 actions are nearly double Grimaldo's. Each individual action generates modest ΔxT, but the accumulated effect of consistently progressing the ball from zone 4 to zone 7, possession after possession, is enormous. This is the Xhaka that Leverkusen paid for — not the xG Xhaka, who takes one speculative shot per match.

**Women's players dominate the 5–8 slots.** Greenwood and Bronze (England), Asllani (Sweden), Hemp (England) — these are the players who control the tempo and progression of their teams' attacks. xT surfaces this in a way that xG and traditional stats do not.

### What a PM Takes Away

xT answered a product question I had been circling for months: **how do you build a player evaluation tool for positions that do not score goals?**

xG is a striker's metric. It rewards the person who pulls the trigger. But recruitment decisions are made across every position — and a club evaluating a left-back needs a signal for creative contribution that is not "assists per 90," which is sparse, noisy, and context-dependent.

xT provides that signal. It is model-derived, operates on the full event stream (not just goals and assists), and naturally adjusts for the quality of opposition because the transition matrix is learned from the data. A progressive pass against a compact defensive block is harder to make and less likely to succeed — the model captures that through the transition probabilities.

The xT grid is also interpretable in a way that GNN embeddings are not. I can show a coach a heatmap and say: "this player consistently moves the ball from 0.09 zones to 0.13 zones — that is a +0.04 ΔxT per action, putting them in the top 5% of ball progressors." That is a conversation a non-technical stakeholder can participate in.

---

## Poisson: The Right Way to Predict Scorelines

In Part 3, I validated the xG model by running Monte Carlo simulations — 10,000 Bernoulli draws per match. Each shot is an independent coin flip weighted by its xG. Sum the goals, compare to reality.

That simulation achieved 51.9% accuracy on 3-way match outcome prediction (home win / draw / away win). Not bad. But there is a classical approach that does better.

The **Poisson distribution** models team goals as a Poisson process with rate parameter λ = sum of per-shot xG. It is the industry-standard method for match prediction — used by bookmakers, broadcasters, and betting models worldwide.

The appeal of the Poisson model is that it gives you the full scoreline probability matrix analytically. No simulation needed. For a match where the home team has λ=1.5 xG and the away team has λ=0.8 xG, you get:

```
P(1-0) = 13.4%    P(0-0) = 12.2%    P(2-1) = 8.1%
P(1-1) = 10.7%    P(0-1) = 9.8%     P(2-0) = 10.1%
```

Sum the appropriate cells and you get P(home win), P(draw), P(away win).

### Poisson vs Monte Carlo

| Model | 3-way Accuracy | Home-win Brier | RPS |
|---|---|---|---|
| Poisson (HybridGAT xG) | **56.1%** | 0.143 | 0.093 |
| MC Bernoulli sim (HybridGAT xG) | 51.9% | 0.164 | — |
| Poisson (StatsBomb xG) | 71.0% | 0.111 | 0.068 |

**Poisson outperforms the Monte Carlo simulation by 4.2 percentage points** — using exactly the same per-shot xG inputs.

Why? The Poisson model handles draw probability more gracefully. In a Monte Carlo simulation, draws only occur when simulated home and away goals happen to be exactly equal — a relatively rare event. The Poisson model computes the exact probability of every possible equal-scoreline outcome and sums them. This matters because draws occur in roughly 25% of real football matches, and underestimating them systematically hurts accuracy.

The gap to StatsBomb's Poisson model (~15pp) traces back to the same root cause identified in Part 3: shot placement. StatsBomb's xG incorporates where the ball was aimed on goal — a post-shot feature that makes individual λ estimates more precise. Our model uses only pre-shot spatial context.

### The Per-Competition Story

| Competition | Poisson Accuracy | StatsBomb Accuracy |
|---|---|---|
| Euro 2024 | **60.4%** | 68.8% |
| WC 2022 | 57.4% | 75.4% |
| WWC 2023 | 57.4% | 68.9% |
| Euro 2020 | 56.2% | 64.6% |
| Women's Euro 2025 | 54.8% | 74.2% |
| Bundesliga | 51.5% | 69.7% |
| Women's Euro 2022 | 50.0% | 78.6% |

The gap is narrowest in men's international tournaments (Euro 2024: 8.4pp) and widest in Women's Euro 2022 (28.6pp). This is consistent with the xG model's established strength: freeze-frame spatial geometry is most predictive in men's tournaments with diverse tactical styles. Competitions where defensive compactness varies less (women's early-round matches, league play) leave less room for spatial models to differentiate.

---

## The Full Toolkit

xT and Poisson were the two models that changed how I think about the product. But I built four more to complete the scope. Here is the full set:

| Model | What It Answers | Key Finding |
|---|---|---|
| **Random Forest** | Is the GNN actually better than a well-tuned tree? | AUC 0.759 — nearly matches HybridGAT+T (0.760). The graph adds calibration value, not ranking value. |
| **XGBoost** | Same question, different algorithm | AUC 0.728 but Brier 0.131 — better calibrated than RF despite lower AUC |
| **Elo Ratings** | How strong is each team? | Leverkusen top-rated (1,746). 73.5% accuracy in Bundesliga; 33.3% in Euro 2020 |
| **Voronoi Pitch Control** | Who controls the space when the shot is taken? | Goals have +6.9pp more shooting team control than misses |
| **K-Means Clustering** | What types of shots exist? What types of finishers? | 5 shot archetypes; headers convert at 16.1%, set pieces at 25.2% |

### The Random Forest Result Surprised Me

Random Forest on the same 27 features achieved AUC 0.759. The HybridGAT+T — a graph neural network with temperature scaling and per-competition calibration — achieved 0.760.

One point of AUC.

Does that mean the graph is useless? No. The GNN's advantage is in *calibration*, not ranking. HybridGAT+T achieves Brier 0.148 vs RF's 0.169 — a meaningful gap when your product outputs probability estimates that clubs use for decision-making. The graph component does not find more goals; it produces better-calibrated probabilities about the ones it finds.

This is a product insight. If your use case is "rank shots from best to worst," Random Forest is 95% of the way there with 5% of the complexity. If your use case is "tell a coach that this shot had a 23% chance and mean it," you need the GNN.

### The Elo Result Was Predictable in Retrospect

Elo ratings predicted match outcomes at 50.3% accuracy across all 326 matches. In the Bundesliga (34 matches, same teams playing repeatedly), accuracy was 73.5%. In Euro 2020 (51 matches, most teams play 3 games then go home), accuracy was 33.3%.

This is exactly what you would expect. Elo needs repeated observations to converge. A league season gives you that. A tournament does not.

The PM lesson: **Elo is a season-scale metric, not a tournament-scale metric.** If you are building a product for league analytics, Elo is a strong prior. If you are building for tournament prediction, you need match-level models (Poisson, MC simulation) that use in-game data rather than historical ratings.

---

## What Building Six Models Taught Me About Product Scope

In my first sprint, I built a single model and optimised it relentlessly — pushing AUC from 0.593 to 0.760 across 10 experiments. That was the right decision then: establish the core technical contribution and validate it rigorously.

This sprint was the opposite. I built six models in quick succession, none of them optimised beyond reasonable defaults. Random Forest with 500 trees and default hyperparameters. Elo with K=32 and a standard home advantage. K-Means with silhouette-score-selected k.

Both sprints were correct for their moment.

### The research sprint establishes credibility

You need one model that is defensible under scrutiny. Ablation tables, bootstrap confidence intervals, per-competition breakdowns, cross-gender evaluation. That is the core of the MLSA paper submission.

### The product sprint establishes coverage

A club analyst does not care that your xG model has AUC 0.760. They care that they can open a dashboard and see:
- Which players generate the most threat (xT)
- What the likely scoreline is (Poisson)
- How strong the opposition is (Elo)
- Who controls the pitch at the moment of a shot (Voronoi)
- What type of finisher a transfer target is (clustering)

Each of these is a **10-minute implementation** compared to the months of work on the GNN. But without them, the product has a single-metric problem: it can only answer one question.

### Diminishing returns on a single metric vs increasing returns on coverage

The last 0.01 AUC on the GNN cost me: shot placement features, edge feature debugging, per-competition temperature scaling, three rounds of ablation. Meaningful research contributions, but marginal product value.

The six new models each took a few hours and filled a genuine gap in the analytical surface area. A coach who asks "who should we sign as a progressive midfielder?" needs xT, not another decimal place on xG.

**The product management lesson: once your core metric is defensible, breadth beats depth.** A product that answers six questions competently is more useful than a product that answers one question to four decimal places.

---

## What Is Next

The paper is being formatted for MIT Sloan Sports Analytics Conference. The six new models are not paper contributions — they are product contributions that make the demo more compelling and the tool more useful.

The next article will cover the Streamlit dashboard itself — the eight interactive tabs, the design decisions behind each one, and what I learned about building a data product that a non-technical football analyst can pick up in under five minutes.

---

*All six models, figures, and results are available in the GitHub repository. Every result is reproducible from free StatsBomb open data. The codebase now covers: GNN xG, logistic regression, Random Forest, XGBoost, Poisson match prediction, Elo ratings, Expected Threat (Markov chains), Voronoi pitch control, K-Means clustering, Monte Carlo simulation, and MC Dropout uncertainty quantification — eleven mathematical models applied to football analytics.*
