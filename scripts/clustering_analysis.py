#!/usr/bin/env python3
"""
clustering_analysis.py
----------------------
K-Means clustering on shot features and player profiles.

Two analyses:
  1. Shot clustering: K-Means on 27-dim shot metadata → shot archetypes
     (close-range headers, long-range drives, cutback shots, etc.)
  2. Player clustering: Aggregate per-player stats → finishing profiles

Uses elbow method + silhouette score to determine optimal k.

Outputs
-------
  data/processed/clustering_results.json
  assets/fig_clustering.png

Usage
-----
  python scripts/clustering_analysis.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

REPO_ROOT = Path(__file__).parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
ASSETS    = REPO_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading — same as other scripts
# ---------------------------------------------------------------------------

def load_graphs() -> list:
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
        print(f"  {pt.name}: {len(gs)}")
    print(f"  Total: {len(graphs)} graphs")
    return graphs


def _safe(g, attr, default):
    if hasattr(g, attr):
        val = getattr(g, attr)
        return float(val.item() if hasattr(val, "item") else val)
    return default


FEAT_NAMES = (
    ["shot_dist", "shot_angle", "is_header", "is_open_play"]
    + [f"tech_{i}" for i in range(8)]
    + ["gk_dist", "n_def_in_cone", "gk_off_centre"]
    + ["gk_perp_offset", "n_def_direct_line", "is_right_foot"]
    + [f"placement_{i}" for i in range(9)]
)


def extract_features(graphs):
    """Extract 27-dim metadata and labels."""
    meta, labels, sb_xg, players, teams, comps = [], [], [], [], [], []
    for g in graphs:
        dist      = float(g.shot_dist.item())
        angle     = float(g.shot_angle.item())
        header    = float(g.is_header.item())
        open_play = float(g.is_open_play.item())
        tech      = g.technique.tolist()
        gk_dist   = float(g.gk_dist.item())
        n_def     = float(g.n_def_in_cone.item())
        gk_off    = float(g.gk_off_centre.item())
        gk_perp   = _safe(g, "gk_perp_offset",   3.0)
        n_direct  = _safe(g, "n_def_direct_line", 0.0)
        right_ft  = _safe(g, "is_right_foot",     0.5)
        plc       = (g.shot_placement.tolist()
                     if hasattr(g, "shot_placement")
                     else [0.0] * 9)

        meta.append([dist, angle, header, open_play] + tech +
                    [gk_dist, n_def, gk_off] + [gk_perp, n_direct, right_ft] + plc)
        labels.append(int(g.y.item()))
        sb_xg.append(float(g.sb_xg.item()))
        players.append(g.player_name)
        teams.append(g.team_name)
        comps.append(getattr(g, "comp_label", "unknown"))

    return {
        "X":       np.array(meta, dtype=np.float32),
        "y":       np.array(labels),
        "sb_xg":   np.array(sb_xg),
        "players": players,
        "teams":   teams,
        "comps":   comps,
    }


# ---------------------------------------------------------------------------
# Shot clustering
# ---------------------------------------------------------------------------

def cluster_shots(X, y, sb_xg, k_range=range(3, 10)):
    """Run K-Means on standardized shot features, find optimal k."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_,
                                             sample_size=min(5000, len(X))))

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"    Best k by silhouette: {best_k} "
          f"(score={max(silhouettes):.3f})")

    # Final clustering with best k
    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
    labels = km.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_scaled)
    print(f"    PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # Cluster profiles
    clusters = []
    for c in range(best_k):
        mask = labels == c
        n = mask.sum()
        goal_rate = y[mask].mean()
        mean_xg = sb_xg[mask].mean()
        mean_feat = X[mask].mean(axis=0)

        # Characterize by dominant features
        profile = {
            "cluster": c,
            "n_shots": int(n),
            "goal_rate": round(float(goal_rate), 4),
            "mean_xg": round(float(mean_xg), 4),
            "mean_shot_dist": round(float(mean_feat[0]), 2),
            "mean_shot_angle": round(float(mean_feat[1]), 4),
            "pct_header": round(float(mean_feat[2]), 3),
            "pct_open_play": round(float(mean_feat[3]), 3),
            "mean_gk_dist": round(float(mean_feat[12]), 2),
            "mean_n_def": round(float(mean_feat[13]), 2),
        }

        # Name the cluster
        name = _name_cluster(mean_feat, goal_rate)
        profile["name"] = name
        clusters.append(profile)
        print(f"    Cluster {c} ({name}): n={n}, goal%={goal_rate:.1%}, "
              f"dist={mean_feat[0]:.1f}m, header={mean_feat[2]:.0%}")

    return {
        "k": best_k,
        "labels": labels,
        "X_2d": X_2d,
        "clusters": clusters,
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def _name_cluster(mean_feat, goal_rate):
    """Heuristic naming based on feature means."""
    dist = mean_feat[0]
    header = mean_feat[2]
    open_play = mean_feat[3]

    if header > 0.5:
        return "Headers"
    if dist < 8:
        return "Close-range"
    if dist > 22:
        return "Long-range"
    if open_play < 0.3:
        return "Set pieces"
    if goal_rate > 0.2:
        return "High-quality chances"
    return "Mid-range open play"


# ---------------------------------------------------------------------------
# Player clustering
# ---------------------------------------------------------------------------

def cluster_players(X, y, sb_xg, players, teams, comps, min_shots=5):
    """Aggregate per-player stats and cluster."""
    player_stats = defaultdict(lambda: {
        "shots": 0, "goals": 0, "xg_sum": 0.0,
        "dists": [], "angles": [], "headers": 0,
        "team": "", "comp": "",
    })

    for i, p in enumerate(players):
        ps = player_stats[p]
        ps["shots"] += 1
        ps["goals"] += y[i]
        ps["xg_sum"] += sb_xg[i]
        ps["dists"].append(X[i, 0])
        ps["angles"].append(X[i, 1])
        ps["headers"] += int(X[i, 2] > 0.5)
        ps["team"] = teams[i]
        ps["comp"] = comps[i]

    # Filter to players with enough shots
    qualified = {p: s for p, s in player_stats.items() if s["shots"] >= min_shots}
    print(f"    Players with >= {min_shots} shots: {len(qualified)}")

    # Build feature matrix for clustering
    pnames, features = [], []
    for p, s in qualified.items():
        pnames.append(p)
        features.append([
            s["shots"],
            s["goals"],
            s["xg_sum"],
            s["goals"] - s["xg_sum"],  # overperformance
            s["goals"] / max(s["shots"], 1),  # conversion rate
            s["xg_sum"] / max(s["shots"], 1),  # mean xG per shot
            np.mean(s["dists"]),  # avg shot distance
            np.mean(s["angles"]),  # avg shot angle
            s["headers"] / max(s["shots"], 1),  # header rate
        ])

    X_players = np.array(features, dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_players)

    # Find optimal k
    k_range = range(3, 8)
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"    Best k: {best_k} (silhouette={max(silhouettes):.3f})")

    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
    labels = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_scaled)

    player_feat_names = ["shots", "goals", "xg_sum", "overperf",
                          "conv_rate", "mean_xg", "avg_dist", "avg_angle",
                          "header_rate"]

    clusters = []
    for c in range(best_k):
        mask = labels == c
        n = mask.sum()
        mean_f = X_players[mask].mean(axis=0)
        members = [pnames[i] for i in range(len(pnames)) if labels[i] == c]
        # Sort by goals descending
        member_goals = [(m, qualified[m]["goals"]) for m in members]
        member_goals.sort(key=lambda x: -x[1])

        profile = {
            "cluster": c,
            "n_players": int(n),
            "mean_shots": round(float(mean_f[0]), 1),
            "mean_goals": round(float(mean_f[1]), 1),
            "mean_xg": round(float(mean_f[2]), 2),
            "mean_overperf": round(float(mean_f[3]), 2),
            "mean_conv_rate": round(float(mean_f[4]), 3),
            "mean_avg_dist": round(float(mean_f[6]), 1),
            "mean_header_rate": round(float(mean_f[8]), 3),
            "top_players": [m[0] for m in member_goals[:5]],
        }
        clusters.append(profile)
        top3 = ", ".join(m[0] for m in member_goals[:3])
        print(f"    Cluster {c}: n={n}, shots={mean_f[0]:.0f}, "
              f"goals={mean_f[1]:.0f}, conv={mean_f[4]:.1%} | {top3}")

    return {
        "k": best_k,
        "labels": labels.tolist(),
        "player_names": pnames,
        "X_2d": X_2d,
        "clusters": clusters,
        "feature_names": player_feat_names,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(shot_result, player_result):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("K-Means Clustering — Shot Archetypes & Player Profiles",
                 fontsize=13, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, max(shot_result["k"],
                                                 player_result["k"])))

    # ── Panel 1: Shot clusters (PCA) ─────────────────────────────────────
    ax = axes[0, 0]
    X_2d = shot_result["X_2d"]
    labels = shot_result["labels"]
    for c in range(shot_result["k"]):
        mask = labels == c
        name = shot_result["clusters"][c]["name"]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, alpha=0.4,
                   color=colors[c], label=f"{name} (n={mask.sum()})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Shot Archetypes (PCA projection)")
    ax.legend(fontsize=7, markerscale=3)

    # ── Panel 2: Elbow + silhouette ───────────────────────────────────────
    ax = axes[0, 1]
    k_range = shot_result["k_range"]
    ax2 = ax.twinx()
    ax.plot(k_range, shot_result["inertias"], "b-o", ms=5, label="Inertia")
    ax2.plot(k_range, shot_result["silhouettes"], "r-s", ms=5, label="Silhouette")
    ax.axvline(shot_result["k"], color="green", ls="--", lw=1,
               label=f"Best k={shot_result['k']}")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia", color="blue")
    ax2.set_ylabel("Silhouette", color="red")
    ax.set_title("Elbow Method + Silhouette Score")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # ── Panel 3: Shot cluster profiles ────────────────────────────────────
    ax = axes[1, 0]
    clusters = shot_result["clusters"]
    names = [c["name"] for c in clusters]
    goal_rates = [c["goal_rate"] for c in clusters]
    mean_dists = [c["mean_shot_dist"] for c in clusters]
    x = np.arange(len(clusters))
    w = 0.35
    b1 = ax.bar(x - w/2, goal_rates, w, color="green", alpha=0.7, label="Goal rate")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w/2, mean_dists, w, color="steelblue", alpha=0.7, label="Mean dist (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Goal rate", color="green")
    ax2.set_ylabel("Mean shot distance (m)", color="steelblue")
    ax.set_title("Shot Cluster Profiles")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # ── Panel 4: Player clusters (PCA) ────────────────────────────────────
    ax = axes[1, 1]
    X_2d = player_result["X_2d"]
    labels = np.array(player_result["labels"])
    pnames = player_result["player_names"]
    for c in range(player_result["k"]):
        mask = labels == c
        top = player_result["clusters"][c]["top_players"][:2]
        label = f"C{c} ({', '.join(top)})"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=20, alpha=0.6,
                   color=colors[c], label=label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Player Finishing Profiles (PCA)")
    ax.legend(fontsize=6, markerscale=2)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  K-Means Clustering Analysis")
    print("=" * 64)

    print("\n── Loading graphs ──────────────────────────────────────────")
    graphs = load_graphs()
    data = extract_features(graphs)

    # ── Shot clustering ───────────────────────────────────────────────────
    print("\n── Shot archetype clustering ────────────────────────────────")
    shot_result = cluster_shots(data["X"], data["y"], data["sb_xg"])

    # ── Player clustering ─────────────────────────────────────────────────
    print("\n── Player profile clustering ────────────────────────────────")
    player_result = cluster_players(
        data["X"], data["y"], data["sb_xg"],
        data["players"], data["teams"], data["comps"],
    )

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "description": "K-Means clustering on shot features and player profiles",
        "shot_clustering": {
            "k": shot_result["k"],
            "clusters": shot_result["clusters"],
            "silhouettes": [round(s, 4) for s in shot_result["silhouettes"]],
        },
        "player_clustering": {
            "k": player_result["k"],
            "clusters": player_result["clusters"],
        },
    }
    out_path = PROCESSED / "clustering_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    # ── Figure ────────────────────────────────────────────────────────────
    print("\n── Generating figure …")
    fig = make_figure(shot_result, player_result)
    fig_path = ASSETS / "fig_clustering.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
