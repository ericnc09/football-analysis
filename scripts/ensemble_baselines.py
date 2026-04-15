#!/usr/bin/env python3
"""
ensemble_baselines.py
---------------------
Random Forest and XGBoost xG baselines on the same 27-dim shot metadata
used by LR-27d and HybridGAT+T.

Uses identical data loading, stratified split (seed=42), and evaluation
metrics as lr_baseline.py for direct comparison with Table 1.

Results saved to data/processed/ensemble_baseline_results.json

Usage
-----
    python scripts/ensemble_baselines.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
SEED      = 42

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading — identical to lr_baseline.py
# ---------------------------------------------------------------------------

def load_graphs() -> list:
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
        print(f"  {pt.name}: {len(gs)} graphs")
    print(f"  Total: {len(graphs)} graphs")
    return graphs


def stratified_split(graphs: list, train_frac=0.70, val_frac=0.15):
    """Identical stratified split to train_xg_hybrid.py / lr_baseline.py."""
    rng      = random.Random(SEED)
    goals    = [g for g in graphs if g.y.item() == 1]
    no_goals = [g for g in graphs if g.y.item() == 0]
    rng.shuffle(goals)
    rng.shuffle(no_goals)

    def _split(lst):
        t = int(len(lst) * train_frac)
        v = int(len(lst) * (train_frac + val_frac))
        return lst[:t], lst[t:v], lst[v:]

    g_tr, g_va, g_te = _split(goals)
    n_tr, n_va, n_te = _split(no_goals)
    train = g_tr + n_tr;  rng.shuffle(train)
    val   = g_va + n_va;  rng.shuffle(val)
    test  = g_te + n_te;  rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Feature extraction — identical to lr_baseline.py
# ---------------------------------------------------------------------------

def _safe(g, attr, default):
    if hasattr(g, attr):
        val = getattr(g, attr)
        return float(val.item() if hasattr(val, "item") else val)
    return default


def extract_meta27(graphs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract 27-dim metadata, StatsBomb xG, and labels."""
    meta, sb_xg, labels = [], [], []

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

        b4  = [dist, angle, header, open_play]
        gk3 = [gk_dist, n_def, gk_off]
        new3 = [gk_perp, n_direct, right_ft]
        meta.append(b4 + tech + gk3 + new3 + plc)

        sb_xg.append(float(g.sb_xg.item()))
        labels.append(float(g.y.item()))

    return (np.array(meta, dtype=np.float32),
            np.array(sb_xg, dtype=np.float32),
            np.array(labels, dtype=np.float32))


# ---------------------------------------------------------------------------
# Metrics — identical to lr_baseline.py
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos  = y_true[mask].mean()
        mean_conf = y_prob[mask].mean()
        ece      += (mask.sum() / n) * abs(frac_pos - mean_conf)
    return float(ece)


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=2000, ci=0.95):
    """Bootstrap 95% confidence interval for a metric."""
    rng = np.random.RandomState(SEED)
    n   = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except ValueError:
            continue
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def evaluate(name: str, y_true, y_prob, width=30) -> dict:
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    bs   = brier_score_loss(y_true, y_prob)
    ece  = expected_calibration_error(y_true, y_prob)

    auc_lo, auc_hi     = bootstrap_ci(y_true, y_prob, roc_auc_score)
    brier_lo, brier_hi = bootstrap_ci(y_true, y_prob, brier_score_loss)

    print(f"  {name:<{width}}  AUC={auc:.4f} [{auc_lo:.3f}-{auc_hi:.3f}]  "
          f"AP={ap:.4f}  Brier={bs:.4f} [{brier_lo:.3f}-{brier_hi:.3f}]  ECE={ece:.4f}")

    return {
        "name": name, "auc": auc, "ap": ap, "brier": bs, "ece": ece,
        "auc_ci": [auc_lo, auc_hi], "brier_ci": [brier_lo, brier_hi],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading graphs …")
    all_graphs = load_graphs()

    print("\nSplitting (stratified, seed=42) …")
    train_g, val_g, test_g = stratified_split(all_graphs)
    print(f"  train={len(train_g)}  val={len(val_g)}  test={len(test_g)}")

    # Extract 27-dim features for each split
    X_tr, sb_tr, y_tr_raw = extract_meta27(train_g)
    X_va, sb_va, y_va_raw = extract_meta27(val_g)
    X_te, sb_te, y_te     = extract_meta27(test_g)

    # Train on train+val (same convention as lr_baseline.py)
    X_train = np.vstack([X_tr, X_va])
    y_train = np.concatenate([y_tr_raw, y_va_raw])

    goal_rate = y_train.mean()
    scale_pos = (1 - goal_rate) / goal_rate  # ~8.6 for ~10.4% goal rate

    results = []

    print("\n── Test-set performance ─────────────────────────────────────────────────")

    # StatsBomb xG reference
    results.append(evaluate("StatsBomb xG (reference)", y_te, sb_te))

    # --- Random Forest ---
    print("\n  Training Random Forest (500 trees) …")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_te)[:, 1]
    results.append(evaluate("Random Forest (27 features)", y_te, rf_probs))

    # RF feature importance (Gini)
    feat_names = (
        ["shot_dist", "shot_angle", "is_header", "is_open_play"]
        + [f"tech_{i}" for i in range(8)]
        + ["gk_dist", "n_def_in_cone", "gk_off_centre"]
        + ["gk_perp_offset", "n_def_direct_line", "is_right_foot"]
        + [f"placement_{i}" for i in range(9)]
    )
    rf_importance = dict(zip(feat_names,
                             rf.feature_importances_.tolist()))

    # --- XGBoost ---
    print("  Training XGBoost (500 rounds) …")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_te)[:, 1]
    results.append(evaluate("XGBoost (27 features)", y_te, xgb_probs))

    # XGBoost feature importance (gain)
    xgb_importance = dict(zip(feat_names,
                              xgb.feature_importances_.tolist()))

    # ── Comparison with LR-27d and HybridGAT+T ───────────────────────────────
    print("\n── Context (from existing baselines) ───────────────────────────────────")
    print("  LR-27d:        AUC ≈ 0.749  Brier ≈ 0.187")
    print("  HybridGAT+T:   AUC = 0.760  Brier = 0.148")
    print("  StatsBomb xG:  AUC = 0.794  Brier = 0.076")

    # ── Top features ──────────────────────────────────────────────────────────
    print("\n── Top 10 features by importance ────────────────────────────────────────")
    print(f"\n  {'Random Forest (Gini)':40} {'XGBoost (Gain)':40}")
    print(f"  {'-'*40} {'-'*40}")
    rf_top  = sorted(rf_importance.items(),  key=lambda x: -x[1])[:10]
    xgb_top = sorted(xgb_importance.items(), key=lambda x: -x[1])[:10]
    for (rf_name, rf_val), (xgb_name, xgb_val) in zip(rf_top, xgb_top):
        print(f"  {rf_name:30} {rf_val:.4f}    {xgb_name:30} {xgb_val:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = PROCESSED / "ensemble_baseline_results.json"
    with open(out, "w") as f:
        json.dump({
            "description": "RF and XGBoost baselines on 27-dim shot metadata",
            "split": {
                "train": len(train_g),
                "val": len(val_g),
                "test": len(test_g),
            },
            "seed": SEED,
            "results": results,
            "rf_feature_importance": rf_importance,
            "xgb_feature_importance": xgb_importance,
        }, f, indent=2)
    print(f"\nSaved → {out}")

    # ── Paper table ───────────────────────────────────────────────────────────
    print("\n── Paper table rows (copy-paste) ────────────────────────────────────────")
    print(f"  {'Model':<35} {'AUC':>6}  {'AUC 95% CI':>16}  "
          f"{'Brier':>6}  {'Brier 95% CI':>16}  {'ECE':>6}  {'AP':>6}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*16}  {'-'*6}  {'-'*16}  {'-'*6}  {'-'*6}")
    for r in results:
        ci_a = f"[{r['auc_ci'][0]:.3f}-{r['auc_ci'][1]:.3f}]"
        ci_b = f"[{r['brier_ci'][0]:.3f}-{r['brier_ci'][1]:.3f}]"
        print(f"  {r['name']:<35} {r['auc']:6.4f}  {ci_a:>16}  "
              f"{r['brier']:6.4f}  {ci_b:>16}  {r['ece']:6.4f}  {r['ap']:6.4f}")


if __name__ == "__main__":
    main()
