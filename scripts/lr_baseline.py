#!/usr/bin/env python3
"""
lr_baseline.py
--------------
Metadata-only Logistic Regression baselines to quantify the marginal value
of the GNN graph component.

Three LR variants trained on the SAME stratified split as train_xg_hybrid.py:

  LR-basic    4 features: shot_dist, shot_angle, is_header, is_open_play
  LR-meta12   12 features: above + technique×8
  LR-meta27   27 features: full metadata vector (same dims as HybridGAT head input)

Also shows StatsBomb xG (industry reference) for context.

Results saved to data/processed/lr_baseline_results.json

Usage
-----
    python scripts/lr_baseline.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

REPO_ROOT  = Path(__file__).parent.parent
PROCESSED  = REPO_ROOT / "data" / "processed"
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading — mirrors train_xg_hybrid.py exactly
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
    """Identical stratified split to train_xg_hybrid.py."""
    rng    = random.Random(SEED)
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
# Feature extraction
# ---------------------------------------------------------------------------

def _safe(g, attr, default):
    """Get a scalar graph attribute, falling back to default if absent."""
    if hasattr(g, attr):
        val = getattr(g, attr)
        return float(val.item() if hasattr(val, "item") else val)
    return default


def extract_features(graphs: list) -> dict[str, np.ndarray]:
    """Return dict of feature matrices keyed by variant name."""
    basic, meta12, meta27, sb_xg, labels = [], [], [], [], []

    for g in graphs:
        dist      = float(g.shot_dist.item())
        angle     = float(g.shot_angle.item())
        header    = float(g.is_header.item())
        open_play = float(g.is_open_play.item())
        tech      = g.technique.tolist()            # 8-dim one-hot
        gk_dist   = float(g.gk_dist.item())
        n_def     = float(g.n_def_in_cone.item())
        gk_off    = float(g.gk_off_centre.item())
        gk_perp   = _safe(g, "gk_perp_offset",    3.0)
        n_direct  = _safe(g, "n_def_direct_line",  0.0)
        right_ft  = _safe(g, "is_right_foot",      0.5)
        plc       = (g.shot_placement.tolist()
                     if hasattr(g, "shot_placement")
                     else [0.0] * 9)                # 9-dim one-hot

        b4  = [dist, angle, header, open_play]
        gk3 = [gk_dist, n_def, gk_off]
        new3 = [gk_perp, n_direct, right_ft]

        basic.append(b4)
        meta12.append(b4 + tech)
        meta27.append(b4 + tech + gk3 + new3 + plc)
        sb_xg.append(float(g.sb_xg.item()))
        labels.append(float(g.y.item()))

    return {
        "basic":   np.array(basic,   dtype=np.float32),
        "meta12":  np.array(meta12,  dtype=np.float32),
        "meta27":  np.array(meta27,  dtype=np.float32),
        "sb_xg":   np.array(sb_xg,   dtype=np.float32),
        "y":       np.array(labels,   dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    """Mean absolute calibration error across equal-width probability bins."""
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


def evaluate(name: str, y_true, y_prob, width=30) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    bs  = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    print(f"  {name:<{width}}  AUC={auc:.4f}  AP={ap:.4f}  Brier={bs:.4f}  ECE={ece:.4f}")
    return {"name": name, "auc": auc, "ap": ap, "brier": bs, "ece": ece}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading graphs …")
    all_graphs = load_graphs()

    print("\nSplitting (stratified, seed=42) …")
    train_g, val_g, test_g = stratified_split(all_graphs)
    print(f"  train={len(train_g)}  val={len(val_g)}  test={len(test_g)}")

    # Extract features for each split
    tr  = extract_features(train_g)
    va  = extract_features(val_g)
    te  = extract_features(test_g)

    # LR trains on train+val (same convention as existing run_logreg)
    X_tr = {k: np.vstack([tr[k], va[k]])
            for k in ("basic", "meta12", "meta27")}
    y_tr = np.concatenate([tr["y"], va["y"]])
    X_te = te
    y_te = te["y"]

    results = []

    print("\n── Test-set performance ─────────────────────────────────────────────────")

    # StatsBomb xG reference (no training needed)
    sb_probs = np.concatenate([te["sb_xg"]])
    results.append(evaluate("StatsBomb xG (reference)", y_te, sb_probs))

    # Three LR variants
    for variant, n_feat in [("basic", 4), ("meta12", 12), ("meta27", 27)]:
        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
            solver="lbfgs",
        )
        lr.fit(X_tr[variant], y_tr)
        probs = lr.predict_proba(X_te[variant])[:, 1]
        label = f"LR ({n_feat} features)"
        results.append(evaluate(label, y_te, probs))

    # ── Delta table vs best LR-meta27 ────────────────────────────────────────
    lr27_auc   = next(r["auc"]   for r in results if "27" in r["name"])
    lr27_brier = next(r["brier"] for r in results if "27" in r["name"])

    print("\n── Delta vs LR-meta27 (what the GNN must beat) ──────────────────────────")
    print(f"  LR-meta27 AUC   = {lr27_auc:.4f}")
    print(f"  LR-meta27 Brier = {lr27_brier:.4f}")
    print()
    print("  Target thresholds for GNN to be 'clearly better':")
    print(f"    AUC   > {lr27_auc + 0.005:.4f}  (+0.5 pp minimum)")
    print(f"    Brier < {lr27_brier - 0.002:.4f}  (−0.002 minimum)")

    # ── Save results ──────────────────────────────────────────────────────────
    out = PROCESSED / "lr_baseline_results.json"
    with open(out, "w") as f:
        json.dump({
            "description": "Metadata-only LR baselines vs StatsBomb xG",
            "split": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
            "seed": SEED,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved → {out}")

    # ── Print summary for paper table ─────────────────────────────────────────
    print("\n── Paper table rows (copy-paste) ────────────────────────────────────────")
    print(f"  {'Model':<35} {'AUC':>6}  {'AP':>6}  {'Brier':>6}  {'ECE':>6}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in results:
        print(f"  {r['name']:<35} {r['auc']:6.4f}  {r['ap']:6.4f}  "
              f"{r['brier']:6.4f}  {r['ece']:6.4f}")


if __name__ == "__main__":
    main()
