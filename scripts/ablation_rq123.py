#!/usr/bin/env python3
"""
ablation_rq123.py
-----------------
Publication-ready ablation covering three research questions:

  RQ1 – Three-way ablation: LR-meta-27d vs GCN-only vs HybridGATv2+T
         → AUC, Brier, ECE with bootstrap 95% CIs
  RQ2 – ECE before/after temperature scaling, per-competition breakdown
  RQ3 – GK precision feature ablation (drop gk_perp_offset + n_def_direct_line)

All evaluations run on the SAME held-out test split (15 %) used during training.
No model weights are modified; all checkpoints are loaded read-only.

Outputs
-------
  data/processed/ablation_results.json   full numeric results
  data/processed/ablation_table.txt      markdown tables for paper

Usage
-----
  python scripts/ablation_rq123.py
  python scripts/ablation_rq123.py --n-boot 5000   # more bootstrap resamples
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.gcn import FootballGCN
from src.models.hybrid_gat import HybridGATModel

PROCESSED = REPO_ROOT / "data" / "processed"
SEED = 42
BATCH = 256
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading & splitting (mirrors train_xg_hybrid.py exactly)
# ---------------------------------------------------------------------------

def load_all_graphs() -> list:
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
        print(f"  {pt.name}: {len(gs)} shots")
    print(f"  Total: {len(graphs)} shots")
    return graphs


def stratified_split(graphs: list, train_frac=0.70, val_frac=0.15, seed=SEED):
    """Exact replica of stratified_split() in train_xg_hybrid.py."""
    rng = random.Random(seed)
    goals    = [g for g in graphs if g.y.item() == 1]
    no_goals = [g for g in graphs if g.y.item() == 0]
    rng.shuffle(goals)
    rng.shuffle(no_goals)

    def split_list(lst):
        t = int(len(lst) * train_frac)
        v = int(len(lst) * (train_frac + val_frac))
        return lst[:t], lst[t:v], lst[v:]

    g_tr, g_va, g_te = split_list(goals)
    n_tr, n_va, n_te = split_list(no_goals)
    train = g_tr + n_tr; rng.shuffle(train)
    val   = g_va + n_va; rng.shuffle(val)
    test  = g_te + n_te; rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def build_meta_tensor(batch) -> torch.Tensor:
    """27-dim metadata — mirrors _metadata_tensor() in train_xg_hybrid.py."""
    n = batch.shot_dist.shape[0]

    def _safe(attr, default):
        if hasattr(batch, attr):
            return getattr(batch, attr).squeeze()
        return torch.full((n,), default)

    base = torch.stack([
        batch.shot_dist.squeeze(),
        batch.shot_angle.squeeze(),
        batch.is_header.squeeze().float(),
        batch.is_open_play.squeeze().float(),
    ], dim=1)
    tech = batch.technique.view(-1, 8)
    gk   = torch.stack([
        batch.gk_dist.squeeze(),
        batch.n_def_in_cone.squeeze(),
        batch.gk_off_centre.squeeze(),
    ], dim=1)
    new  = torch.stack([
        _safe("gk_perp_offset",    3.0),
        _safe("n_def_direct_line", 0.0),
        _safe("is_right_foot",     0.5),
    ], dim=1)
    plc = (batch.shot_placement.view(-1, 9)
           if hasattr(batch, "shot_placement")
           else torch.zeros(n, 9))
    return torch.cat([base, tech, gk, new, plc], dim=1)  # [n, 27]


def collect_meta_np(graphs: list) -> np.ndarray:
    """Return all 27-dim metadata as numpy array, one row per graph."""
    loader = DataLoader(graphs, batch_size=BATCH)
    rows = []
    with torch.no_grad():
        for batch in loader:
            rows.append(build_meta_tensor(batch).numpy())
    return np.vstack(rows)  # [N, 27]


def collect_labels(graphs: list) -> np.ndarray:
    return np.array([g.y.item() for g in graphs], dtype=np.float32)


def collect_sb_xg(graphs: list) -> np.ndarray:
    return np.array([g.sb_xg.item() for g in graphs], dtype=np.float32)


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (uniform-width bins, weighted by bin size)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Bootstrap CI for AUC
# ---------------------------------------------------------------------------

def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                     n_boot: int = 2000, seed: int = SEED):
    """Return (auc, ci_low, ci_high) with 95 % bootstrap CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    ci = np.percentile(boot_aucs, [2.5, 97.5])
    return float(roc_auc_score(y_true, y_prob)), float(ci[0]), float(ci[1])


# ---------------------------------------------------------------------------
# GNN inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_gcn(model: FootballGCN, graphs: list) -> np.ndarray:
    model.eval()
    probs_list = []
    for batch in DataLoader(graphs, batch_size=BATCH):
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs_list.append(torch.sigmoid(logits.squeeze()).numpy())
    return np.concatenate(probs_list)


@torch.no_grad()
def infer_hybrid_gat(model: HybridGATModel, graphs: list,
                     T: float = 1.0,
                     zero_dims: list[int] | None = None) -> np.ndarray:
    """
    Run HybridGAT inference, optionally zeroing specific metadata dims.
    zero_dims: list of meta indices to zero out (GK ablation).
    """
    model.eval()
    probs_list = []
    for batch in DataLoader(graphs, batch_size=BATCH):
        meta = build_meta_tensor(batch)
        if zero_dims:
            meta = meta.clone()
            meta[:, zero_dims] = 0.0
        edge_attr = batch.edge_attr if batch.edge_attr is not None else None
        logits = model(batch.x, batch.edge_index, batch.batch, meta,
                       edge_attr=edge_attr)
        probs = torch.sigmoid(logits.squeeze() / T).numpy()
        probs_list.append(probs)
    return np.concatenate(probs_list)


# ---------------------------------------------------------------------------
# Per-competition breakdown
# ---------------------------------------------------------------------------

def per_comp_ece(graphs: list, probs_raw: np.ndarray,
                 probs_cal: np.ndarray) -> list[dict]:
    """ECE and AUC per competition, before and after T scaling."""
    comp_labels = [getattr(g, "comp_label", "unknown") or "unknown" for g in graphs]
    unique = sorted(set(comp_labels))
    rows = []
    for cl in unique:
        idx = np.array([i for i, c in enumerate(comp_labels) if c == cl])
        if len(idx) < 20:
            continue
        y = collect_labels([graphs[i] for i in idx])
        p_raw = probs_raw[idx]
        p_cal = probs_cal[idx]
        if len(np.unique(y)) < 2:
            continue
        rows.append({
            "competition":  cl,
            "n":            int(len(idx)),
            "goal_rate":    float(y.mean()),
            "auc":          float(roc_auc_score(y, p_cal)),
            "brier_raw":    float(brier_score_loss(y, p_raw)),
            "brier_cal":    float(brier_score_loss(y, p_cal)),
            "ece_raw":      compute_ece(y, p_raw),
            "ece_cal":      compute_ece(y, p_cal),
        })
    return rows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gcn(path: Path, in_channels: int = 9) -> FootballGCN:
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    # Auto-detect hidden_dim from first conv weight [hidden, in_channels]
    hidden_dim = int(ckpt["convs.0.lin.weight"].shape[0])
    n_layers   = sum(1 for k in ckpt if k.startswith("convs."))  // 2  # lin.weight + lin.bias
    model = FootballGCN(in_channels=in_channels, hidden_dim=hidden_dim,
                        out_channels=1, n_layers=n_layers, dropout=0.0)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  Loaded FootballGCN (in={in_channels}, hidden={hidden_dim}, "
          f"layers={n_layers}) from {path.name}")
    return model


def load_hybrid_gat(ckpt_path: Path, T_path: Path) -> tuple[HybridGATModel, float]:
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    pool_dim   = 32                      # hidden * heads / n_layers — fixed in training
    actual_meta = int(ckpt["head.0.weight"].shape[1]) - pool_dim
    in_channels = 9
    edge_dim    = int(ckpt["convs.0.lin_edge.weight"].shape[1])
    model = HybridGATModel(node_in=in_channels, edge_dim=edge_dim,
                           meta_dim=actual_meta, hidden=32, heads=4,
                           n_layers=3, dropout=0.0)
    model.load_state_dict(ckpt)
    model.eval()
    T = 1.0
    if T_path.exists():
        T = float(torch.load(T_path, weights_only=True)["T"])
    print(f"  Loaded HybridGATModel (meta={actual_meta}, edge={edge_dim}, T={T:.4f}) "
          f"from {ckpt_path.name}")
    return model, T


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_row(name: str, auc: float, ci_lo: float, ci_hi: float,
            brier: float, ece: float, ap: float) -> str:
    return (f"  {name:<42}  "
            f"{auc:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]  "
            f"{brier:.3f}  {ece:.3f}  {ap:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ1-3 ablation script")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap resamples for CI (default 2000)")
    args = parser.parse_args()

    print("=" * 72)
    print("  RQ1-3 Ablation — HybridGATv2 xG")
    print("=" * 72)

    # ── 1. Load data & split ───────────────────────────────────────────────
    print("\n── Loading graphs ────────────────────────────────────────────────")
    all_graphs = load_all_graphs()
    train_g, val_g, test_g = stratified_split(all_graphs)
    y_test   = collect_labels(test_g)
    sb_xg    = collect_sb_xg(test_g)
    print(f"\n  Train: {len(train_g)}  Val: {len(val_g)}  Test: {len(test_g)}")
    print(f"  Test goal rate: {y_test.mean():.3f}  goals: {int(y_test.sum())}")

    # ── 2. Load pre-trained models ─────────────────────────────────────────
    print("\n── Loading models ────────────────────────────────────────────────")
    gcn = load_gcn(PROCESSED / "pool_7comp_gcn_xg.pt", in_channels=9)
    hybrid_gat, T_gat = load_hybrid_gat(
        PROCESSED / "pool_7comp_hybrid_gat_xg.pt",
        PROCESSED / "pool_7comp_gat_T.pt",
    )

    # ── 3. LR baselines (metadata only, no graph) ──────────────────────────
    print("\n── Fitting LR baselines (metadata only) ──────────────────────────")
    X_trainval = collect_meta_np(train_g + val_g)
    y_trainval = collect_labels(train_g + val_g)
    X_test     = collect_meta_np(test_g)

    # LR-12d: basic (dist, angle, header, open_play + technique — indices 0:12)
    lr12 = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    lr12.fit(X_trainval[:, :12], y_trainval)
    probs_lr12 = lr12.predict_proba(X_test[:, :12])[:, 1]
    print("  LR-12d (basic metadata) fitted")

    # LR-27d: full metadata, no graph  ← the key RQ1 baseline
    lr27 = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    lr27.fit(X_trainval, y_trainval)
    probs_lr27 = lr27.predict_proba(X_test)[:, 1]
    print("  LR-27d (full metadata, no graph) fitted")

    # ── 4. GNN inference ──────────────────────────────────────────────────
    print("\n── GNN inference ─────────────────────────────────────────────────")
    probs_gcn     = infer_gcn(gcn, test_g)
    probs_gat_raw = infer_hybrid_gat(hybrid_gat, test_g, T=1.0)   # before T
    probs_gat_cal = infer_hybrid_gat(hybrid_gat, test_g, T=T_gat) # after T (RQ1 winner)

    # ── 5. GK ablation (RQ3) ──────────────────────────────────────────────
    # Dims [15] = gk_perp_offset, [16] = n_def_direct_line
    GK_PRECISION_DIMS = [15, 16]
    print("\n── GK precision feature ablation (RQ3) ───────────────────────────")
    probs_gat_nogk = infer_hybrid_gat(hybrid_gat, test_g, T=T_gat,
                                       zero_dims=GK_PRECISION_DIMS)
    # LR without GK precision features
    X_trainval_nogk = X_trainval.copy()
    X_trainval_nogk[:, GK_PRECISION_DIMS] = 0.0
    X_test_nogk     = X_test.copy()
    X_test_nogk[:, GK_PRECISION_DIMS] = 0.0
    lr27_nogk = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    lr27_nogk.fit(X_trainval_nogk, y_trainval)
    probs_lr27_nogk = lr27_nogk.predict_proba(X_test_nogk)[:, 1]
    print("  Ablation inference complete")

    # ── 6. Compute metrics with bootstrap CIs ─────────────────────────────
    print(f"\n── Computing metrics (bootstrap n={args.n_boot}) ─────────────────")

    models = [
        ("StatsBomb xG [industry ref]",         sb_xg),
        ("LR-12d [basic metadata, no graph]",   probs_lr12),
        ("LR-27d [full metadata, no graph]",    probs_lr27),
        ("GCN-only [graph spatial]",            probs_gcn),
        ("HybridGAT [graph+meta, no T]",        probs_gat_raw),
        ("HybridGAT+T [graph+meta+calib] ★",   probs_gat_cal),
        ("HybridGAT+T −GKprec [ablation]",      probs_gat_nogk),
        ("LR-27d −GKprec [ablation]",           probs_lr27_nogk),
    ]

    results = {}
    for name, probs in models:
        auc, ci_lo, ci_hi = bootstrap_auc_ci(y_test, probs, n_boot=args.n_boot)
        brier = float(brier_score_loss(y_test, probs))
        ece   = compute_ece(y_test, probs)
        ap    = float(average_precision_score(y_test, probs))
        results[name] = dict(auc=auc, ci_lo=ci_lo, ci_hi=ci_hi,
                             brier=brier, ece=ece, ap=ap)

    # ── 7. Per-competition breakdown (RQ2) ────────────────────────────────
    print("\n── Per-competition ECE (RQ2) ─────────────────────────────────────")
    per_comp = per_comp_ece(test_g, probs_gat_raw, probs_gat_cal)

    # ── 8. Print tables ───────────────────────────────────────────────────
    header = (f"  {'Model':<42}  "
              f"{'AUC [95% CI]':<24}  "
              f"{'Brier':>5}  {'ECE':>5}  {'AP':>5}")

    print(f"\n{'=' * 78}")
    print("  RQ1: Three-way Ablation — Test Set Results")
    print(f"{'=' * 78}")
    print(header)
    print(f"  {'-' * 76}")
    for name, m in results.items():
        print(fmt_row(name, m["auc"], m["ci_lo"], m["ci_hi"],
                      m["brier"], m["ece"], m["ap"]))

    # Key deltas
    auc_lr27  = results["LR-27d [full metadata, no graph]"]["auc"]
    auc_gcn   = results["GCN-only [graph spatial]"]["auc"]
    auc_gat   = results["HybridGAT+T [graph+meta+calib] ★"]["auc"]
    auc_sb    = results["StatsBomb xG [industry ref]"]["auc"]
    auc_nogk  = results["HybridGAT+T −GKprec [ablation]"]["auc"]

    print(f"\n  Key deltas (vs LR-27d baseline):")
    print(f"    GCN-only  vs LR-27d       : {auc_gcn - auc_lr27:+.3f} AUC")
    print(f"    HybridGAT+T vs LR-27d     : {auc_gat - auc_lr27:+.3f} AUC  ← RQ1 claim")
    print(f"    HybridGAT+T vs StatsBomb  : {auc_gat - auc_sb:+.3f} AUC  ({100*auc_gat/auc_sb:.1f}% of SB)")
    print(f"\n  RQ3: GK precision feature impact:")
    print(f"    HybridGAT+T full vs −GKprec: {auc_gat - auc_nogk:+.3f} AUC drop when removed")

    # ECE before/after T
    ece_raw = results["HybridGAT [graph+meta, no T]"]["ece"]
    ece_cal = results["HybridGAT+T [graph+meta+calib] ★"]["ece"]
    bri_raw = results["HybridGAT [graph+meta, no T]"]["brier"]
    bri_cal = results["HybridGAT+T [graph+meta+calib] ★"]["brier"]
    print(f"\n  RQ2: Temperature scaling impact (global T={T_gat:.3f}):")
    print(f"    ECE  {ece_raw:.4f} → {ece_cal:.4f}  (Δ {ece_cal - ece_raw:+.4f})")
    print(f"    Brier {bri_raw:.4f} → {bri_cal:.4f}  (Δ {bri_cal - bri_raw:+.4f})")

    # Per-competition table
    print(f"\n{'=' * 78}")
    print("  RQ2: Per-Competition ECE Before/After Temperature Scaling")
    print(f"{'=' * 78}")
    print(f"  {'Competition':<25}  {'n':>5}  {'GoalRate':>8}  "
          f"{'AUC':>5}  {'ECE_raw':>7}  {'ECE_cal':>7}  {'ΔECE':>6}")
    print(f"  {'-' * 74}")
    for row in sorted(per_comp, key=lambda r: r["competition"]):
        delta = row["ece_cal"] - row["ece_raw"]
        print(f"  {row['competition']:<25}  {row['n']:>5}  {row['goal_rate']:>8.3f}  "
              f"{row['auc']:>5.3f}  {row['ece_raw']:>7.4f}  {row['ece_cal']:>7.4f}  "
              f"{delta:>+6.4f}")

    # ── 9. Save results ───────────────────────────────────────────────────
    out = {
        "test_n":      int(len(test_g)),
        "test_goals":  int(y_test.sum()),
        "goal_rate":   float(y_test.mean()),
        "T_gat":       T_gat,
        "n_boot":      args.n_boot,
        "models":      results,
        "per_comp":    per_comp,
        "gk_precision_dims": GK_PRECISION_DIMS,
        "rq1_delta_hybrid_vs_lr27": float(auc_gat - auc_lr27),
        "rq1_delta_gcn_vs_lr27":    float(auc_gcn - auc_lr27),
        "rq1_pct_of_statsbomb":     float(100 * auc_gat / auc_sb),
        "rq2_ece_before_T":         ece_raw,
        "rq2_ece_after_T":          ece_cal,
        "rq3_gk_auc_drop":          float(auc_gat - auc_nogk),
    }

    json_path = PROCESSED / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {json_path}")

    # ── 10. Write markdown table for paper ────────────────────────────────
    txt_path = PROCESSED / "ablation_table.txt"
    with open(txt_path, "w") as f:
        f.write("# RQ1-3 Ablation — xG Model Comparison\n\n")
        f.write("## Table 1: Three-way Ablation (held-out test set)\n\n")
        f.write(f"Test set: n={len(test_g)}, goals={int(y_test.sum())} "
                f"({100*y_test.mean():.1f}%), 7 competitions\n\n")
        f.write("| Model | AUC | 95% CI | Brier | ECE | AP |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, m in results.items():
            f.write(f"| {name} | {m['auc']:.3f} | "
                    f"[{m['ci_lo']:.3f}–{m['ci_hi']:.3f}] | "
                    f"{m['brier']:.3f} | {m['ece']:.3f} | {m['ap']:.3f} |\n")
        f.write(f"\n★ HybridGAT+T reaches {100*auc_gat/auc_sb:.1f}% of StatsBomb xG AUC "
                f"(Δ AUC vs LR-27d: {auc_gat - auc_lr27:+.3f})\n")
        f.write("\n## Table 2: Per-Competition Calibration (RQ2)\n\n")
        f.write(f"Global temperature T={T_gat:.3f} applied after training\n\n")
        f.write("| Competition | n | Goal Rate | AUC | ECE (raw) | ECE (T-scaled) | ΔECE |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in sorted(per_comp, key=lambda r: r["competition"]):
            delta = row["ece_cal"] - row["ece_raw"]
            f.write(f"| {row['competition']} | {row['n']} | {row['goal_rate']:.3f} | "
                    f"{row['auc']:.3f} | {row['ece_raw']:.4f} | "
                    f"{row['ece_cal']:.4f} | {delta:+.4f} |\n")
        f.write("\n## RQ3: GK Precision Feature Ablation\n\n")
        f.write(f"Zeroing gk_perp_offset (dim 15) + n_def_direct_line (dim 16):\n\n")
        f.write(f"| Model | AUC | ΔAUC vs full |\n")
        f.write("|---|---|---|\n")
        f.write(f"| HybridGAT+T (full) | {auc_gat:.3f} | — |\n")
        f.write(f"| HybridGAT+T −GKprec | "
                f"{results['HybridGAT+T −GKprec [ablation]']['auc']:.3f} | "
                f"{results['HybridGAT+T −GKprec [ablation]']['auc'] - auc_gat:+.3f} |\n")
        f.write(f"| LR-27d (full) | {auc_lr27:.3f} | — |\n")
        f.write(f"| LR-27d −GKprec | "
                f"{results['LR-27d −GKprec [ablation]']['auc']:.3f} | "
                f"{results['LR-27d −GKprec [ablation]']['auc'] - auc_lr27:+.3f} |\n")
    print(f"  Markdown → {txt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
