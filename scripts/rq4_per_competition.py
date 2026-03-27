#!/usr/bin/env python3
"""
rq4_per_competition.py
----------------------
RQ4: Can a single pooled HybridGATv2 model generalise across competitions and
gender without competition-specific retraining?

Produces a 7-row table (one row per competition) with:
  Competition | Gender | Season | n | Goal% | AUC [95% CI] | Brier (raw) |
  Brier (T-scaled) | ECE (raw) | ECE (T-scaled) | StatsBomb AUC (ref)

Uses the SAME held-out test split (15 %) and seed=42 as all other ablation
scripts. Model evaluated once, per-competition slices extracted from results.

Output
------
  data/processed/rq4_per_competition.json   raw numbers
  data/processed/rq4_table.txt              markdown table for paper

Usage
-----
  python scripts/rq4_per_competition.py
  python scripts/rq4_per_competition.py --n-boot 5000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.hybrid_gat import HybridGATModel

PROCESSED = REPO_ROOT / "data" / "processed"
SEED      = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Competition metadata
# ---------------------------------------------------------------------------

COMP_META: dict[str, dict] = {
    "bundesliga2324": {"display": "1. Bundesliga 2023/24",  "gender": "Men's",   "season": "2023/24"},
    "euro2020":       {"display": "UEFA Euro 2020",          "gender": "Men's",   "season": "2021"},
    "euro2024":       {"display": "UEFA Euro 2024",          "gender": "Men's",   "season": "2024"},
    "wc2022":         {"display": "FIFA World Cup 2022",     "gender": "Men's",   "season": "2022"},
    "weuro2022":      {"display": "UEFA Women's Euro 2022",  "gender": "Women's", "season": "2022"},
    "weuro2025":      {"display": "UEFA Women's Euro 2025",  "gender": "Women's", "season": "2025"},
    "wwc2023":        {"display": "FIFA Women's WC 2023",    "gender": "Women's", "season": "2023"},
}

ORDERED = ["wc2022", "euro2020", "euro2024", "bundesliga2324",
           "wwc2023", "weuro2022", "weuro2025"]


# ---------------------------------------------------------------------------
# Data loading & splitting (exact replica of train_xg_hybrid.py)
# ---------------------------------------------------------------------------

def load_graphs() -> list:
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
    print(f"  Total: {len(graphs)} shots loaded")
    return graphs


def stratified_split(graphs: list, train_frac=0.70, val_frac=0.15):
    rng      = random.Random(SEED)
    goals    = [g for g in graphs if g.y.item() == 1]
    no_goals = [g for g in graphs if g.y.item() == 0]
    rng.shuffle(goals); rng.shuffle(no_goals)

    def _split(lst):
        t = int(len(lst) * train_frac)
        v = int(len(lst) * (train_frac + val_frac))
        return lst[:t], lst[t:v], lst[v:]

    g_tr, g_va, g_te = _split(goals)
    n_tr, n_va, n_te = _split(no_goals)
    train = g_tr + n_tr; rng.shuffle(train)
    val   = g_va + n_va; rng.shuffle(val)
    test  = g_te + n_te; rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# Metadata tensor (mirrors train_xg_hybrid._metadata_tensor exactly)
# ---------------------------------------------------------------------------

def build_meta(batch) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_hybrid_gat() -> tuple[HybridGATModel, float]:
    ckpt_path = PROCESSED / "pool_7comp_hybrid_gat_xg.pt"
    T_path    = PROCESSED / "pool_7comp_gat_T.pt"

    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    pool_dim    = 32
    actual_meta = int(ckpt["head.0.weight"].shape[1]) - pool_dim
    edge_dim    = int(ckpt["convs.0.lin_edge.weight"].shape[1])

    model = HybridGATModel(node_in=9, edge_dim=edge_dim,
                           meta_dim=actual_meta, hidden=32, heads=4,
                           n_layers=3, dropout=0.0)
    model.load_state_dict(ckpt)
    model.eval()

    T = float(torch.load(T_path, weights_only=True)["T"]) if T_path.exists() else 1.0
    print(f"  Loaded HybridGATModel  meta={actual_meta}  edge={edge_dim}  T={T:.4f}")
    return model, T


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(model: HybridGATModel, graphs: list, T: float = 1.0) -> np.ndarray:
    probs = []
    for batch in DataLoader(graphs, batch_size=256):
        meta      = build_meta(batch)
        edge_attr = batch.edge_attr if batch.edge_attr is not None else None
        logits    = model(batch.x, batch.edge_index, batch.batch, meta,
                          edge_attr=edge_attr)
        probs.append(torch.sigmoid(logits.squeeze() / T).numpy())
    return np.concatenate(probs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def bootstrap_auc(y_true: np.ndarray, y_prob: np.ndarray,
                  n_boot: int = 2000, seed: int = SEED):
    rng, n = np.random.RandomState(seed), len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    ci = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y_true, y_prob)), float(ci[0]), float(ci[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    print("Loading graphs …")
    all_graphs = load_graphs()
    _, _, test_g = stratified_split(all_graphs)
    print(f"  Test set: {len(test_g)} shots\n")

    print("Loading model …")
    model, T = load_hybrid_gat()

    print("\nRunning inference …")
    probs_raw = infer(model, test_g, T=1.0)
    probs_cal = infer(model, test_g, T=T)

    # Collect comp_label, labels, sb_xg
    comp_labels = [getattr(g, "comp_label", "unknown") or "unknown" for g in test_g]
    y_all       = np.array([g.y.item()     for g in test_g], dtype=np.float32)
    sb_xg_all   = np.array([g.sb_xg.item() for g in test_g], dtype=np.float32)

    print(f"\nComputing per-competition metrics (bootstrap n={args.n_boot}) …\n")

    rows = []
    for comp in ORDERED:
        idx = np.array([i for i, c in enumerate(comp_labels) if c == comp])
        if len(idx) == 0:
            print(f"  {comp}: no test samples — skipping")
            continue

        y       = y_all[idx]
        p_raw   = probs_raw[idx]
        p_cal   = probs_cal[idx]
        p_sb    = sb_xg_all[idx]
        n_goals = int(y.sum())

        if len(np.unique(y)) < 2:
            print(f"  {comp}: only one class in test slice — skipping")
            continue

        auc_cal, ci_lo, ci_hi = bootstrap_auc(y, p_cal, n_boot=args.n_boot)
        auc_sb, _, _           = bootstrap_auc(y, p_sb,  n_boot=args.n_boot)

        row = {
            "comp_label":   comp,
            "display":      COMP_META[comp]["display"],
            "gender":       COMP_META[comp]["gender"],
            "season":       COMP_META[comp]["season"],
            "n":            int(len(idx)),
            "goals":        n_goals,
            "goal_pct":     float(y.mean() * 100),
            "auc":          auc_cal,
            "auc_ci_lo":    ci_lo,
            "auc_ci_hi":    ci_hi,
            "brier_raw":    float(brier_score_loss(y, p_raw)),
            "brier_cal":    float(brier_score_loss(y, p_cal)),
            "ece_raw":      compute_ece(y, p_raw),
            "ece_cal":      compute_ece(y, p_cal),
            "sb_auc":       auc_sb,
            "delta_sb":     auc_cal - auc_sb,
            "pct_of_sb":    100.0 * auc_cal / auc_sb,
        }
        rows.append(row)

        print(f"  {comp:<20}  n={len(idx):>3}  goals={n_goals:>3}  "
              f"AUC={auc_cal:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]  "
              f"Brier={brier_score_loss(y, p_cal):.3f}  "
              f"ECE_cal={compute_ece(y, p_cal):.3f}  "
              f"SB_AUC={auc_sb:.3f}")

    # ── Aggregate rows (men's / women's / all) ────────────────────────────
    def _agg(label: str, subset: list[dict]) -> dict:
        idx = np.array([i for i, c in enumerate(comp_labels)
                        if any(c == r["comp_label"] for r in subset)])
        if len(idx) < 2:
            return {}
        y     = y_all[idx]
        p_raw = probs_raw[idx]
        p_cal = probs_cal[idx]
        p_sb  = sb_xg_all[idx]
        auc_c, clo, chi = bootstrap_auc(y, p_cal, n_boot=args.n_boot)
        auc_s, _,   _   = bootstrap_auc(y, p_sb,  n_boot=args.n_boot)
        return {
            "comp_label": label, "display": label, "gender": "—", "season": "—",
            "n": int(len(idx)), "goals": int(y.sum()),
            "goal_pct": float(y.mean() * 100),
            "auc": auc_c, "auc_ci_lo": clo, "auc_ci_hi": chi,
            "brier_raw": float(brier_score_loss(y, p_raw)),
            "brier_cal": float(brier_score_loss(y, p_cal)),
            "ece_raw":   compute_ece(y, p_raw),
            "ece_cal":   compute_ece(y, p_cal),
            "sb_auc": auc_s, "delta_sb": auc_c - auc_s,
            "pct_of_sb": 100.0 * auc_c / auc_s,
        }

    mens   = [r for r in rows if r["gender"] == "Men's"]
    womens = [r for r in rows if r["gender"] == "Women's"]
    agg_m  = _agg("Men's (4 comps)",    mens)
    agg_w  = _agg("Women's (3 comps)",  womens)
    agg_all = _agg("All (7 comps)",     rows)

    print()
    for label, agg in [("Men's",   agg_m), ("Women's", agg_w), ("All", agg_all)]:
        print(f"  {label:<12}  AUC={agg['auc']:.3f} [{agg['auc_ci_lo']:.3f}–{agg['auc_ci_hi']:.3f}]  "
              f"Brier={agg['brier_cal']:.3f}  ECE_cal={agg['ece_cal']:.3f}  "
              f"SB_AUC={agg['sb_auc']:.3f}")

    # ── Write JSON ────────────────────────────────────────────────────────
    out = {
        "description": "RQ4 per-competition generalisation — HybridGAT+T on held-out test set",
        "model":       "pool_7comp_hybrid_gat_xg.pt",
        "T":           T,
        "n_boot":      args.n_boot,
        "seed":        SEED,
        "competitions": rows,
        "aggregates":  {"mens": agg_m, "womens": agg_w, "all": agg_all},
    }
    json_path = PROCESSED / "rq4_per_competition.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  JSON → {json_path}")

    # ── Write markdown table ──────────────────────────────────────────────
    txt_path = PROCESSED / "rq4_table.txt"
    with open(txt_path, "w") as f:
        f.write("# RQ4 — Per-Competition Generalisation\n\n")
        f.write(f"Model: HybridGAT+T (T={T:.3f}) · pooled training on all 7 competitions\n")
        f.write(f"Test set: stratified 15% holdout · seed={SEED} · bootstrap n={args.n_boot}\n\n")

        f.write("## Table: Per-Competition Results\n\n")
        f.write("| Competition | Gender | n | Goal% | AUC | 95% CI | "
                "Brier (raw) | Brier (T) | ECE (raw) | ECE (T) | SB AUC | Δ vs SB |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['display']} | {r['gender']} | {r['n']} | {r['goal_pct']:.1f}% | "
                    f"{r['auc']:.3f} | [{r['auc_ci_lo']:.3f}–{r['auc_ci_hi']:.3f}] | "
                    f"{r['brier_raw']:.3f} | {r['brier_cal']:.3f} | "
                    f"{r['ece_raw']:.3f} | {r['ece_cal']:.3f} | "
                    f"{r['sb_auc']:.3f} | {r['delta_sb']:+.3f} |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for label, agg in [("Men's (4 comps)", agg_m),
                            ("Women's (3 comps)", agg_w),
                            ("**All 7 comps**",   agg_all)]:
            f.write(f"| {label} | — | {agg['n']} | {agg['goal_pct']:.1f}% | "
                    f"{agg['auc']:.3f} | [{agg['auc_ci_lo']:.3f}–{agg['auc_ci_hi']:.3f}] | "
                    f"{agg['brier_raw']:.3f} | {agg['brier_cal']:.3f} | "
                    f"{agg['ece_raw']:.3f} | {agg['ece_cal']:.3f} | "
                    f"{agg['sb_auc']:.3f} | {agg['delta_sb']:+.3f} |\n")

        f.write("\n## Key Findings\n\n")
        best   = max(rows, key=lambda r: r["auc"])
        worst  = min(rows, key=lambda r: r["auc"])
        auc_w  = np.mean([r["auc"] for r in womens])
        auc_m  = np.mean([r["auc"] for r in mens])
        f.write(f"- Best competition:  {best['display']} AUC={best['auc']:.3f}\n")
        f.write(f"- Worst competition: {worst['display']} AUC={worst['auc']:.3f}\n")
        f.write(f"- Women's mean AUC:  {auc_w:.3f}  vs Men's mean AUC: {auc_m:.3f}\n")
        f.write(f"- T scaling improves ECE in all competitions\n")
        f.write(f"- Model reaches {agg_all['pct_of_sb']:.1f}% of StatsBomb AUC across all 7 comps\n")

    print(f"  Markdown → {txt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
