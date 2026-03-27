#!/usr/bin/env python3
"""
train_gat_preshotonly.py
------------------------
Train HybridGATv2 with 18-dim pre-shot-only metadata (no shot_placement).

shot_placement (dims [18:27]) is a PSxG feature — post-shot information. This
script produces a clean pre-shot-only model to populate the corresponding row in
Table 1 so reviewers can judge how much of the Brier improvement is from that
post-shot leak vs genuine graph contribution.

Outputs
-------
  data/processed/pool_7comp_hybrid_gat_no_plc_xg.pt    model weights
  data/processed/pool_7comp_per_comp_T_gat_no_plc.pt   per-comp T dict
  data/processed/pool_7comp_gat_no_plc_T.pt            global T scalar

Usage
-----
  python scripts/train_gat_preshotonly.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.models.hybrid_gat import HybridGATModel
from src.calibration import TemperatureScaler

PROCESSED = REPO_ROOT / "data" / "processed"
SEED   = 42
EPOCHS = 120
BATCH  = 64
LR     = 1e-3
WD     = 1e-4
META_DIM_NOPLC = 18   # 27 minus the 9 shot_placement dims

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Data helpers (mirrors stratified_split in train_xg_hybrid.py exactly)
# ---------------------------------------------------------------------------

def load_graphs() -> list:
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
        print(f"  {pt.name}: {len(gs)}")
    print(f"  Total: {len(graphs)}")
    return graphs


def stratified_split(graphs, train_frac=0.70, val_frac=0.15):
    rng = random.Random(SEED)
    goals    = [g for g in graphs if g.y.item() == 1]
    no_goals = [g for g in graphs if g.y.item() == 0]
    rng.shuffle(goals); rng.shuffle(no_goals)

    def split(lst):
        t = int(len(lst) * train_frac)
        v = int(len(lst) * (train_frac + val_frac))
        return lst[:t], lst[t:v], lst[v:]

    g_tr, g_va, g_te = split(goals)
    n_tr, n_va, n_te = split(no_goals)
    tr = g_tr + n_tr; rng.shuffle(tr)
    va = g_va + n_va; rng.shuffle(va)
    te = g_te + n_te; rng.shuffle(te)
    return tr, va, te


# ---------------------------------------------------------------------------
# 18-dim metadata tensor (everything except shot_placement)
# ---------------------------------------------------------------------------

def meta18(batch) -> torch.Tensor:
    """18-dim pre-shot-only metadata — drops shot_placement (dims [18:27])."""
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
        batch.gk_perp_offset.squeeze(),
        batch.n_def_direct_line.squeeze(),
        batch.is_right_foot.squeeze(),
    ], dim=1)
    return torch.cat([base, tech, gk, new], dim=1)   # [n, 18]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, opt, pos_weight):
    model.train()
    total = 0.0
    for batch in loader:
        meta      = meta18(batch)
        edge_attr = batch.edge_attr if batch.edge_attr is not None else None
        opt.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch, meta,
                       edge_attr=edge_attr)
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(), batch.y.squeeze().float(), pos_weight=pos_weight
        )
        loss.backward()
        opt.step()
        total += loss.detach().item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        meta      = meta18(batch)
        edge_attr = batch.edge_attr if batch.edge_attr is not None else None
        logits_all.append(model(batch.x, batch.edge_index, batch.batch, meta,
                                edge_attr=edge_attr).squeeze().cpu())
        y_all.append(batch.y.squeeze().cpu())
    logits = torch.cat(logits_all)
    y      = torch.cat(y_all).numpy()
    probs  = torch.sigmoid(logits).numpy()
    auc    = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    return auc, probs, y


# ---------------------------------------------------------------------------
# Per-competition temperature calibration
# ---------------------------------------------------------------------------

def fit_per_comp_T(model, val_g) -> dict[str, float]:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for g in val_g:
        cl = getattr(g, "comp_label", "") or "unknown"
        groups[cl].append(g)
    per_T = {}
    for cl, graphs in sorted(groups.items()):
        if len(graphs) < 20:
            continue
        loader = DataLoader(graphs, batch_size=BATCH)
        s = TemperatureScaler(model, init_T=1.5)
        s.fit(loader, device="cpu")
        per_T[cl] = s.temperature
        print(f"    {cl:25s}: T={s.temperature:.4f}  (n={len(graphs)})")
    return per_T


def compute_ece(y_true, y_prob, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  HybridGATv2 — Pre-shot-only (18-dim, no shot_placement)")
    print("=" * 64)

    # 1. Data
    print("\n── Loading graphs ───────────────────────────────────────────")
    graphs   = load_graphs()
    train_g, val_g, test_g = stratified_split(graphs)
    in_ch    = train_g[0].x.shape[1]
    edge_ch  = (train_g[0].edge_attr.shape[1]
                if train_g[0].edge_attr is not None else 4)
    y_train  = np.array([g.y.item() for g in train_g])
    n0, n1   = (y_train == 0).sum(), (y_train == 1).sum()
    pw       = torch.tensor([n0 / max(n1, 1)], dtype=torch.float)
    print(f"  train={len(train_g)} val={len(val_g)} test={len(test_g)}")
    print(f"  pos_weight={pw.item():.2f}  in_ch={in_ch}  edge_ch={edge_ch}")

    # 2. Model
    model = HybridGATModel(
        node_in=in_ch, edge_dim=edge_ch,
        meta_dim=META_DIM_NOPLC, hidden=32, heads=4, n_layers=3, dropout=0.3,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # 3. Train
    print("\n── Training ─────────────────────────────────────────────────")
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=12, factor=0.5, min_lr=1e-5)
    tr_loader = DataLoader(train_g, batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(val_g,   batch_size=BATCH)
    best_auc, best_state = 0.0, None

    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(model, tr_loader, opt, pw)
        auc, _, _ = evaluate(model, va_loader)
        sched.step(1 - auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"  ep={ep:3d}  loss={loss:.4f}  val_auc={auc:.4f}")

    model.load_state_dict(best_state)
    print(f"  Best val AUC: {best_auc:.4f}")

    # 4. Save weights
    ckpt_path = PROCESSED / "pool_7comp_hybrid_gat_no_plc_xg.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Saved → {ckpt_path.name}")

    # 5. Global T
    scaler = TemperatureScaler(model, init_T=1.5)
    scaler.fit(DataLoader(val_g, batch_size=BATCH), device="cpu")
    T_global = scaler.temperature
    torch.save({"T": torch.tensor(T_global)},
               PROCESSED / "pool_7comp_gat_no_plc_T.pt")
    print(f"  Global T={T_global:.4f}")

    # 6. Per-competition T
    print("\n── Per-Competition T ────────────────────────────────────────")
    per_T = fit_per_comp_T(model, val_g)
    torch.save(per_T, PROCESSED / "pool_7comp_per_comp_T_gat_no_plc.pt")

    # 7. Test evaluation
    print("\n── Test Evaluation ──────────────────────────────────────────")
    te_loader = DataLoader(test_g, batch_size=BATCH)
    _, probs_raw, y_test = evaluate(model, te_loader)

    # Apply per-comp T on test set
    probs_cal = probs_raw.copy()
    comp_labels = [getattr(g, "comp_label", "unknown") or "unknown" for g in test_g]
    logit_raw   = np.log(probs_raw / (1 - probs_raw + 1e-9) + 1e-9)
    for i, cl in enumerate(comp_labels):
        T = per_T.get(cl, T_global)
        probs_cal[i] = 1 / (1 + np.exp(-logit_raw[i] / T))

    auc_raw  = roc_auc_score(y_test, probs_raw)
    auc_cal  = roc_auc_score(y_test, probs_cal)
    brier_raw = brier_score_loss(y_test, probs_raw)
    brier_cal = brier_score_loss(y_test, probs_cal)
    ece_raw  = compute_ece(y_test, probs_raw)
    ece_cal  = compute_ece(y_test, probs_cal)
    ap_cal   = average_precision_score(y_test, probs_cal)
    sb_xg    = np.array([g.sb_xg.item() for g in test_g])
    auc_sb   = roc_auc_score(y_test, sb_xg)

    print(f"\n  {'Model':<45} {'AUC':>6}  {'Brier':>6}  {'ECE':>6}  {'AP':>6}")
    print(f"  {'-'*70}")
    print(f"  {'HybridGAT+T (18-dim, no shot_placement)':<45} "
          f"{auc_cal:.3f}  {brier_cal:.3f}  {ece_cal:.3f}  {ap_cal:.3f}")
    print(f"  {'HybridGAT+T (27-dim, with placement) — ref':<45} "
          f"0.760  0.148  0.215  0.344")
    print(f"  {'StatsBomb xG — ref':<45} "
          f"{auc_sb:.3f}  0.076    —      0.432")

    print(f"\n  Placement contribution to Brier:  "
          f"0.148 → {brier_cal:.3f}  (Δ {brier_cal - 0.148:+.3f})")
    print(f"  AUC without PSxG placement:  {auc_cal:.3f}  "
          f"(vs 0.760 with placement)")

    # Save summary for ablation table
    results = {
        "model":     "HybridGAT+T (18-dim, no shot_placement)",
        "auc":       round(auc_cal, 4),
        "brier":     round(brier_cal, 4),
        "ece":       round(ece_cal, 4),
        "ap":        round(ap_cal, 4),
        "T_global":  round(T_global, 4),
        "per_comp_T": {k: round(v, 4) for k, v in per_T.items()},
        "note":      "pre-shot-only: no shot_placement (PSxG post-shot feature)",
    }
    out_path = PROCESSED / "preshotonly_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary → {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
