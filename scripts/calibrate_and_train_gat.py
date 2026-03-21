#!/usr/bin/env python3
"""
calibrate_and_train_gat.py
--------------------------
Fast post-training pipeline — skips retraining GCN / GAT / HybridGCN.

Steps
-----
1. Load all 7-competition shot graphs → same stratified split as main training
2. Load existing pool_7comp_hybrid_xg.pt (HybridGCN)
   → TemperatureScaler.fit(val_loader)  → pool_7comp_T.pt
3. Train HybridGATModel from scratch on train_g (~60-90 s)
   → pool_7comp_hybrid_gat_xg.pt
4. TemperatureScaler.fit(val_loader) on HybridGAT
   → pool_7comp_gat_T.pt
5. Print before/after Brier scores for both models

Usage
-----
    python scripts/calibrate_and_train_gat.py
"""

import sys
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.models.hybrid_gat import HybridGATModel
from src.calibration import TemperatureScaler

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SEED   = 42
DEVICE = torch.device("cpu")
EPOCHS = 120
BATCH  = 64
LR     = 1e-3
WD     = 1e-4
META_DIM = 15   # dist, angle, header, open_play + technique×8 + gk_dist, n_def_in_cone, gk_off_centre

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ── HybridXGModel (mirrors train_xg_hybrid.py) ───────────────────────────────
class HybridXGModel(nn.Module):
    def __init__(self, in_channels=9, hidden_dim=64, meta_dim=META_DIM, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_dim),
            GCNConv(hidden_dim,  hidden_dim),
            GCNConv(hidden_dim,  hidden_dim),
        ])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch, metadata):
        return self.head(torch.cat([self.encode(x, edge_index, batch), metadata], dim=1))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_graphs(paths):
    all_graphs = []
    for p in paths:
        gs = torch.load(p, weights_only=False)
        print(f"  {p.name}: {len(gs)} graphs")
        all_graphs.extend(gs)
    return all_graphs


def stratified_split(graphs, train_frac=0.70, val_frac=0.15, seed=SEED):
    rng = random.Random(seed)
    goals    = [g for g in graphs if g.y.item() == 1]
    no_goals = [g for g in graphs if g.y.item() == 0]
    rng.shuffle(goals);  rng.shuffle(no_goals)

    def split(lst):
        t = int(len(lst) * train_frac)
        v = int(len(lst) * (train_frac + val_frac))
        return lst[:t], lst[t:v], lst[v:]

    g_tr, g_va, g_te = split(goals)
    n_tr, n_va, n_te = split(no_goals)
    train = g_tr + n_tr;  rng.shuffle(train)
    val   = g_va + n_va;  rng.shuffle(val)
    test  = g_te + n_te;  rng.shuffle(test)
    return train, val, test


def _metadata_tensor(batch):
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
    return torch.cat([base, tech, gk], dim=1).to(DEVICE)


@torch.no_grad()
def eval_hybrid(model, loader):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        meta  = _metadata_tensor(batch)
        logits_all.append(model(batch.x, batch.edge_index, batch.batch, meta).squeeze().cpu())
        y_all.append(batch.y.squeeze().cpu())
    logits = torch.cat(logits_all)
    y      = torch.cat(y_all).numpy().astype(int)
    probs  = torch.sigmoid(logits).numpy()
    auc    = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    brier  = brier_score_loss(y, probs)
    return auc, brier, probs, y


# ── Step 1: load data ─────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  calibrate_and_train_gat.py")
    print("=" * 64)

    paths = sorted(PROCESSED_DIR.glob("statsbomb_*_shot_graphs.pt"))
    if not paths:
        print("ERROR: no shot graph files found. Run build_shot_graphs.py first.")
        sys.exit(1)

    print(f"\n[1/4] Loading {len(paths)} competitions…")
    all_graphs = load_graphs(paths)
    print(f"  Total: {len(all_graphs)} graphs")

    train_g, val_g, test_g = stratified_split(all_graphs)
    print(f"  Train={len(train_g)}  Val={len(val_g)}  Test={len(test_g)}")

    in_ch   = train_g[0].x.shape[1]
    edge_ch = train_g[0].edge_attr.shape[1] if train_g[0].edge_attr is not None else 0
    n1 = sum(int(g.y.item()) for g in train_g)
    n0 = len(train_g) - n1
    pos_weight = torch.tensor([n0 / max(n1, 1)], dtype=torch.float).to(DEVICE)
    print(f"  in_ch={in_ch}  edge_ch={edge_ch}  pos_weight={pos_weight.item():.2f}")

    val_loader  = DataLoader(val_g,  batch_size=BATCH)
    test_loader = DataLoader(test_g, batch_size=BATCH)

    # ── Step 2: calibrate existing HybridGCN ─────────────────────────────────
    print("\n[2/4] Calibrating HybridGCN…")
    gcn_path = PROCESSED_DIR / "pool_7comp_hybrid_xg.pt"
    if not gcn_path.exists():
        print(f"  WARNING: {gcn_path} not found — skipping HybridGCN calibration")
    else:
        gcn_model = HybridXGModel(in_channels=in_ch).to(DEVICE)
        gcn_model.load_state_dict(torch.load(gcn_path, weights_only=True, map_location="cpu"))
        gcn_model.eval()

        auc_raw, brier_raw, _, _ = eval_hybrid(gcn_model, test_loader)
        print(f"  HybridGCN raw:  AUC={auc_raw:.3f}  Brier={brier_raw:.4f}")

        scaler = TemperatureScaler(gcn_model, init_T=1.5)
        cal    = scaler.fit(val_loader, device=str(DEVICE))
        T_gcn  = scaler.temperature

        # Re-evaluate with T applied
        auc_cal, brier_cal, _, _ = eval_hybrid(scaler, test_loader)
        print(f"  HybridGCN +T:   AUC={auc_cal:.3f}  Brier={brier_cal:.4f}  (T={T_gcn:.4f})")
        print(f"  Brier improvement: {brier_raw:.4f} → {brier_cal:.4f}  "
              f"(Δ={brier_raw - brier_cal:+.4f})")

        scaler.save(PROCESSED_DIR / "pool_7comp_T.pt")

    # ── Step 3: train HybridGAT ───────────────────────────────────────────────
    print("\n[3/4] Training HybridGAT…")
    gat_model = HybridGATModel(
        node_in=in_ch, edge_dim=edge_ch,
        meta_dim=META_DIM, hidden=32, heads=4, n_layers=3, dropout=0.3,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in gat_model.parameters() if p.requires_grad)
    print(f"  HybridGATModel: {n_params:,} parameters")

    tr_loader = DataLoader(train_g, batch_size=BATCH, shuffle=True)
    opt   = torch.optim.Adam(gat_model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=12, factor=0.5, min_lr=1e-5)

    best_auc, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        gat_model.train()
        total_loss = 0.0
        for batch in tr_loader:
            batch = batch.to(DEVICE)
            meta  = _metadata_tensor(batch)
            opt.zero_grad()
            logits = gat_model(batch.x, batch.edge_index, batch.batch, meta,
                               edge_attr=batch.edge_attr if edge_ch > 0 else None)
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(), batch.y.squeeze().float(), pos_weight=pos_weight
            )
            loss.backward()
            opt.step()
            total_loss += loss.detach().item() * batch.num_graphs

        ep_loss = total_loss / len(tr_loader.dataset)
        auc_v, _, _, _ = eval_hybrid(gat_model, val_loader)
        sched.step(1 - auc_v)

        if auc_v > best_auc:
            best_auc   = auc_v
            best_state = {k: v.clone() for k, v in gat_model.state_dict().items()}

        if ep % 20 == 0:
            print(f"  ep={ep:3d}  loss={ep_loss:.4f}  val_auc={auc_v:.3f}  best={best_auc:.3f}")

    gat_model.load_state_dict(best_state)
    auc_te, brier_te, _, _ = eval_hybrid(gat_model, test_loader)
    print(f"\n  HybridGAT test: AUC={auc_te:.3f}  Brier={brier_te:.4f}")

    gat_weights_path = PROCESSED_DIR / "pool_7comp_hybrid_gat_xg.pt"
    torch.save(gat_model.state_dict(), gat_weights_path)
    print(f"  Saved → {gat_weights_path}")

    # ── Step 4: calibrate HybridGAT ──────────────────────────────────────────
    print("\n[4/4] Calibrating HybridGAT…")
    scaler_gat = TemperatureScaler(gat_model, init_T=1.5)
    cal_gat    = scaler_gat.fit(val_loader, device=str(DEVICE))
    T_gat      = scaler_gat.temperature

    auc_gat_cal, brier_gat_cal, _, _ = eval_hybrid(scaler_gat, test_loader)
    print(f"  HybridGAT +T:  AUC={auc_gat_cal:.3f}  Brier={brier_gat_cal:.4f}  (T={T_gat:.4f})")
    print(f"  Brier improvement: {brier_te:.4f} → {brier_gat_cal:.4f}  "
          f"(Δ={brier_te - brier_gat_cal:+.4f})")

    scaler_gat.save(PROCESSED_DIR / "pool_7comp_gat_T.pt")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  DONE — files written:")
    for f in ["pool_7comp_T.pt", "pool_7comp_hybrid_gat_xg.pt", "pool_7comp_gat_T.pt"]:
        p = PROCESSED_DIR / f
        exists = "✅" if p.exists() else "❌"
        print(f"  {exists}  {f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
