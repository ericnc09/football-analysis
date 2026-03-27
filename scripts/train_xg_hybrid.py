#!/usr/bin/env python3
"""
train_xg_hybrid.py
-------------------
Extended xG benchmark with a Hybrid model that concatenates
a GCN graph embedding with shot metadata (distance, angle, body part).

Models compared:
  1. Majority classifier
  2. LogReg (distance + angle + header + open_play)
  3. StatsBomb xG  [industry reference]
  4. GCN           [graph spatial only]
  5. GAT           [graph + edge attention]
  6. HybridGCN     [GCN embedding  +  shot metadata → MLP head]

Usage:
    # Pool all available competitions (auto-detected from data/processed/)
    python scripts/train_xg_hybrid.py

    # Specify which datasets to pool
    python scripts/train_xg_hybrid.py --data \\
        data/processed/statsbomb_wc2022_shot_graphs.pt \\
        data/processed/statsbomb_euro2020_shot_graphs.pt

    # Cross-competition: explicit train / test split
    python scripts/train_xg_hybrid.py \\
        --train data/processed/statsbomb_wc2022_shot_graphs.pt \\
                data/processed/statsbomb_euro2020_shot_graphs.pt \\
        --test  data/processed/statsbomb_wwc2023_shot_graphs.pt
"""

import sys
import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, average_precision_score,
    brier_score_loss, roc_curve, classification_report,
)
from sklearn.calibration import calibration_curve

from src.models.gcn import FootballGCN
from src.models.gat import FootballGAT
from src.models.hybrid_gat import HybridGATModel
from src.calibration import TemperatureScaler

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cpu")

EPOCHS = 120
BATCH  = 64
LR     = 1e-3
WD     = 1e-4
META_DIM = 27  # shot_dist, shot_angle, is_header, is_open_play + technique×8
               # + gk_dist, n_def_in_cone, gk_off_centre          (original 3)
               # + gk_perp_offset, n_def_direct_line, is_right_foot (precision 3)
               # + shot_placement×9 (PSxG goal-face zone, one-hot)


# ---------------------------------------------------------------------------
# Hybrid model: GCN encoder  +  metadata MLP head
# ---------------------------------------------------------------------------

class HybridXGModel(nn.Module):
    """
    GCN that produces a graph-level embedding, then concatenates it with
    12 hand-crafted shot features before a small MLP head.

    GCN path : node features → 3× GCNConv → global_mean_pool → hidden_dim
    Meta path : [dist, angle, header, open_play, technique×8]
    Head      : Linear(hidden + meta, hidden) → ReLU → Dropout → Linear(→ 1)
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64,
                 meta_dim: int = META_DIM, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_channels,  hidden_dim),
            GCNConv(hidden_dim,   hidden_dim),
            GCNConv(hidden_dim,   hidden_dim),
        ])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index, batch):
        """Return graph-level embedding before the MLP head."""
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)),
                          p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)          # [n_graphs, hidden_dim]

    def forward(self, x, edge_index, batch, metadata, edge_attr=None):
        emb = self.encode(x, edge_index, batch)    # [n_graphs, hidden_dim]
        combined = torch.cat([emb, metadata], dim=1)
        return self.head(combined)                 # [n_graphs, 1]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_graphs(paths) -> list:
    all_graphs = []
    for p in paths:
        p = Path(p)
        gs = torch.load(p, weights_only=False)
        print(f"  {p.name}: {len(gs)} graphs")
        all_graphs.extend(gs)
    return all_graphs


def stratified_split(graphs: list, train_frac=0.70, val_frac=0.15, seed=SEED):
    """Stratified random split preserving goal rate in each split."""
    rng = random.Random(seed)
    goals     = [g for g in graphs if g.y.item() == 1]
    no_goals  = [g for g in graphs if g.y.item() == 0]
    rng.shuffle(goals)
    rng.shuffle(no_goals)

    def split_list(lst):
        t = int(len(lst) * train_frac)
        v = int(len(lst) * (train_frac + val_frac))
        return lst[:t], lst[t:v], lst[v:]

    g_tr, g_va, g_te = split_list(goals)
    n_tr, n_va, n_te = split_list(no_goals)
    train = g_tr + n_tr;  rng.shuffle(train)
    val   = g_va + n_va;  rng.shuffle(val)
    test  = g_te + n_te;  rng.shuffle(test)
    return train, val, test


def label_dist(graphs, name=""):
    n1 = sum(int(g.y.item()) for g in graphs)
    n0 = len(graphs) - n1
    print(f"  {name:40s}  n={len(graphs):5d}  goals={n1} ({100*n1/len(graphs):.1f}%)")
    return n0, n1


def get_metadata(graphs) -> np.ndarray:
    rows = []
    for g in graphs:
        base = [
            g.shot_dist.item(),
            g.shot_angle.item(),
            g.is_header.item(),
            g.is_open_play.item(),
        ]
        tech = g.technique.tolist()  # 8-dim one-hot
        rows.append(base + tech)
    return np.array(rows, dtype=np.float32)


def get_sb_xg(graphs) -> np.ndarray:
    return np.array([g.sb_xg.item() for g in graphs], dtype=np.float32)


def get_labels(graphs) -> np.ndarray:
    return np.array([g.y.item() for g in graphs], dtype=np.float32)


def print_row(name, y_true, y_prob, width=48):
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    bs  = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    print(f"  {name:<{width}}  AUC={auc:.3f}  AP={ap:.3f}  Brier={bs:.3f}  Acc={acc:.3f}")
    return {"auc": auc, "ap": ap, "brier": bs, "acc": acc, "probs": y_prob}


# ---------------------------------------------------------------------------
# Logistic regression baseline
# ---------------------------------------------------------------------------

def run_logreg(train_graphs, val_graphs, test_graphs):
    X_tr = get_metadata(train_graphs + val_graphs)
    y_tr = get_labels(train_graphs + val_graphs)
    X_te = get_metadata(test_graphs)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr.fit(X_tr, y_tr)
    return lr.predict_proba(X_te)[:, 1]


# ---------------------------------------------------------------------------
# Standard GNN training (existing GCN / GAT)
# ---------------------------------------------------------------------------

def _forward_gnn(model, batch):
    import inspect
    params = list(inspect.signature(model.forward).parameters.keys())
    if "edge_attr" in params:
        return model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
    return model(batch.x, batch.edge_index, batch.batch)


def train_epoch_gnn(model, loader, optimizer, pos_weight):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = _forward_gnn(model, batch)
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(), batch.y.squeeze().float(), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        total += loss.detach().item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_gnn(model, loader):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits_all.append(_forward_gnn(model, batch).squeeze().cpu())
        y_all.append(batch.y.squeeze().cpu())
    logits = torch.cat(logits_all)
    y      = torch.cat(y_all).numpy().astype(int)
    probs  = torch.sigmoid(logits).numpy()
    auc    = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    return auc, probs, y


def train_standard_gnn(ModelClass, kwargs, train_g, val_g, pos_weight, label):
    model = ModelClass(**kwargs).to(DEVICE)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     {label}: {n:,} params")
    opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=12, factor=0.5, min_lr=1e-5)
    tr_loader = DataLoader(train_g, batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(val_g,   batch_size=BATCH)
    best_auc, best_state, val_aucs = 0.0, None, []
    for ep in range(1, EPOCHS + 1):
        loss = train_epoch_gnn(model, tr_loader, opt, pos_weight)
        auc, _, _ = eval_gnn(model, va_loader)
        sched.step(1 - auc)
        val_aucs.append(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"     ep={ep:3d}  loss={loss:.4f}  val_auc={auc:.3f}")
    model.load_state_dict(best_state)
    return model, val_aucs


# ---------------------------------------------------------------------------
# Hybrid model training
# ---------------------------------------------------------------------------

def _metadata_tensor(batch) -> torch.Tensor:
    """Stack per-graph metadata features into [n_graphs, META_DIM=18].

    Layout
    ------
    [0]    shot_dist          metres, ~0-50
    [1]    shot_angle         radians, 0-π/2
    [2]    is_header          0/1
    [3]    is_open_play       0/1
    [4:12] technique          8-dim one-hot
    [12]   gk_dist            metres, shooter→GK Euclidean distance
    [13]   n_def_in_cone      defenders in wide shooting-cone triangle (goal posts)
    [14]   gk_off_centre      GK lateral displacement / half-goal-width
    [15]   gk_perp_offset     GK perpendicular distance (m) from shooter→goal line
    [16]   n_def_direct_line  defenders in ≤3° cone, directly in shot path
    [17]   is_right_foot      1=right foot, 0=left/header (weak-foot proxy)
    [18:27] shot_placement    9-dim one-hot: goal-face zone (PSxG feature)
    """
    base = torch.stack([
        batch.shot_dist.squeeze(),
        batch.shot_angle.squeeze(),
        batch.is_header.squeeze().float(),
        batch.is_open_play.squeeze().float(),
    ], dim=1)                              # [n, 4]
    tech = batch.technique.view(-1, 8)    # [n, 8]
    gk   = torch.stack([
        batch.gk_dist.squeeze(),
        batch.n_def_in_cone.squeeze(),
        batch.gk_off_centre.squeeze(),
    ], dim=1)                              # [n, 3]
    new  = torch.stack([
        batch.gk_perp_offset.squeeze(),
        batch.n_def_direct_line.squeeze(),
        batch.is_right_foot.squeeze(),
    ], dim=1)                              # [n, 3]
    plc  = batch.shot_placement.view(-1, 9)  # [n, 9] one-hot goal-face zone
    return torch.cat([base, tech, gk, new, plc], dim=1).to(DEVICE)  # [n, 27]


def train_epoch_hybrid(model, loader, optimizer, pos_weight):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        meta  = _metadata_tensor(batch)
        optimizer.zero_grad()
        # Pass edge_attr so HybridGATModel can use edge features in its attention layers.
        # HybridXGModel (GCN) ignores the kwarg safely.
        edge_attr = batch.edge_attr if batch.edge_attr is not None else None
        logits = model(batch.x, batch.edge_index, batch.batch, meta,
                       edge_attr=edge_attr)
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(), batch.y.squeeze().float(), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        total += loss.detach().item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_hybrid(model, loader):
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        meta  = _metadata_tensor(batch)
        edge_attr = batch.edge_attr if batch.edge_attr is not None else None
        logits_all.append(model(batch.x, batch.edge_index, batch.batch, meta,
                                edge_attr=edge_attr).squeeze().cpu())
        y_all.append(batch.y.squeeze().cpu())
    logits = torch.cat(logits_all)
    y      = torch.cat(y_all).numpy().astype(int)
    probs  = torch.sigmoid(logits).numpy()
    auc    = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    return auc, probs, y


def _fit_per_comp_T(model, val_g: list, label: str) -> dict:
    """Fit one temperature scaler per competition on the validation set.

    Groups val graphs by their comp_label attribute, fits a separate T for each
    group (skipping groups with fewer than 20 shots), and returns a dict mapping
    comp_label → optimal T.

    This reduces systematic bias between competitions — e.g. men's WC shots have
    a different spatial distribution than women's WEURO, so a single global T
    inevitably over- or under-calibrates one group.
    """
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for g in val_g:
        cl = getattr(g, "comp_label", "") or "unknown"
        groups[cl].append(g)

    print(f"\n── Per-Competition T ({label}) ───────────────────────────────────")
    per_comp_T: dict[str, float] = {}
    for cl, graphs in sorted(groups.items()):
        if len(graphs) < 20:
            print(f"    {cl:25s}: skipped (n={len(graphs)} < 20)")
            continue
        loader = DataLoader(graphs, batch_size=BATCH)
        s = TemperatureScaler(model, init_T=1.5)
        s.fit(loader, device=str(DEVICE))
        per_comp_T[cl] = s.temperature
        print(f"    {cl:25s}: T={s.temperature:.4f}  (n={len(graphs)})")
    return per_comp_T


def train_hybrid(train_g, val_g, in_channels, pos_weight):
    model = HybridXGModel(in_channels=in_channels, hidden_dim=64,
                          meta_dim=META_DIM, dropout=0.3).to(DEVICE)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     HybridGCN: {n:,} params")
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=12, factor=0.5, min_lr=1e-5)
    tr_loader = DataLoader(train_g, batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(val_g,   batch_size=BATCH)
    best_auc, best_state, val_aucs = 0.0, None, []
    for ep in range(1, EPOCHS + 1):
        loss = train_epoch_hybrid(model, tr_loader, opt, pos_weight)
        auc, _, _ = eval_hybrid(model, va_loader)
        sched.step(1 - auc)
        val_aucs.append(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"     ep={ep:3d}  loss={loss:.4f}  val_auc={auc:.3f}")
    model.load_state_dict(best_state)
    return model, val_aucs


def train_hybrid_gat(train_g, val_g, in_channels, edge_channels, pos_weight):
    """Train HybridGATModel — identical loop to train_hybrid but uses GATv2Conv."""
    model = HybridGATModel(
        node_in=in_channels, edge_dim=edge_channels,
        meta_dim=META_DIM, hidden=32, heads=4, n_layers=3, dropout=0.3,
    ).to(DEVICE)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     HybridGAT: {n:,} params")
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=12, factor=0.5, min_lr=1e-5)
    tr_loader = DataLoader(train_g, batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(val_g,   batch_size=BATCH)
    best_auc, best_state, val_aucs = 0.0, None, []
    for ep in range(1, EPOCHS + 1):
        loss = train_epoch_hybrid(model, tr_loader, opt, pos_weight)
        auc, _, _ = eval_hybrid(model, va_loader)
        sched.step(1 - auc)
        val_aucs.append(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"     ep={ep:3d}  loss={loss:.4f}  val_auc={auc:.3f}")
    model.load_state_dict(best_state)
    return model, val_aucs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(name, train_g, val_g, test_g):
    in_ch   = train_g[0].x.shape[1]
    edge_ch = train_g[0].edge_attr.shape[1] if train_g[0].edge_attr is not None else 0

    n0, n1 = label_dist(train_g, "train")
    pos_weight = torch.tensor([n0 / max(n1, 1)], dtype=torch.float).to(DEVICE)
    print(f"  pos_weight = {pos_weight.item():.2f}")

    y_test  = get_labels(test_g)
    sb_xg   = get_sb_xg(test_g)

    # ── Baselines ─────────────────────────────────────────────────────────
    print("\n── Baselines ─────────────────────────────────────────────────────")
    majority = np.full_like(y_test, n1 / (n0 + n1))
    lr_probs = run_logreg(train_g, val_g, test_g)

    # ── Standard GNNs ────────────────────────────────────────────────────
    print("\n── GCN (graph spatial only) ──────────────────────────────────────")
    gcn, gcn_aucs = train_standard_gnn(
        FootballGCN,
        {"in_channels": in_ch, "hidden_dim": 64, "out_channels": 1,
         "n_layers": 3, "dropout": 0.3},
        train_g, val_g, pos_weight, "GCN"
    )
    _, gcn_probs, _ = eval_gnn(gcn, DataLoader(test_g, batch_size=BATCH))

    print("\n── GAT (graph + edge attention) ──────────────────────────────────")
    gat, gat_aucs = train_standard_gnn(
        FootballGAT,
        {"in_channels": in_ch, "edge_dim": edge_ch, "hidden_dim": 32,
         "out_channels": 1, "n_layers": 3, "heads": 4, "dropout": 0.3},
        train_g, val_g, pos_weight, "GAT"
    )
    _, gat_probs, _ = eval_gnn(gat, DataLoader(test_g, batch_size=BATCH))

    # ── HybridGCN (GCN embedding + shot metadata) ─────────────────────────
    print("\n── HybridGCN (GCN embedding + shot metadata) ────────────────────")
    hybrid, hybrid_aucs = train_hybrid(train_g, val_g, in_ch, pos_weight)
    _, hybrid_probs, _ = eval_hybrid(hybrid, DataLoader(test_g, batch_size=BATCH))

    # ── Temperature scaling for HybridGCN ─────────────────────────────────
    print("\n── Temperature Scaling (HybridGCN) ──────────────────────────────")
    va_loader_for_T = DataLoader(val_g, batch_size=BATCH)
    scaler = TemperatureScaler(hybrid, init_T=1.5)
    cal_result = scaler.fit(va_loader_for_T, device=str(DEVICE))
    # Re-evaluate on test set with calibrated probabilities
    _, hybrid_probs_raw, _ = eval_hybrid(hybrid, DataLoader(test_g, batch_size=BATCH))
    T_val = scaler.temperature
    hybrid_probs_cal = torch.sigmoid(
        torch.logit(torch.tensor(hybrid_probs_raw).clamp(1e-6, 1 - 1e-6)) / T_val
    ).numpy()

    # ── HybridGAT (GAT embedding + shot metadata) ─────────────────────────
    print("\n── HybridGAT (GAT embedding + shot metadata) ────────────────────")
    hybrid_gat, hybrid_gat_aucs = train_hybrid_gat(
        train_g, val_g, in_ch, edge_ch, pos_weight
    )
    _, hybrid_gat_probs, _ = eval_hybrid(
        hybrid_gat, DataLoader(test_g, batch_size=BATCH)
    )

    # Temperature scaling for HybridGAT
    print("\n── Temperature Scaling (HybridGAT) ──────────────────────────────")
    scaler_gat = TemperatureScaler(hybrid_gat, init_T=1.5)
    cal_gat = scaler_gat.fit(va_loader_for_T, device=str(DEVICE))
    T_gat = scaler_gat.temperature
    hybrid_gat_probs_cal = torch.sigmoid(
        torch.logit(torch.tensor(hybrid_gat_probs).clamp(1e-6, 1 - 1e-6)) / T_gat
    ).numpy()

    # Save models + global temperature scalars
    out = PROCESSED_DIR
    torch.save(gcn.state_dict(),         out / f"{name}_gcn_xg.pt")
    torch.save(gat.state_dict(),         out / f"{name}_gat_xg.pt")
    torch.save(hybrid.state_dict(),      out / "pool_7comp_hybrid_xg.pt")       # canonical GCN path
    torch.save(hybrid_gat.state_dict(),  out / "pool_7comp_hybrid_gat_xg.pt")   # canonical GAT path
    scaler.save(    out / "pool_7comp_T.pt")       # GCN global temperature
    scaler_gat.save(out / "pool_7comp_gat_T.pt")   # GAT global temperature

    # Per-competition temperature (one T per competition label)
    per_comp_T_gcn = _fit_per_comp_T(hybrid,     val_g, f"HybridGCN ({name})")
    per_comp_T_gat = _fit_per_comp_T(hybrid_gat, val_g, f"HybridGAT ({name})")
    torch.save(per_comp_T_gcn, out / "pool_7comp_per_comp_T_gcn.pt")
    torch.save(per_comp_T_gat, out / "pool_7comp_per_comp_T_gat.pt")
    print(f"  Per-comp T (GCN) → {out / 'pool_7comp_per_comp_T_gcn.pt'}")
    print(f"  Per-comp T (GAT) → {out / 'pool_7comp_per_comp_T_gat.pt'}")

    # ── Benchmark table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  BENCHMARK — {name}  (test n={len(y_test)}, goals={int(y_test.sum())})")
    print(f"{'='*72}")
    print(f"  {'Model':<50}  AUC    AP     Brier  Acc")
    print(f"  {'-'*70}")

    results = {}
    results["Majority"]        = print_row("Majority (no goal)", y_test, majority)
    results["LogReg"]          = print_row("LogReg (dist + angle + header)", y_test, lr_probs)
    results["StatsBomb xG"]    = print_row("StatsBomb xG  [industry reference]", y_test, sb_xg)
    results["GCN"]             = print_row("GCN  [graph spatial only]", y_test, gcn_probs)
    results["GAT"]             = print_row("GAT  [graph + edge attention]", y_test, gat_probs)
    results["HybridGCN"]       = print_row("HybridGCN  [GCN + metadata]", y_test, hybrid_probs)
    results["HybridGCN+T"]     = print_row(f"HybridGCN+T  [T={T_val:.3f}]", y_test, hybrid_probs_cal)
    results["HybridGAT"]       = print_row("HybridGAT  [GAT + metadata]", y_test, hybrid_gat_probs)
    results["HybridGAT+T"]     = print_row(f"HybridGAT+T  [T={T_gat:.3f}]", y_test, hybrid_gat_probs_cal)
    print(f"  {'-'*70}")

    sb_auc     = results["StatsBomb xG"]["auc"]
    hyb_auc    = results["HybridGCN"]["auc"]
    hyb_t_auc  = results["HybridGCN+T"]["auc"]
    gat_t_auc  = results["HybridGAT+T"]["auc"]
    gcn_auc    = results["GCN"]["auc"]
    lr_auc     = results["LogReg"]["auc"]
    print(f"\n  StatsBomb xG    : {sb_auc:.3f}")
    print(f"  HybridGCN       : {hyb_auc:.3f}  (gap to SB: {sb_auc - hyb_auc:+.3f})")
    print(f"  HybridGCN+T     : {hyb_t_auc:.3f}  Brier {cal_result['brier_before']:.3f} → {cal_result['brier_after']:.3f}")
    print(f"  HybridGAT+T     : {gat_t_auc:.3f}  Brier {cal_gat['brier_before']:.3f} → {cal_gat['brier_after']:.3f}")
    print(f"  GCN             : {gcn_auc:.3f}  (hybrid gain: {hyb_auc - gcn_auc:+.3f})")
    print(f"  LogReg          : {lr_auc:.3f}  (hybrid gain: {hyb_auc - lr_auc:+.3f})")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ROC curves
    ax = axes[0]
    colors = {
        "StatsBomb xG": "gold",    "LogReg":       "steelblue",
        "GCN":          "mediumpurple", "GAT":      "coral",
        "HybridGCN":    "limegreen",   "HybridGCN+T": "#00e5ff",
        "HybridGAT":    "#ff9800",     "HybridGAT+T": "#e040fb",
    }
    for label, probs in [
        ("StatsBomb xG",  sb_xg),
        ("LogReg",        lr_probs),
        ("GCN",           gcn_probs),
        ("GAT",           gat_probs),
        ("HybridGCN",     hybrid_probs),
        ("HybridGCN+T",   hybrid_probs_cal),
        ("HybridGAT+T",   hybrid_gat_probs_cal),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        lw  = 2.5 if label in ("StatsBomb xG", "HybridGCN+T", "HybridGAT+T") else 1.5
        ls  = "--" if label == "StatsBomb xG" else "-"
        ax.plot(fpr, tpr, lw=lw, ls=ls, color=colors.get(label),
                label=f"{label} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Val AUC curves
    ax = axes[1]
    for label, aucs in [
        ("GCN",       gcn_aucs),
        ("GAT",       gat_aucs),
        ("HybridGCN", hybrid_aucs),
        ("HybridGAT", hybrid_gat_aucs),
    ]:
        lw = 2.0 if "Hybrid" in label else 1.2
        ax.plot(aucs, lw=lw, label=label, color=colors.get(label))
    ax.axhline(sb_auc, color="gold", ls="--", lw=1.5,
               label=f"StatsBomb xG ({sb_auc:.3f})")
    ax.axhline(lr_auc, color="steelblue", ls=":", lw=1.2,
               label=f"LogReg ({lr_auc:.3f})")
    ax.set(title="Val AUC vs Epochs", xlabel="Epoch", ylabel="AUC")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Brier score comparison (calibration impact)
    ax = axes[2]
    brier_names = [
        "HybridGCN (raw)",
        f"HybridGCN+T (T={T_val:.2f})",
        "HybridGAT (raw)",
        f"HybridGAT+T (T={T_gat:.2f})",
        "StatsBomb xG",
    ]
    from sklearn.metrics import brier_score_loss
    brier_vals = [
        brier_score_loss(y_test, hybrid_probs),
        brier_score_loss(y_test, hybrid_probs_cal),
        brier_score_loss(y_test, hybrid_gat_probs),
        brier_score_loss(y_test, hybrid_gat_probs_cal),
        brier_score_loss(y_test, sb_xg),
    ]
    b_colors = ["limegreen", "#00e5ff", "#ff9800", "#e040fb", "gold"]
    bars = ax.barh(brier_names, brier_vals, color=b_colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, brier_vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set(title="Brier Score (lower = better calibration)",
           xlabel="Brier Score", xlim=(0.0, 0.25))
    ax.axvline(brier_score_loss(y_test, sb_xg), color="gold", ls="--", alpha=0.6)
    ax.grid(alpha=0.3, axis="x")

    plt.suptitle(f"xG Benchmark — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_fig = PROCESSED_DIR / f"{name}_hybrid_benchmark.png"
    fig.savefig(out_fig, dpi=130)
    plt.close()
    print(f"\n  Plot → {out_fig}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  nargs="+", type=Path, default=None,
                        help="One or more .pt shot graph files to pool (in-competition mode)")
    parser.add_argument("--train", nargs="+", type=Path, default=None,
                        help="Training .pt files (cross-competition mode)")
    parser.add_argument("--test",  nargs="+", type=Path, default=None,
                        help="Test .pt files (cross-competition mode)")
    args = parser.parse_args()

    print("=" * 72)
    print("xG Benchmark — GCN / GAT / HybridGCN vs StatsBomb vs LogReg")
    print("=" * 72)

    if args.train and args.test:
        # ── Cross-competition ──────────────────────────────────────────────
        print("\n[Mode] Cross-competition")
        print("Train files:")
        train_all = load_graphs(args.train)
        print("Test files:")
        test_g    = load_graphs(args.test)
        label_dist(train_all, "train+val combined")
        label_dist(test_g,    "test")
        t = int(len(train_all) * 0.85)
        train_g, val_g = train_all[:t], train_all[t:]
        print(f"\n  Train={len(train_g)}  Val={len(val_g)}  Test={len(test_g)}")
        train_stems = "+".join(p.stem[:8] for p in args.train)
        test_stems  = "+".join(p.stem[:8] for p in args.test)
        name = f"cross_{train_stems}_to_{test_stems}"[:60]
        run_experiment(name, train_g, val_g, test_g)

    else:
        # ── Pooled in-competition ──────────────────────────────────────────
        print("\n[Mode] Pooled in-competition")
        if args.data:
            paths = args.data
        else:
            paths = sorted(PROCESSED_DIR.glob("statsbomb_*_shot_graphs.pt"))
            if not paths:
                print("No shot graph files found. Run build_shot_graphs.py first.")
                sys.exit(1)
        print(f"Pooling {len(paths)} competition(s):")
        all_graphs = load_graphs(paths)

        print("\nLabel distribution:")
        label_dist(all_graphs, "full pool")
        train_g, val_g, test_g = stratified_split(all_graphs)
        print(f"\n  Train={len(train_g)}  Val={len(val_g)}  Test={len(test_g)}")
        label_dist(train_g, "train")
        label_dist(val_g,   "val")
        label_dist(test_g,  "test")

        stems = "_".join(p.stem.replace("statsbomb_","").replace("_shot_graphs","")
                         for p in paths)
        name = f"pool_{len(paths)}comp"
        run_experiment(name, train_g, val_g, test_g)

    print("\nDone.")


if __name__ == "__main__":
    main()
