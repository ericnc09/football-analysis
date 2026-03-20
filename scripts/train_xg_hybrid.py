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
META_DIM = 12  # shot_dist, shot_angle, is_header, is_open_play + technique (8-dim one-hot)


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

    def forward(self, x, edge_index, batch, metadata):
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
    """Stack per-graph metadata features into [n_graphs, META_DIM=12]."""
    base = torch.stack([
        batch.shot_dist.squeeze(),
        batch.shot_angle.squeeze(),
        batch.is_header.squeeze().float(),
        batch.is_open_play.squeeze().float(),
    ], dim=1)                          # [n, 4]
    tech = batch.technique.view(-1, 8) # [n, 8]
    return torch.cat([base, tech], dim=1).to(DEVICE)  # [n, 12]


def train_epoch_hybrid(model, loader, optimizer, pos_weight):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        meta  = _metadata_tensor(batch)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch, meta)
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
        logits_all.append(model(batch.x, batch.edge_index, batch.batch, meta).squeeze().cpu())
        y_all.append(batch.y.squeeze().cpu())
    logits = torch.cat(logits_all)
    y      = torch.cat(y_all).numpy().astype(int)
    probs  = torch.sigmoid(logits).numpy()
    auc    = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    return auc, probs, y


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

    # ── Hybrid model ──────────────────────────────────────────────────────
    print("\n── HybridGCN (GCN embedding + shot metadata) ────────────────────")
    hybrid, hybrid_aucs = train_hybrid(train_g, val_g, in_ch, pos_weight)
    _, hybrid_probs, _ = eval_hybrid(hybrid, DataLoader(test_g, batch_size=BATCH))

    # Save models
    out = PROCESSED_DIR
    torch.save(gcn.state_dict(),    out / f"{name}_gcn_xg.pt")
    torch.save(gat.state_dict(),    out / f"{name}_gat_xg.pt")
    torch.save(hybrid.state_dict(), out / f"pool_7comp_hybrid_xg.pt")  # canonical path used by app.py

    # ── Benchmark table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  BENCHMARK — {name}  (test n={len(y_test)}, goals={int(y_test.sum())})")
    print(f"{'='*72}")
    print(f"  {'Model':<50}  AUC    AP     Brier  Acc")
    print(f"  {'-'*70}")

    results = {}
    results["Majority"]    = print_row("Majority (no goal)", y_test, majority)
    results["LogReg"]      = print_row("LogReg (dist + angle + header)", y_test, lr_probs)
    results["StatsBomb xG"]= print_row("StatsBomb xG  [industry reference]", y_test, sb_xg)
    results["GCN"]         = print_row("GCN  [graph spatial only]", y_test, gcn_probs)
    results["GAT"]         = print_row("GAT  [graph + edge attention]", y_test, gat_probs)
    results["HybridGCN"]   = print_row("HybridGCN  [GCN + dist/angle/header/technique]", y_test, hybrid_probs)
    print(f"  {'-'*70}")

    sb_auc     = results["StatsBomb xG"]["auc"]
    hyb_auc    = results["HybridGCN"]["auc"]
    gcn_auc    = results["GCN"]["auc"]
    lr_auc     = results["LogReg"]["auc"]
    print(f"\n  StatsBomb xG  : {sb_auc:.3f}")
    print(f"  HybridGCN     : {hyb_auc:.3f}  (gap to SB: {sb_auc - hyb_auc:+.3f})")
    print(f"  GCN           : {gcn_auc:.3f}  (hybrid gain: {hyb_auc - gcn_auc:+.3f})")
    print(f"  LogReg        : {lr_auc:.3f}  (hybrid gain: {hyb_auc - lr_auc:+.3f})")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ROC curves
    ax = axes[0]
    colors = {"StatsBomb xG": "gold", "LogReg": "steelblue",
              "GCN": "mediumpurple", "GAT": "coral", "HybridGCN": "limegreen"}
    for label, probs in [
        ("StatsBomb xG", sb_xg),
        ("LogReg",       lr_probs),
        ("GCN",          gcn_probs),
        ("GAT",          gat_probs),
        ("HybridGCN",    hybrid_probs),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        lw  = 2.5 if label in ("StatsBomb xG", "HybridGCN") else 1.5
        ls  = "--" if label == "StatsBomb xG" else "-"
        ax.plot(fpr, tpr, lw=lw, ls=ls, color=colors.get(label),
                label=f"{label} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Val AUC curves
    ax = axes[1]
    for label, aucs in [("GCN", gcn_aucs), ("GAT", gat_aucs), ("HybridGCN", hybrid_aucs)]:
        lw = 2.0 if label == "HybridGCN" else 1.2
        ax.plot(aucs, lw=lw, label=label, color=colors.get(label))
    ax.axhline(sb_auc, color="gold", ls="--", lw=1.5,
               label=f"StatsBomb xG ({sb_auc:.3f})")
    ax.axhline(lr_auc, color="steelblue", ls=":", lw=1.2,
               label=f"LogReg ({lr_auc:.3f})")
    ax.set(title="Val AUC vs Epochs", xlabel="Epoch", ylabel="AUC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # AUC bar chart comparison
    ax = axes[2]
    model_names = ["Majority", "LogReg", "GCN", "GAT", "HybridGCN", "StatsBomb xG"]
    aucs_bar = [results[m]["auc"] for m in model_names]
    bar_colors = ["#555", "steelblue", "mediumpurple", "coral", "limegreen", "gold"]
    bars = ax.barh(model_names, aucs_bar, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, aucs_bar):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set(title="AUC Comparison", xlabel="ROC AUC", xlim=(0.45, 0.90))
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
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
