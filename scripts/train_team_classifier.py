"""
scripts/train_team_classifier.py
---------------------------------
Train GCN and GAT on Metrica pass graphs.
Task: predict which team is making the pass (Home=0, Away=1).

Run from repo root:
    python scripts/train_team_classifier.py
"""

import os
import sys

# __file__ is always available in a real script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

from src.models.gcn import FootballGCN
from src.models.gat import FootballGAT

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED  = os.path.join(REPO_ROOT, "data", "processed", "metrica_game1_pass_graphs.pt")
EVENTS_CSV = os.path.join(REPO_ROOT, "data", "raw", "metrica", "data",
                          "Sample_Game_1", "Sample_Game_1_RawEventsData.csv")
SAVE_DIR   = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

# ── Hyperparams ────────────────────────────────────────────────────────────
EPOCHS = 80
BATCH  = 32
LR     = 3e-3
WD     = 1e-4

print("=" * 60)
print(f"Repo root : {REPO_ROOT}")
print(f"Device    : {DEVICE}")
print("=" * 60)


# ── 1. Load & relabel dataset ──────────────────────────────────────────────
print("\n[1/4] Loading dataset...")
raw_graphs = torch.load(PROCESSED, weights_only=False)

events = pd.read_csv(EVENTS_CSV)
events.columns = [c.strip().lower().replace(" ", "_").replace("[", "").replace("]", "")
                  for c in events.columns]
pass_events = events[events["type"] == "PASS"].reset_index(drop=True)

assert len(raw_graphs) == len(pass_events), "Graph/event count mismatch"

dataset = []
for g, (_, row) in zip(raw_graphs, pass_events.iterrows()):
    g = g.clone()
    g.y = torch.tensor([0.0 if row["team"] == "Home" else 1.0], dtype=torch.float)
    # Drop the team flag (col 4) so the model must infer team from spatial
    # patterns alone — without this the task is trivially solved by averaging
    # node features. Kept features: [x, y, vx, vy, dist_atk, dist_def, angle, pressure]
    g.x = torch.cat([g.x[:, :4], g.x[:, 5:]], dim=1)
    dataset.append(g)

labels = [int(g.y.item()) for g in dataset]
print(f"  Total graphs : {len(dataset)}")
print(f"  Home (0)     : {labels.count(0)}")
print(f"  Away (1)     : {labels.count(1)}")
print(f"  Node feat dim: {dataset[0].x.shape[1]}")
print(f"  Edge feat dim: {dataset[0].edge_attr.shape[1]}")


# ── 2. Split ───────────────────────────────────────────────────────────────
n    = len(dataset)
n_tr = int(0.70 * n)
n_va = int(0.15 * n)

train_data, val_data, test_data = dataset[:n_tr], dataset[n_tr:n_tr+n_va], dataset[n_tr+n_va:]
train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH)
test_loader  = DataLoader(test_data,  batch_size=BATCH)
print(f"\n[2/4] Split — train:{len(train_data)}  val:{len(val_data)}  test:{len(test_data)}")


# ── 3. Training utilities ──────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, use_edge_attr):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch,
                    **{"edge_attr": batch.edge_attr} if use_edge_attr else {})
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * batch.num_graphs
        correct    += int(((out.squeeze() > 0).long() == batch.y.squeeze().long()).sum())
        total      += batch.num_graphs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, use_edge_attr):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch,
                    **{"edge_attr": batch.edge_attr} if use_edge_attr else {})
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())
        total_loss += loss.detach().item() * batch.num_graphs
        correct    += int(((out.squeeze() > 0).long() == batch.y.squeeze().long()).sum())
        total      += batch.num_graphs
        all_probs.extend(torch.sigmoid(out.squeeze()).tolist())
        all_labels.extend(batch.y.squeeze().long().tolist())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc, all_probs, all_labels


def run(model, use_edge_attr, tag):
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
    best_auc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc              = train_epoch(model, train_loader, opt, use_edge_attr)
        va_loss, va_acc, va_auc, _, _ = evaluate(model, val_loader, use_edge_attr)
        sched.step(va_loss)
        for k, v in zip(history, [tr_loss, va_loss, tr_acc, va_acc, va_auc]):
            history[k].append(v)
        if va_auc > best_auc:
            best_auc   = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  [{tag}] ep {epoch:3d}  tr loss={tr_loss:.4f} acc={tr_acc:.3f}"
                  f"  val loss={va_loss:.4f} acc={va_acc:.3f} auc={va_auc:.3f}")

    model.load_state_dict(best_state)
    return history


# ── 4. Train ───────────────────────────────────────────────────────────────
IN_CH    = dataset[0].x.shape[1]
EDGE_DIM = dataset[0].edge_attr.shape[1]

print(f"\n[3/4] Training  (epochs={EPOCHS}, batch={BATCH}, lr={LR})")
print(f"  Node features={IN_CH}  Edge features={EDGE_DIM}")

# GCN
print("\n── GCN ──")
gcn = FootballGCN(in_channels=IN_CH, hidden_dim=64, out_channels=1, n_layers=3, dropout=0.3)
gcn_hist = run(gcn, use_edge_attr=False, tag="GCN")
gcn_loss, gcn_acc, gcn_auc, gcn_probs, gcn_labels = evaluate(gcn, test_loader, False)
print(f"  GCN  test → loss={gcn_loss:.4f}  acc={gcn_acc:.3f}  AUC={gcn_auc:.3f}")

# GAT
print("\n── GAT ──")
gat = FootballGAT(in_channels=IN_CH, edge_dim=EDGE_DIM, hidden_dim=32,
                  out_channels=1, n_layers=3, heads=4, dropout=0.3)
gat_hist = run(gat, use_edge_attr=True, tag="GAT")
gat_loss, gat_acc, gat_auc, gat_probs, gat_labels = evaluate(gat, test_loader, True)
print(f"  GAT  test → loss={gat_loss:.4f}  acc={gat_acc:.3f}  AUC={gat_auc:.3f}")


# ── 5. Results ─────────────────────────────────────────────────────────────
print("\n[4/4] Results")
print("-" * 50)
print(f"{'Model':<6} {'Acc':>8} {'AUC':>8} {'Loss':>8} {'Params':>10}")
print("-" * 50)
for name, model, acc, auc, loss in [
    ("GCN", gcn, gcn_acc, gcn_auc, gcn_loss),
    ("GAT", gat, gat_acc, gat_auc, gat_loss),
]:
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:<6} {acc:>8.3f} {auc:>8.3f} {loss:>8.4f} {params:>10,}")
print("-" * 50)
winner = "GCN" if gcn_auc >= gat_auc else "GAT"
print(f"Best AUC: {winner}")


# ── 6. Plots ───────────────────────────────────────────────────────────────
FIG_DIR = os.path.join(REPO_ROOT, "data", "processed")

# Learning curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
epochs = range(1, EPOCHS + 1)
for hist, name, color in [(gcn_hist, "GCN", "#1565c0"), (gat_hist, "GAT", "#2e7d32")]:
    axes[0].plot(epochs, hist["train_loss"], lw=1.5, linestyle="--", color=color, alpha=0.6)
    axes[0].plot(epochs, hist["val_loss"],   lw=2,   label=name, color=color)
    axes[1].plot(epochs, hist["val_acc"],    lw=2,   label=name, color=color)
    axes[2].plot(epochs, hist["val_auc"],    lw=2,   label=name, color=color)
axes[0].set_title("Val Loss");  axes[0].legend(); axes[0].set_xlabel("Epoch")
axes[1].set_title("Val Acc");   axes[1].legend(); axes[1].set_ylim(0, 1)
axes[2].set_title("Val AUC");   axes[2].legend(); axes[2].set_ylim(0.4, 1.0)
fig.suptitle("GCN vs GAT — Team Classifier", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=130)
print(f"\nSaved training_curves.png")

# ROC curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for probs, labels, name, color in [
    (gcn_probs, gcn_labels, "GCN", "#1565c0"),
    (gat_probs, gat_labels, "GAT", "#2e7d32"),
]:
    fpr, tpr, _ = roc_curve(labels, probs)
    axes[0].plot(fpr, tpr, lw=2, label=f"{name} AUC={roc_auc_score(labels,probs):.3f}", color=color)
axes[0].plot([0,1],[0,1],"k--",lw=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC — Test Set")
axes[0].legend()

# Confusion matrices side by side
for ax, probs, labels, name in [
    (axes[1], gcn_probs, gcn_labels, "GCN"),
]:
    cm = confusion_matrix(labels, [int(p > 0.5) for p in probs])
    ConfusionMatrixDisplay(cm, display_labels=["Home","Away"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=130)
print("Saved roc_curves.png")

# ── 7. Save models ─────────────────────────────────────────────────────────
torch.save({"model_state": gcn.state_dict(), "in_channels": IN_CH,
            "hidden_dim": 64, "out_channels": 1,
            "test_acc": gcn_acc, "test_auc": gcn_auc},
           os.path.join(SAVE_DIR, "gcn_team_classifier.pt"))

torch.save({"model_state": gat.state_dict(), "in_channels": IN_CH,
            "edge_dim": EDGE_DIM, "hidden_dim": 32, "out_channels": 1,
            "test_acc": gat_acc, "test_auc": gat_auc},
           os.path.join(SAVE_DIR, "gat_team_classifier.pt"))

print(f"Models saved to {SAVE_DIR}/")
print("\nDone.")
