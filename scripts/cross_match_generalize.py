"""
scripts/cross_match_generalize.py
----------------------------------
Cross-match generalization experiment.

  Train: Game 1 (799 pass graphs)  — 80/20 train/val split for model selection
  Test:  Game 2 (964 pass graphs)  — fully held-out, different game

Task: predict which team is making the pass (Home=0, Away=1) from spatial
      structure alone — team identity flag removed from node features.

Run from repo root:
    python scripts/cross_match_generalize.py
"""

import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

from src.models.gcn import FootballGCN
from src.models.gat import FootballGAT

# ── Config ─────────────────────────────────────────────────────────────────
SEED    = 42
EPOCHS  = 100
BATCH   = 32
LR      = 3e-3
WD      = 1e-4
VAL_FRAC = 0.20          # fraction of Game 1 used for val / model selection
DEVICE  = torch.device("cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

SAVE_DIR = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 65)
print("Cross-Match Generalization: train=Game1  test=Game2")
print("=" * 65)

# ── 1. Load datasets ────────────────────────────────────────────────────────
def load(game_num):
    path = os.path.join(SAVE_DIR, f"metrica_game{game_num}_pass_graphs.pt")
    graphs = torch.load(path, weights_only=False)
    labels = [int(g.y.item()) for g in graphs]
    home, away = labels.count(0), labels.count(1)
    print(f"  Game {game_num}: {len(graphs)} graphs  "
          f"Home={home} ({home/len(graphs)*100:.1f}%)  "
          f"Away={away} ({away/len(graphs)*100:.1f}%)")
    return graphs

print("\n[1/4] Loading graphs...")
game1 = load(1)
game2 = load(2)

IN_CH    = game1[0].x.shape[1]          # 8  [x,y,vx,vy,dist_atk,dist_def,angle,pressure]
EDGE_DIM = game1[0].edge_attr.shape[1]  # 6
print(f"  Node feat dim : {IN_CH}")
print(f"  Edge feat dim : {EDGE_DIM}")

# ── 2. Splits ───────────────────────────────────────────────────────────────
print("\n[2/4] Splits...")
n_val = int(VAL_FRAC * len(game1))
# Chronological: val = last 20% of Game 1
train_data = game1[:-n_val]
val_data   = game1[-n_val:]
test_data  = game2                       # entirely held-out game

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH)
test_loader  = DataLoader(test_data,  batch_size=BATCH)

print(f"  Train (Game 1, first 80%) : {len(train_data)}")
print(f"  Val   (Game 1, last  20%) : {len(val_data)}")
print(f"  Test  (Game 2, all)       : {len(test_data)}")

# ── 3. Train utilities ──────────────────────────────────────────────────────
def forward(model, batch, use_edge_attr):
    if use_edge_attr:
        return model(batch.x, batch.edge_index, batch.batch,
                     edge_attr=batch.edge_attr)
    return model(batch.x, batch.edge_index, batch.batch)


def train_epoch(model, loader, opt, use_edge_attr):
    model.train()
    total_loss = correct = total = 0
    for batch in loader:
        opt.zero_grad()
        out  = forward(model, batch, use_edge_attr)
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())
        loss.backward()
        opt.step()
        total_loss += loss.detach().item() * batch.num_graphs
        correct    += int(((out.squeeze() > 0).long() ==
                           batch.y.squeeze().long()).sum())
        total      += batch.num_graphs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, use_edge_attr):
    model.eval()
    total_loss = correct = total = 0
    probs, labels = [], []
    for batch in loader:
        out  = forward(model, batch, use_edge_attr)
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        correct    += int(((out.squeeze() > 0).long() ==
                           batch.y.squeeze().long()).sum())
        total      += batch.num_graphs
        probs.extend(torch.sigmoid(out.squeeze()).tolist())
        labels.extend(batch.y.squeeze().long().tolist())
    auc = roc_auc_score(labels, probs)
    return total_loss / total, correct / total, auc, probs, labels


def run(model, use_edge_attr, tag):
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    hist  = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": [], "va_auc": []}
    best_auc, best_state = 0.0, None

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, use_edge_attr)
        va_loss, va_acc, va_auc, _, _ = evaluate(model, val_loader, use_edge_attr)
        sched.step(va_loss)

        hist["tr_loss"].append(tr_loss);  hist["va_loss"].append(va_loss)
        hist["tr_acc"].append(tr_acc);    hist["va_acc"].append(va_acc)
        hist["va_auc"].append(va_auc)

        if va_auc > best_auc:
            best_auc   = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0:
            print(f"  [{tag}] ep {ep:3d}  "
                  f"tr loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                  f"val loss={va_loss:.4f} acc={va_acc:.3f} auc={va_auc:.3f}")

    model.load_state_dict(best_state)
    return hist


# ── 4. Train & evaluate ─────────────────────────────────────────────────────
print(f"\n[3/4] Training  (epochs={EPOCHS}  batch={BATCH}  lr={LR})")

results = {}

for name, model, use_edge_attr in [
    ("GCN", FootballGCN(IN_CH, hidden_dim=64, out_channels=1, n_layers=3, dropout=0.3), False),
    ("GAT", FootballGAT(IN_CH, edge_dim=EDGE_DIM, hidden_dim=32, out_channels=1,
                        n_layers=3, heads=4, dropout=0.3), True),
]:
    params = sum(p.numel() for p in model.parameters())
    print(f"\n── {name}  ({params:,} params) ──")
    hist = run(model, use_edge_attr, name)

    # Evaluate on val (Game 1) and test (Game 2)
    _, val_acc, val_auc, _, _           = evaluate(model, val_loader, use_edge_attr)
    te_loss, te_acc, te_auc, probs, lbls = evaluate(model, test_loader, use_edge_attr)

    print(f"  Val  (Game 1) → acc={val_acc:.3f}  AUC={val_auc:.3f}")
    print(f"  Test (Game 2) → acc={te_acc:.3f}  AUC={te_auc:.3f}  loss={te_loss:.4f}")

    results[name] = dict(
        model=model, hist=hist, use_edge_attr=use_edge_attr,
        val_acc=val_acc, val_auc=val_auc,
        te_acc=te_acc, te_auc=te_auc, te_loss=te_loss,
        probs=probs, labels=lbls,
    )

# ── 5. Summary ──────────────────────────────────────────────────────────────
print("\n[4/4] Results")
print("─" * 65)
print(f"{'Model':<6}  {'Val Acc':>8} {'Val AUC':>8}  {'Test Acc':>9} {'Test AUC':>9}")
print("─" * 65)
for name, r in results.items():
    print(f"{name:<6}  {r['val_acc']:>8.3f} {r['val_auc']:>8.3f}  "
          f"{r['te_acc']:>9.3f} {r['te_auc']:>9.3f}")
print("─" * 65)

# Classification report for best test-AUC model
best = max(results, key=lambda n: results[n]["te_auc"])
r    = results[best]
print(f"\nClassification report — {best} on Game 2 (test):")
print(classification_report(
    r["labels"], [int(p > 0.5) for p in r["probs"]],
    target_names=["Home", "Away"], digits=3
))

# ── 6. Plots ────────────────────────────────────────────────────────────────
colors = {"GCN": "#1565c0", "GAT": "#2e7d32"}
epochs = range(1, EPOCHS + 1)

# Learning curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name, r in results.items():
    c = colors[name]
    axes[0].plot(epochs, r["hist"]["tr_loss"], lw=1, linestyle="--", color=c, alpha=0.5)
    axes[0].plot(epochs, r["hist"]["va_loss"], lw=2, color=c, label=name)
    axes[1].plot(epochs, r["hist"]["va_acc"],  lw=2, color=c, label=name)
    axes[2].plot(epochs, r["hist"]["va_auc"],  lw=2, color=c, label=name)
for ax, title in zip(axes, ["Val Loss (Game 1)", "Val Acc (Game 1)", "Val AUC (Game 1)"]):
    ax.set_title(title); ax.legend(); ax.set_xlabel("Epoch")
axes[1].set_ylim(0, 1); axes[2].set_ylim(0.4, 1.0)
fig.suptitle("Training curves — Game 1 (val set)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "crossmatch_training_curves.png"), dpi=130)
print("\nSaved crossmatch_training_curves.png")

# ROC + confusion matrix on Game 2
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for name, r in results.items():
    fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
    axes[0].plot(fpr, tpr, lw=2, color=colors[name],
                 label=f"{name} AUC={r['te_auc']:.3f}")
axes[0].plot([0,1],[0,1],"k--",lw=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC — Game 2 (test)"); axes[0].legend()

for ax, (name, r) in zip(axes[1:], results.items()):
    cm = confusion_matrix(r["labels"], [int(p > 0.5) for p in r["probs"]])
    ConfusionMatrixDisplay(cm, display_labels=["Home","Away"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} — Game 2  acc={r['te_acc']:.3f}")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "crossmatch_roc.png"), dpi=130)
print("Saved crossmatch_roc.png")

# ── 7. Save models ──────────────────────────────────────────────────────────
for name, r in results.items():
    torch.save({
        "model_state":   r["model"].state_dict(),
        "in_channels":   IN_CH,
        "edge_dim":      EDGE_DIM,
        "train_game":    1,
        "test_game":     2,
        "val_acc":       r["val_acc"],
        "val_auc":       r["val_auc"],
        "test_acc":      r["te_acc"],
        "test_auc":      r["te_auc"],
    }, os.path.join(SAVE_DIR, f"{name.lower()}_crossmatch.pt"))

print(f"Models saved → {SAVE_DIR}/")
print("\nDone.")
