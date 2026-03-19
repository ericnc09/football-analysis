#!/usr/bin/env python3
"""
train_statsbomb_classifier.py
------------------------------
Train GCN and GAT on StatsBomb 360 pass-completion graphs.

Task: binary classification — did the pass succeed?
  Label 0 = complete, Label 1 = incomplete/out

Usage:
    # Train on WC2022 (assumes build_statsbomb_graphs.py was already run)
    python scripts/train_statsbomb_classifier.py

    # Specify a different dataset file
    python scripts/train_statsbomb_classifier.py --data data/processed/statsbomb_wc2022_pass_graphs.pt

    # Cross-competition: train WC2022, test WWC2023
    python scripts/train_statsbomb_classifier.py \\
        --train data/processed/statsbomb_wc2022_pass_graphs.pt \\
        --test  data/processed/statsbomb_wwc2023_pass_graphs.pt
"""

import sys
import os
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report, roc_curve
)

from src.models.gcn import FootballGCN
from src.models.gat import FootballGAT

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

EPOCHS = 100
BATCH = 64
LR = 3e-3
WD = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list:
    graphs = torch.load(path, weights_only=False)
    print(f"  Loaded {len(graphs)} graphs from {path.name}")
    return graphs


def print_label_distribution(graphs: list, name: str = ""):
    labels = [int(g.y.item()) for g in graphs]
    n0 = labels.count(0)
    n1 = labels.count(1)
    total = len(labels)
    print(f"  {name}  complete={n0} ({100*n0/total:.1f}%)  "
          f"incomplete={n1} ({100*n1/total:.1f}%)")
    return n0, n1


def chronological_split(graphs: list, train_frac=0.70, val_frac=0.15):
    """Chronological 70/15/15 split (preserves ordering)."""
    n = len(graphs)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    return graphs[:t], graphs[t:v], graphs[v:]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _forward(model, batch):
    """Call model with the right arguments (GCN ignores edge_attr, GAT uses it)."""
    import inspect
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())
    if "edge_attr" in params:
        # GAT: forward(x, edge_index, batch, edge_attr=None)
        return model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
    # GCN: forward(x, edge_index, batch)
    return model(batch.x, batch.edge_index, batch.batch)


def train_epoch(model, loader, optimizer, pos_weight):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = _forward(model, batch)
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(), batch.y.squeeze(), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = _forward(model, batch)
        all_logits.append(logits.squeeze().cpu())
        all_labels.append(batch.y.squeeze().cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    y = labels.numpy().astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    f1 = f1_score(y, preds, average="macro", zero_division=0)
    return acc, auc, f1, probs, y


def run_experiment(
    name: str,
    train_graphs: list,
    val_graphs: list,
    test_graphs: list,
    in_channels: int,
    edge_dim: int,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class-weighted loss for imbalanced labels
    y_train = torch.tensor([g.y.item() for g in train_graphs])
    n0 = (y_train == 0).sum().item()
    n1 = (y_train == 1).sum().item()
    pos_weight = torch.tensor([n0 / max(n1, 1)], dtype=torch.float, device=DEVICE)
    print(f"\n  pos_weight (class balance) = {pos_weight.item():.2f}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH)
    test_loader = DataLoader(test_graphs, batch_size=BATCH)

    results = {}

    for ModelClass, model_name, kwargs in [
        (FootballGCN, "GCN", {"in_channels": in_channels, "hidden_dim": 64, "out_channels": 1, "n_layers": 3, "dropout": 0.3}),
        (FootballGAT, "GAT", {"in_channels": in_channels, "edge_dim": edge_dim, "hidden_dim": 32, "out_channels": 1, "n_layers": 3, "heads": 4, "dropout": 0.3}),
    ]:
        print(f"\n  ── Training {model_name} ──────────────────────────")
        model = ModelClass(**kwargs).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"     Parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-5
        )

        train_losses, val_aucs = [], []
        best_val_auc = 0.0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, pos_weight)
            val_acc, val_auc, val_f1, _, _ = evaluate(model, val_loader)
            scheduler.step(1 - val_auc)

            train_losses.append(loss)
            val_aucs.append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 20 == 0:
                print(f"     Epoch {epoch:3d}  loss={loss:.4f}  val_acc={val_acc:.3f}  val_auc={val_auc:.3f}")

        # Restore best model and evaluate on test set
        model.load_state_dict(best_state)
        test_acc, test_auc, test_f1, test_probs, test_y = evaluate(model, test_loader)
        print(f"\n     Test  acc={test_acc:.3f}  auc={test_auc:.3f}  f1={test_f1:.3f}")
        print(classification_report(test_y, (test_probs >= 0.5).astype(int),
                                     target_names=["complete", "incomplete"], zero_division=0))

        results[model_name] = {
            "acc": test_acc, "auc": test_auc, "f1": test_f1,
            "probs": test_probs, "labels": test_y,
            "train_losses": train_losses, "val_aucs": val_aucs,
        }

        # Save model
        save_path = out_dir / f"{name}_{model_name.lower()}.pt"
        torch.save(model.state_dict(), save_path)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model_name, res in results.items():
        axes[0].plot(res["train_losses"], label=model_name)
        axes[1].plot(res["val_aucs"], label=model_name)
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="BCE Loss")
    axes[1].set(title="Validation AUC", xlabel="Epoch", ylabel="AUC-ROC")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle(name)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_training_curves.png", dpi=100)
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(title=f"ROC Curve — {name}", xlabel="FPR", ylabel="TPR")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_roc.png", dpi=100)
    plt.close()

    print(f"\n  Plots saved to {out_dir}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None,
                        help="Single dataset file for in-distribution experiment")
    parser.add_argument("--train", type=Path, default=None,
                        help="Training dataset (cross-competition mode)")
    parser.add_argument("--test", type=Path, default=None,
                        help="Test dataset (cross-competition mode)")
    args = parser.parse_args()

    print("=" * 70)
    print("StatsBomb 360 — Pass Completion Classifier")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    if args.train and args.test:
        # --- Cross-competition experiment ---
        print("\n[Mode] Cross-competition generalization")
        train_all = load_dataset(args.train)
        test_graphs = load_dataset(args.test)

        print("\nLabel distributions:")
        print_label_distribution(train_all, f"Train ({args.train.stem})")
        print_label_distribution(test_graphs, f"Test  ({args.test.stem})")

        # Split train into train+val
        t = int(len(train_all) * 0.85)
        train_graphs = train_all[:t]
        val_graphs = train_all[t:]
        print(f"\n  Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

        in_channels = train_graphs[0].x.shape[1]
        edge_dim = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr is not None else 0
        print(f"  Node features: {in_channels}  Edge features: {edge_dim}")

        name = f"statsbomb_cross_{args.train.stem[:12]}_{args.test.stem[:12]}"
        run_experiment(name, train_graphs, val_graphs, test_graphs,
                       in_channels, edge_dim, PROCESSED_DIR)

    else:
        # --- Single-dataset experiment ---
        data_path = args.data
        if data_path is None:
            # Auto-detect: prefer WC2022, fall back to first match found
            candidates = sorted(PROCESSED_DIR.glob("statsbomb_*_pass_graphs.pt"))
            if not candidates:
                print("ERROR: No StatsBomb dataset found. Run build_statsbomb_graphs.py first.")
                sys.exit(1)
            data_path = candidates[0]
            print(f"[Auto-detected dataset] {data_path.name}")

        all_graphs = load_dataset(data_path)

        print("\nLabel distributions:")
        print_label_distribution(all_graphs, "Full dataset")

        train_graphs, val_graphs, test_graphs = chronological_split(all_graphs)
        print(f"\n  Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

        in_channels = train_graphs[0].x.shape[1]
        edge_dim = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr is not None else 0
        print(f"  Node features: {in_channels}  Edge features: {edge_dim}")

        stem = data_path.stem.replace("_pass_graphs", "")
        run_experiment(stem, train_graphs, val_graphs, test_graphs,
                       in_channels, edge_dim, PROCESSED_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
