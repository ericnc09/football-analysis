#!/usr/bin/env python3
"""
train_xg_model.py
------------------
Train GCN and GAT as xG (Expected Goals) models and benchmark against:
  1. Majority classifier (always predict "no goal")
  2. Logistic regression on shot distance + angle (spatial-only baseline)
  3. StatsBomb's own xG (industry reference)
  4. Our GCN (graph spatial model)
  5. Our GAT (graph spatial model with edge attention)

Task: binary classification — did the shot result in a goal?
  Label 0 = no goal, Label 1 = goal

Usage:
    # In-competition benchmark (WC2022)
    python scripts/train_xg_model.py --data data/processed/statsbomb_wc2022_shot_graphs.pt

    # Cross-competition: train WC2022, test WWC2023
    python scripts/train_xg_model.py \\
        --train data/processed/statsbomb_wc2022_shot_graphs.pt \\
        --test  data/processed/statsbomb_wwc2023_shot_graphs.pt
"""

import sys
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, roc_curve, brier_score_loss,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

from src.models.gcn import FootballGCN
from src.models.gat import FootballGAT

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

EPOCHS = 100
BATCH  = 32
LR     = 1e-3
WD     = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list:
    graphs = torch.load(path, weights_only=False)
    print(f"  Loaded {len(graphs)} shot graphs from {path.name}")
    return graphs


def split_dataset(graphs: list, train_frac=0.70, val_frac=0.15):
    n = len(graphs)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    return graphs[:t], graphs[t:v], graphs[v:]


def label_dist(graphs: list, name: str = ""):
    labels = [int(g.y.item()) for g in graphs]
    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"  {name}  goals={n1} ({100*n1/len(labels):.1f}%)  "
          f"no_goals={n0} ({100*n0/len(labels):.1f}%)")
    return n0, n1


def print_benchmark_row(name: str, y_true, y_prob, width=55):
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    bs  = brier_score_loss(y_true, y_prob)
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    print(f"  {name:<{width}} AUC={auc:.3f}  AP={ap:.3f}  Brier={bs:.3f}  Acc={acc:.3f}")
    return auc, ap, bs


# ---------------------------------------------------------------------------
# Logistic regression baseline (distance + angle + header + open_play)
# ---------------------------------------------------------------------------

def run_logreg_baseline(train_graphs, test_graphs):
    """Logistic regression on shot distance, angle, header, open_play."""
    def featurise(graphs):
        X = np.array([[
            g.shot_dist.item(),
            g.shot_angle.item(),
            g.is_header.item(),
            g.is_open_play.item(),
        ] for g in graphs], dtype=np.float32)
        y = np.array([g.y.item() for g in graphs])
        return X, y

    X_train, y_train = featurise(train_graphs)
    X_test,  y_test  = featurise(test_graphs)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_test)[:, 1]
    return probs, y_test


# ---------------------------------------------------------------------------
# GNN helpers
# ---------------------------------------------------------------------------

def _forward(model, batch):
    import inspect
    params = list(inspect.signature(model.forward).parameters.keys())
    if "edge_attr" in params:
        return model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
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
    probs  = torch.sigmoid(logits).numpy()
    y      = labels.numpy().astype(int)
    auc    = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    return auc, probs, y


def train_gnn(model_class, model_kwargs, train_graphs, val_graphs, pos_weight):
    model = model_class(**model_kwargs).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5
    )
    train_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH)

    best_val_auc = 0.0
    best_state   = None
    train_losses, val_aucs = [], []

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, pos_weight)
        val_auc, _, _ = evaluate(model, val_loader)
        scheduler.step(1 - val_auc)
        train_losses.append(loss)
        val_aucs.append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"     Epoch {epoch:3d}  loss={loss:.4f}  val_auc={val_auc:.3f}")

    model.load_state_dict(best_state)
    return model, train_losses, val_aucs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(name, train_graphs, val_graphs, test_graphs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    in_ch   = train_graphs[0].x.shape[1]
    edge_ch = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr is not None else 0

    y_train = torch.tensor([g.y.item() for g in train_graphs])
    n0 = (y_train == 0).sum().item()
    n1 = (y_train == 1).sum().item()
    pos_weight = torch.tensor([n0 / max(n1, 1)], dtype=torch.float)
    print(f"\n  pos_weight = {pos_weight.item():.2f}  (class balance)")

    # ── Collect test ground truth and StatsBomb xG ────────────────────────
    y_test   = np.array([g.y.item()    for g in test_graphs])
    sb_xg    = np.array([g.sb_xg.item() for g in test_graphs])

    # ── Baselines ──────────────────────────────────────────────────────────
    print("\n  ── Baselines ───────────────────────────────────────────")
    majority_prob = np.full_like(y_test, n1 / (n0 + n1), dtype=float)
    lr_probs, _ = run_logreg_baseline(train_graphs + val_graphs, test_graphs)

    # ── GNNs ──────────────────────────────────────────────────────────────
    gnn_results = {}
    for model_name, ModelClass, kwargs in [
        ("GCN", FootballGCN,
         {"in_channels": in_ch, "hidden_dim": 64, "out_channels": 1,
          "n_layers": 3, "dropout": 0.3}),
        ("GAT", FootballGAT,
         {"in_channels": in_ch, "edge_dim": edge_ch, "hidden_dim": 32,
          "out_channels": 1, "n_layers": 3, "heads": 4, "dropout": 0.3}),
    ]:
        print(f"\n  ── Training {model_name} ──────────────────────────────────")
        model, train_losses, val_aucs = train_gnn(
            ModelClass, kwargs, train_graphs, val_graphs, pos_weight
        )
        test_loader = DataLoader(test_graphs, batch_size=BATCH)
        _, gnn_probs, _ = evaluate(model, test_loader)
        gnn_results[model_name] = {"probs": gnn_probs, "train_losses": train_losses, "val_aucs": val_aucs}

        # Save model
        torch.save(model.state_dict(), out_dir / f"{name}_{model_name.lower()}_xg.pt")

    # ── Benchmark table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BENCHMARK — {name}  (test n={len(y_test)}, goals={y_test.sum():.0f})")
    print(f"{'='*70}")
    print(f"  {'Model':<45}  AUC    AP     Brier  Acc")
    print(f"  {'-'*65}")

    results = {}
    results["Majority (no goal)"]        = print_benchmark_row("Majority (always predict no goal)", y_test, majority_prob)
    results["LogReg (dist+angle+header)"]= print_benchmark_row("LogReg (distance + angle + header)", y_test, lr_probs)
    results["StatsBomb xG"]              = print_benchmark_row("StatsBomb xG  [industry reference]", y_test, sb_xg)
    results["GCN (GNN spatial)"]         = print_benchmark_row("GCN  [GNN, freeze-frame spatial only]", y_test, gnn_results["GCN"]["probs"])
    results["GAT (GNN + edge attn)"]     = print_benchmark_row("GAT  [GNN, freeze-frame + edge attn]", y_test, gnn_results["GAT"]["probs"])
    print(f"  {'-'*65}")

    # Best model
    best = max(["GCN (GNN spatial)", "GAT (GNN + edge attn)"],
               key=lambda k: results[k][0])
    sb_auc = results["StatsBomb xG"][0]
    best_auc = results[best][0]
    gap = sb_auc - best_auc
    print(f"\n  StatsBomb xG AUC  : {sb_auc:.3f}")
    print(f"  Best GNN AUC      : {best_auc:.3f}  ({best})")
    print(f"  Gap to industry   : {gap:+.3f}  ({'behind' if gap > 0 else 'ahead of'} StatsBomb)")

    # Classification report for best GNN
    best_probs = gnn_results[best.split()[0]]["probs"]
    print(f"\n  {best} — Classification Report (test set):")
    print(classification_report(y_test, (best_probs >= 0.5).astype(int),
                                 target_names=["no goal", "goal"], zero_division=0))

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ROC curves
    ax = axes[0]
    for label_str, probs in [
        ("StatsBomb xG",            sb_xg),
        ("LogReg (dist+angle)",     lr_probs),
        ("GCN",                     gnn_results["GCN"]["probs"]),
        ("GAT",                     gnn_results["GAT"]["probs"]),
    ]:
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            lw = 2.5 if "StatsBomb" in label_str else 1.5
            ls = "--" if "StatsBomb" in label_str else "-"
            ax.plot(fpr, tpr, lw=lw, ls=ls, label=f"{label_str} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Training curves
    ax = axes[1]
    for model_name, res in gnn_results.items():
        ax.plot(res["val_aucs"], label=f"{model_name} val AUC")
    ax.axhline(results["StatsBomb xG"][0], color="red", ls="--",
               label=f"StatsBomb xG ({results['StatsBomb xG'][0]:.3f})")
    ax.set(title="Val AUC vs Epochs", xlabel="Epoch", ylabel="AUC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Calibration curves
    ax = axes[2]
    for label_str, probs in [
        ("StatsBomb xG", sb_xg),
        ("GCN",          gnn_results["GCN"]["probs"]),
        ("GAT",          gnn_results["GAT"]["probs"]),
    ]:
        try:
            fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=8)
            lw = 2.5 if "StatsBomb" in label_str else 1.5
            ls = "--" if "StatsBomb" in label_str else "-"
            ax.plot(mean_pred, fraction_pos, lw=lw, ls=ls, marker="o", ms=4,
                    label=label_str)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set(title="Calibration Curve", xlabel="Predicted probability",
           ylabel="Fraction that are goals")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f"xG Model Benchmark — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_xg_benchmark.png", dpi=120)
    plt.close()

    print(f"\n  Plots → {out_dir / f'{name}_xg_benchmark.png'}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=Path, default=None,
                        help="Single dataset for in-competition benchmark")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test",  type=Path, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("xG Model — GNN vs StatsBomb vs Baselines")
    print("=" * 70)

    if args.train and args.test:
        print("\n[Mode] Cross-competition")
        train_all = load_dataset(args.train)
        test_graphs = load_dataset(args.test)
        label_dist(train_all,   f"Train ({args.train.stem})")
        label_dist(test_graphs, f"Test  ({args.test.stem})")
        t = int(len(train_all) * 0.85)
        train_graphs, val_graphs = train_all[:t], train_all[t:]
        print(f"\n  Train={len(train_graphs)}  Val={len(val_graphs)}  Test={len(test_graphs)}")
        name = f"xg_cross_{args.train.stem[:12]}_{args.test.stem[:12]}"
        run_experiment(name, train_graphs, val_graphs, test_graphs, PROCESSED_DIR)

    else:
        data_path = args.data
        if data_path is None:
            candidates = sorted(PROCESSED_DIR.glob("statsbomb_*_shot_graphs.pt"))
            if not candidates:
                print("ERROR: No shot dataset found. Run build_shot_graphs.py first.")
                sys.exit(1)
            data_path = candidates[0]
            print(f"[Auto-detected] {data_path.name}")

        all_graphs = load_dataset(data_path)
        print("\nLabel distributions:")
        label_dist(all_graphs, "Full dataset")
        train_graphs, val_graphs, test_graphs = split_dataset(all_graphs)
        print(f"\n  Train={len(train_graphs)}  Val={len(val_graphs)}  Test={len(test_graphs)}")
        stem = data_path.stem.replace("_shot_graphs", "")
        run_experiment(stem, train_graphs, val_graphs, test_graphs, PROCESSED_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
