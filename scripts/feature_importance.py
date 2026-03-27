#!/usr/bin/env python3
"""
feature_importance.py
---------------------
Permutation feature importance for the 27-dim HybridGAT metadata vector.

For each feature group, the corresponding metadata dimensions are shuffled
randomly across all validation samples and the resulting AUC drop is measured.
Groups with a larger AUC drop had a bigger influence on the model's predictions.

Feature groups (matching _metadata_tensor() layout in train_xg_hybrid.py):
  [0]    shot_dist          distance to goal (m)
  [1]    shot_angle         goal width angle (rad)
  [2]    is_header          header flag
  [3]    is_open_play       open-play flag
  [4:12] technique          8-dim one-hot (Normal / Volley / Half-volley / …)
  [12]   gk_dist            distance to goalkeeper (m)
  [13]   n_def_in_cone      defenders in wide shooting cone
  [14]   gk_off_centre      GK lateral displacement (normalised)
  [15]   gk_perp_offset     GK perp. distance from shot line (m)
  [16]   n_def_direct_line  defenders in ≤3° cone
  [17]   is_right_foot      right-foot flag
  [18:27] shot_placement    9-dim one-hot PSxG goal-face zone

Outputs
-------
    data/processed/feature_importance.json   {group_name: auc_drop}
    data/processed/feature_importance.png    horizontal bar chart

Usage
-----
    python scripts/feature_importance.py            # default: val split of pooled data
    python scripts/feature_importance.py --n-reps 5 # average over 5 random permutations
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.calibration import TemperatureScaler
from src.models.hybrid_gat import HybridGATModel

PROCESSED  = REPO_ROOT / "data" / "processed"
DEVICE     = "cpu"
BATCH      = 256
SEED       = 42

# ── Feature groups ─────────────────────────────────────────────────────────
# Each entry: (display_name, list_of_meta_dim_indices)
FEATURE_GROUPS: list[tuple[str, list[int]]] = [
    ("shot_dist",          [0]),
    ("shot_angle",         [1]),
    ("is_header",          [2]),
    ("is_open_play",       [3]),
    ("technique",          list(range(4, 12))),
    ("gk_dist",            [12]),
    ("n_def_in_cone",      [13]),
    ("gk_off_centre",      [14]),
    ("gk_perp_offset",     [15]),
    ("n_def_direct_line",  [16]),
    ("is_right_foot",      [17]),
    ("shot_placement",     list(range(18, 27))),
]

FEATURE_LABELS = {
    "shot_dist":          "Shot distance",
    "shot_angle":         "Shot angle",
    "is_header":          "Header flag",
    "is_open_play":       "Open play flag",
    "technique":          "Shot technique (one-hot)",
    "gk_dist":            "GK distance",
    "n_def_in_cone":      "Defenders in cone",
    "gk_off_centre":      "GK off-centre",
    "gk_perp_offset":     "GK perp. offset",
    "n_def_direct_line":  "Defenders on shot line",
    "is_right_foot":      "Right foot flag",
    "shot_placement":     "Shot placement zone (PSxG)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_graphs() -> list:
    graphs = []
    for pt in sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt")):
        gs = torch.load(pt, weights_only=False)
        graphs.extend(gs)
    print(f"Loaded {len(graphs)} graphs from {PROCESSED}")
    return graphs


def split_val(graphs: list, val_frac: float = 0.15) -> list:
    """Return the same deterministic val split used during training."""
    rng = random.Random(SEED)
    idxs = list(range(len(graphs)))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * 0.70)
    n_val   = int(len(idxs) * 0.15)
    val_idxs = idxs[n_train: n_train + n_val]
    return [graphs[i] for i in val_idxs]


def build_meta(batch) -> torch.Tensor:
    """Build the standard 27-dim metadata tensor."""
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

    return torch.cat([base, tech, gk, new, plc], dim=1)   # [n, 27]


def collect_logits_and_meta(model: torch.nn.Module,
                             val_g: list) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Return (all_logits [N], all_meta [N,27], y_np [N])."""
    loader = DataLoader(val_g, batch_size=BATCH)
    all_logits, all_meta, all_y = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            meta  = build_meta(batch).to(DEVICE)
            edge_attr = batch.edge_attr if batch.edge_attr is not None else None
            logits = model(batch.x, batch.edge_index, batch.batch, meta,
                           edge_attr=edge_attr)
            all_logits.append(logits.squeeze().cpu())
            all_meta.append(meta.cpu())
            all_y.append(batch.y.squeeze().cpu().numpy())
    return (
        torch.cat(all_logits),
        torch.cat(all_meta),
        np.concatenate(all_y),
    )


def auc_from_logits_and_meta(inner_model: torch.nn.Module,
                              logits_original: torch.Tensor,
                              meta_all: torch.Tensor,
                              y_np: np.ndarray,
                              permuted_meta: torch.Tensor,
                              T: float = 1.0) -> float:
    """Re-score the model with permuted metadata (graph path unchanged)."""
    # We need the graph embedding separately — use a hook trick via the head only
    # logits_original = head(cat(emb, meta_orig))
    # We can't easily separate emb from logits without re-running the GNN.
    # Instead, re-run only the MLP head with (emb_original, meta_permuted).
    # emb_original = logits -> head^-1 is not straightforward.
    # So we batch-infer the full model with swapped meta (slower but accurate).
    raise NotImplementedError  # handled via full re-inference below


@torch.no_grad()
def eval_with_permuted_meta(model: torch.nn.Module,
                             val_g: list,
                             group_dims: list[int],
                             rep: int = 0) -> float:
    """
    Evaluate model AUC with one feature group's dims shuffled across samples.

    The shuffle is applied to the PRE-BUILT meta tensor so all other dims
    are unchanged. We need a full forward pass because the graph path
    (GATv2Conv layers) re-uses the same x/edge_index.
    """
    # 1. Build all metadata upfront
    loader     = DataLoader(val_g, batch_size=BATCH)
    meta_list, y_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            meta_list.append(build_meta(batch))
            y_list.append(batch.y.squeeze().float())

    meta_all = torch.cat(meta_list)   # [N, 27]
    y_all    = torch.cat(y_list).numpy()
    N        = meta_all.shape[0]

    # 2. Shuffle the target dims using a fixed-seed permutation
    rng   = np.random.RandomState(SEED + rep)
    perm  = rng.permutation(N)
    meta_perm = meta_all.clone()
    meta_perm[:, group_dims] = meta_all[perm][:, group_dims]

    # 3. Full re-inference with permuted metadata, one graph at a time isn't
    #    feasible — instead we do a single batched pass storing meta externally.
    logits_list = []
    idx = 0
    for batch in DataLoader(val_g, batch_size=BATCH):
        batch      = batch.to(DEVICE)
        n          = batch.num_graphs
        meta_slice = meta_perm[idx: idx + n].to(DEVICE)
        edge_attr  = batch.edge_attr if batch.edge_attr is not None else None
        logit      = model(batch.x, batch.edge_index, batch.batch, meta_slice,
                           edge_attr=edge_attr)
        logits_list.append(logit.squeeze().cpu())
        idx += n

    probs = torch.sigmoid(torch.cat(logits_list)).numpy()
    return float(roc_auc_score(y_all, probs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Permutation feature importance for HybridGAT")
    parser.add_argument("--n-reps", type=int, default=3,
                        help="Number of random permutations to average (default 3)")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    gat_path = PROCESSED / "pool_7comp_hybrid_gat_xg.pt"
    T_path   = PROCESSED / "pool_7comp_gat_T.pt"

    if not gat_path.exists():
        print(f"ERROR: model not found at {gat_path}. Run train_xg_hybrid.py first.")
        sys.exit(1)

    ckpt = torch.load(gat_path, weights_only=True, map_location=DEVICE)
    _pool_dim   = 32
    actual_meta = int(ckpt["head.0.weight"].shape[1]) - _pool_dim
    in_channels = 9
    # Auto-detect edge_dim from the first lin_edge weight (shape: [hidden*heads, edge_dim])
    edge_dim    = int(ckpt["convs.0.lin_edge.weight"].shape[1])

    inner = HybridGATModel(node_in=in_channels, edge_dim=edge_dim,
                           meta_dim=actual_meta, hidden=32, heads=4,
                           n_layers=3, dropout=0.0)   # dropout=0 for eval
    inner.load_state_dict(ckpt)
    inner.eval()

    T = 1.0
    if T_path.exists():
        T = float(torch.load(T_path, weights_only=True)["T"])
    print(f"Loaded HybridGAT (meta_dim={actual_meta}, T={T:.4f})")

    # Wrap with T-scaling
    scaler = TemperatureScaler(inner, init_T=1.0)
    with torch.no_grad():
        scaler.log_T.fill_(float(np.log(T)))
    model = scaler.model   # use inner model directly (T applied manually below)

    # ── Load val set ──────────────────────────────────────────────────────────
    all_graphs = load_graphs()
    val_g      = split_val(all_graphs)
    print(f"Val set: {len(val_g)} graphs")

    # ── Baseline AUC ─────────────────────────────────────────────────────────
    loader = DataLoader(val_g, batch_size=BATCH)
    logits_list, y_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch     = batch.to(DEVICE)
            meta      = build_meta(batch)
            edge_attr = batch.edge_attr if batch.edge_attr is not None else None
            logit     = model(batch.x, batch.edge_index, batch.batch, meta,
                              edge_attr=edge_attr)
            logits_list.append(logit.squeeze().cpu())
            y_list.append(batch.y.squeeze().float())

    probs_base = torch.sigmoid(torch.cat(logits_list) / T).numpy()
    y_np       = torch.cat(y_list).numpy()
    baseline   = float(roc_auc_score(y_np, probs_base))
    print(f"\nBaseline AUC: {baseline:.4f}")

    # ── Permutation importance ────────────────────────────────────────────────
    print(f"\nRunning permutation importance ({args.n_reps} rep(s) per feature group)…\n")
    results: dict[str, float] = {}

    for name, dims in FEATURE_GROUPS:
        auc_drops = []
        for rep in range(args.n_reps):
            auc_perm = eval_with_permuted_meta(model, val_g, dims, rep=rep)
            auc_perm_scaled = auc_perm   # T cancels out when comparing AUCs
            auc_drops.append(baseline - auc_perm_scaled)
        drop = float(np.mean(auc_drops))
        results[name] = drop
        label = FEATURE_LABELS.get(name, name)
        print(f"  {label:<35} AUC drop = {drop:+.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = PROCESSED / "feature_importance.json"
    with open(out_json, "w") as f:
        json.dump({"baseline_auc": baseline, "auc_drops": results,
                   "model": "HybridGAT+T", "meta_dim": actual_meta,
                   "n_reps": args.n_reps}, f, indent=2)
    print(f"\nSaved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
        names   = [FEATURE_LABELS.get(k, k) for k, _ in sorted_items]
        drops   = [v for _, v in sorted_items]
        colours = ["#e05c5c" if d > 0.005 else
                   "#f5a623" if d > 0.001 else "#6b9ec7"
                   for d in drops]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        bars = ax.barh(names, drops, color=colours, edgecolor="none", height=0.6)
        ax.axvline(0, color="white", lw=0.8, alpha=0.4)
        ax.set_xlabel("AUC drop when feature group is permuted", color="white", fontsize=11)
        ax.set_title(f"Feature Importance — HybridGAT (baseline AUC {baseline:.3f})",
                     color="white", fontsize=13, pad=12)
        ax.tick_params(colors="white")
        ax.spines[["top", "right", "left", "bottom"]].set_color("#333")
        for bar, d in zip(bars, drops):
            ax.text(max(d + 0.0003, 0.0003), bar.get_y() + bar.get_height() / 2,
                    f"{d:+.4f}", va="center", ha="left", color="white", fontsize=9)
        plt.tight_layout()

        out_png = PROCESSED / "feature_importance.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight",
                    facecolor="#0e1117", edgecolor="none")
        print(f"Plot  → {out_png}")
        plt.close(fig)
    except Exception as e:
        print(f"Warning: plot failed ({e})")


if __name__ == "__main__":
    main()
