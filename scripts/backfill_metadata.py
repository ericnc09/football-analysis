#!/usr/bin/env python3
"""
backfill_metadata.py
--------------------
One-time utility: synthesise metadata sidecars for legacy `.pt` checkpoints
that were trained before `src/model_metadata.py` existed.

Why
---
Every model trained with the post-review `train_xg_hybrid.py` auto-writes its
sidecar. But the 4 model checkpoints already on HF Hub don't have one. Rather
than retrain, we inspect the tensor shapes of each existing `.pt` and emit a
sidecar that documents what we can infer, plus a `backfilled=true` marker so
the serving layer can treat them with appropriate suspicion.

Usage
-----
    # Default — writes sidecars for every model .pt in data/processed/
    python scripts/backfill_metadata.py

    # Single file
    python scripts/backfill_metadata.py --ckpt data/processed/pool_7comp_hybrid_gat_xg.pt

    # Dry run — print what would be written, don't write
    python scripts/backfill_metadata.py --dry-run

Safety
------
Backfilled sidecars fill every architecture field but leave training-lineage
fields (`trained_commit`, `val_auc`, etc.) as `None`. The loader should NOT
treat these as authoritative metrics — they were reverse-engineered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.model_metadata import (  # noqa: E402
    CURRENT_FEATURE_SCHEMA_VERSION,
    KNOWN_SCHEMAS,
    ModelMetadata,
    save_sidecar,
    sidecar_path_for,
)

PROCESSED = REPO_ROOT / "data" / "processed"

# Files known to be model checkpoints (not shot graphs, not temperature scalars).
# Shot graphs are PyG Data pickles and should not get a model sidecar.
# Temperature scalars are scalar-only .pt files; ignored here.
MODEL_CHECKPOINTS = [
    "pool_7comp_hybrid_xg.pt",
    "pool_7comp_hybrid_gat_xg.pt",
]


def _infer_gcn_meta(ckpt: dict) -> dict:
    """Reverse-engineer HybridXGModel architecture from its state_dict."""
    # head.0 = Linear(hidden_dim + meta_dim, hidden_dim)
    # conv.0.lin.weight = (hidden_dim, node_in)  — first GCN layer
    head_in = int(ckpt["head.0.weight"].shape[1])
    head_out = int(ckpt["head.0.weight"].shape[0])
    hidden_dim = head_out
    meta_dim = head_in - hidden_dim

    # Find the first conv layer weight — GCNConv uses "lin.weight"
    node_in = None
    for key in ckpt:
        if key.startswith("convs.0.") and key.endswith("lin.weight"):
            node_in = int(ckpt[key].shape[1])
            break
    if node_in is None:
        # Some PyG versions name it "convs.0.lin_l.weight"; try fallbacks
        for key in ckpt:
            if key.startswith("convs.0.") and "weight" in key and "bias" not in key:
                # Shape: (hidden_dim, node_in) for a typical GCNConv
                shp = ckpt[key].shape
                if len(shp) == 2 and shp[0] == hidden_dim:
                    node_in = int(shp[1])
                    break
    if node_in is None:
        raise RuntimeError("Could not infer node_in from GCN state_dict")

    # Count conv layers
    n_layers = len({k.split(".")[1] for k in ckpt if k.startswith("convs.")})

    return {
        "model_class": "HybridXGModel",
        "hidden_dim": hidden_dim,
        "node_in": node_in,
        "edge_dim": 0,
        "meta_dim": meta_dim,
        "heads": 1,
        "n_layers": n_layers,
        "dropout": 0.3,
    }


def _infer_gat_meta(ckpt: dict) -> dict:
    """Reverse-engineer HybridGATModel architecture from its state_dict."""
    # convs.0.lin_l.weight shape = (heads * hidden, node_in)
    # head.0 = Linear(pool_dim + meta_dim, pool_dim) with pool_dim == hidden
    # For n_layers>1 the final GATv2Conv has heads=1, concat=False → pool_dim = hidden
    # So head.0.out_features == hidden.
    node_in = int(ckpt["convs.0.lin_l.weight"].shape[1])
    edge_dim = (int(ckpt["convs.0.lin_edge.weight"].shape[1])
                if "convs.0.lin_edge.weight" in ckpt else 0)

    pool_dim = int(ckpt["head.0.weight"].shape[0])   # head[0] out_features
    head_in = int(ckpt["head.0.weight"].shape[1])    # pool_dim + meta_dim
    meta_dim = head_in - pool_dim
    hidden_dim = pool_dim

    # heads: layer-0 weights are shape (heads * hidden, node_in) so we solve for heads
    layer0_out = int(ckpt["convs.0.lin_l.weight"].shape[0])
    heads = layer0_out // hidden_dim if hidden_dim else 1

    n_layers = len({k.split(".")[1] for k in ckpt if k.startswith("convs.")})

    return {
        "model_class": "HybridGATModel",
        "hidden_dim": hidden_dim,
        "node_in": node_in,
        "edge_dim": edge_dim,
        "meta_dim": meta_dim,
        "heads": heads,
        "n_layers": n_layers,
        "dropout": 0.3,
    }


def infer_metadata(ckpt_path: Path) -> ModelMetadata:
    """
    Inspect a checkpoint and produce a best-effort ModelMetadata.

    We match on filename convention:
      *_hybrid_xg.pt  → HybridXGModel (GCN-backed)
      *_hybrid_gat_xg.pt → HybridGATModel
    """
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")

    if "gat" in ckpt_path.name:
        arch = _infer_gat_meta(ckpt)
    else:
        arch = _infer_gcn_meta(ckpt)

    meta_dim = arch["meta_dim"]
    # Map meta_dim to a known schema version
    schema_version = CURRENT_FEATURE_SCHEMA_VERSION
    for v, spec in KNOWN_SCHEMAS.items():
        if spec["meta_dim"] == meta_dim:
            schema_version = v
            break

    return ModelMetadata(
        model_class=arch["model_class"],
        hidden_dim=arch["hidden_dim"],
        node_in=arch["node_in"],
        edge_dim=arch["edge_dim"],
        meta_dim=meta_dim,
        heads=arch["heads"],
        n_layers=arch["n_layers"],
        dropout=arch["dropout"],
        feature_schema_version=schema_version,
        feature_order=list(KNOWN_SCHEMAS.get(
            schema_version, {"feature_order": []})["feature_order"]),
        trained_on="backfilled-unknown",
        trained_commit=None,
        trained_at=None,
        val_auc=None,
        val_brier=None,
        val_ece=None,
        temperature_global=None,
        temperature_per_competition=None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill metadata sidecars for legacy checkpoints")
    p.add_argument("--ckpt", type=Path, help="Single checkpoint to process (overrides default set)")
    p.add_argument("--dry-run", action="store_true", help="Print inferred metadata, don't write sidecars")
    p.add_argument("--force", action="store_true", help="Overwrite an existing sidecar")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    targets: list[Path] = (
        [args.ckpt] if args.ckpt else [PROCESSED / name for name in MODEL_CHECKPOINTS]
    )

    for ckpt_path in targets:
        if not ckpt_path.exists():
            print(f"  SKIP (missing): {ckpt_path}")
            continue

        sidecar = sidecar_path_for(ckpt_path)
        if sidecar.exists() and not args.force:
            print(f"  SKIP (sidecar exists, use --force to overwrite): {sidecar.name}")
            continue

        meta = infer_metadata(ckpt_path)
        print(f"\n  {ckpt_path.name}")
        print(f"    {json.dumps(meta.to_dict(), indent=4, sort_keys=True)}")

        if args.dry_run:
            continue

        save_sidecar(meta, ckpt_path)
        print(f"  wrote {sidecar.name}")

    if args.dry_run:
        print("\n[dry-run] No files written.")


if __name__ == "__main__":
    main()
