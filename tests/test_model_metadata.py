"""
test_model_metadata.py
----------------------
Unit tests for the metadata-sidecar contract (src/model_metadata.py).

These tests verify the training↔serving contract that the ML review flagged
as the #2 deploy-blocker: the reverse-engineering of `meta_dim` from tensor
shapes. The sidecar replaces that by encoding architecture and feature-schema
version explicitly. Each test below locks down one invariant:

1. Sidecars round-trip: save → load → dataclass equality (modulo sha256).
2. Compatibility gate rejects schema drift.
3. Compatibility gate rejects meta_dim drift.
4. Compatibility gate rejects model-class mismatch.
5. from_hybrid_xg / from_hybrid_gat read architecture from the head layer.
6. Checksum verification detects a torn file.

No real data dependencies — everything runs on synthetic tensors.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

from src.model_metadata import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    KNOWN_SCHEMAS,
    FeatureSchemaMismatch,
    ModelMetadata,
    assert_compatible,
    load_sidecar,
    save_sidecar,
    sidecar_path_for,
)
from src.models.hybrid_gat import HybridGATModel


# ─── Fixtures ───────────────────────────────────────────────────────────────

class _TinyHybridXG(nn.Module):
    """Minimal replica of app.py's HybridXGModel, used to exercise
    `ModelMetadata.from_hybrid_xg` without pulling in Streamlit."""

    def __init__(self, in_channels: int = 9, hidden_dim: int = 16, meta_dim: int = 27):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
        ])
        self.dropout = 0.3
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch, metadata):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        emb = global_mean_pool(x, batch)
        return self.head(torch.cat([emb, metadata], dim=1))


@pytest.fixture
def gcn_model():
    torch.manual_seed(0)
    return _TinyHybridXG(in_channels=9, hidden_dim=16, meta_dim=27)


@pytest.fixture
def gat_model():
    torch.manual_seed(0)
    return HybridGATModel(
        node_in=9, edge_dim=4, meta_dim=27,
        hidden=8, heads=2, n_layers=2, dropout=0.3,
    )


# ─── Tests ──────────────────────────────────────────────────────────────────

def test_sidecar_roundtrip(tmp_path: Path, gcn_model: nn.Module) -> None:
    """save → load produces an equal ModelMetadata (checksum fills in)."""
    ckpt = tmp_path / "model.pt"
    torch.save(gcn_model.state_dict(), ckpt)

    meta = ModelMetadata.from_hybrid_xg(
        gcn_model, node_in=9, trained_on="test",
        val_auc=0.75, val_brier=0.15,
    )
    assert meta.meta_dim == 27
    assert meta.hidden_dim == 16
    assert meta.feature_schema_version == CURRENT_FEATURE_SCHEMA_VERSION

    sidecar = save_sidecar(meta, ckpt)
    assert sidecar.exists()
    assert sidecar.name.endswith(".meta.json")

    loaded = load_sidecar(ckpt)
    assert loaded.meta_dim == meta.meta_dim
    assert loaded.hidden_dim == meta.hidden_dim
    assert loaded.feature_schema_version == meta.feature_schema_version
    assert loaded.checkpoint_sha256 == meta.checkpoint_sha256  # set during save


def test_assert_compatible_rejects_schema_drift(tmp_path: Path) -> None:
    """A checkpoint declaring schema v2-gk cannot load on a v3-psxg server."""
    meta = ModelMetadata(
        model_class="HybridXGModel",
        hidden_dim=16, node_in=9, edge_dim=0, meta_dim=18,
        feature_schema_version="v2-gk",
        feature_order=list(KNOWN_SCHEMAS["v2-gk"]["feature_order"]),
    )
    with pytest.raises(FeatureSchemaMismatch, match="feature_schema_version"):
        assert_compatible(meta, expected_schema="v3-psxg")


def test_assert_compatible_rejects_meta_dim_drift(gcn_model: nn.Module) -> None:
    """Even with matching schema, wrong meta_dim is a hard fail."""
    meta = ModelMetadata.from_hybrid_xg(gcn_model, node_in=9)
    meta.meta_dim = 18  # pretend someone edited the sidecar
    with pytest.raises(FeatureSchemaMismatch, match="internally inconsistent"):
        assert_compatible(meta)


def test_assert_compatible_rejects_model_class(gcn_model: nn.Module) -> None:
    """A GCN checkpoint handed to the GAT loader is rejected loudly."""
    meta = ModelMetadata.from_hybrid_xg(gcn_model, node_in=9)
    with pytest.raises(FeatureSchemaMismatch, match="model_class"):
        assert_compatible(meta, expected_model_class="HybridGATModel")


def test_from_hybrid_xg_reads_head_layer(gcn_model: nn.Module) -> None:
    """Architecture fields come straight from the head — no magic constants."""
    meta = ModelMetadata.from_hybrid_xg(gcn_model, node_in=9)
    assert meta.model_class == "HybridXGModel"
    assert meta.hidden_dim == 16
    assert meta.meta_dim == 27
    assert meta.n_layers == 2
    # feature_order length must equal meta_dim
    assert len(meta.feature_order) == meta.meta_dim


def test_from_hybrid_gat_reads_head_layer(gat_model: HybridGATModel) -> None:
    """GAT ctor must see heads + pool_dim wired correctly."""
    meta = ModelMetadata.from_hybrid_gat(gat_model, node_in=9, edge_dim=4)
    assert meta.model_class == "HybridGATModel"
    assert meta.hidden_dim == 8
    assert meta.heads == 2
    assert meta.n_layers == 2
    assert meta.meta_dim == 27
    assert meta.edge_dim == 4


def test_checksum_mismatch_detected(tmp_path: Path, gcn_model: nn.Module) -> None:
    """Corrupting the .pt after sidecar creation triggers a checksum failure."""
    ckpt = tmp_path / "model.pt"
    torch.save(gcn_model.state_dict(), ckpt)

    meta = ModelMetadata.from_hybrid_xg(gcn_model, node_in=9)
    save_sidecar(meta, ckpt)

    # Mutate the file after the sidecar was written.
    with open(ckpt, "ab") as f:
        f.write(b"\x00")

    with pytest.raises(FeatureSchemaMismatch, match="sha256 mismatch"):
        load_sidecar(ckpt, verify_checksum=True)


def test_missing_sidecar_raises(tmp_path: Path) -> None:
    """Checkpoint without a sidecar: loader must fail loudly, not guess."""
    ckpt = tmp_path / "orphan.pt"
    ckpt.write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="No metadata sidecar"):
        load_sidecar(ckpt)


def test_sidecar_path_convention(tmp_path: Path) -> None:
    """`foo.pt` → `foo.meta.json` in the same directory."""
    p = tmp_path / "model.pt"
    sidecar = sidecar_path_for(p)
    assert sidecar.parent == p.parent
    assert sidecar.name == "model.meta.json"


def test_forward_compatibility_ignores_unknown_fields(tmp_path: Path,
                                                       gcn_model: nn.Module) -> None:
    """A sidecar written by a future version (with extra fields) still loads."""
    ckpt = tmp_path / "model.pt"
    torch.save(gcn_model.state_dict(), ckpt)
    meta = ModelMetadata.from_hybrid_xg(gcn_model, node_in=9)
    save_sidecar(meta, ckpt)

    sidecar = sidecar_path_for(ckpt)
    data = json.loads(sidecar.read_text())
    data["future_field_added_in_v2"] = {"hello": "world"}
    sidecar.write_text(json.dumps(data))

    # Should not raise — ModelMetadata.from_dict filters unknown keys.
    reloaded = load_sidecar(ckpt, verify_checksum=False)
    assert reloaded.meta_dim == 27
