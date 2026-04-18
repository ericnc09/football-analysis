"""
test_serving.py
---------------
End-to-end round-trip tests for `src/serving.py`:

    train → save → sidecar → load → predict

Covers the serving contract the ML review flagged as the highest-priority
gap (review §2 and §3): every model that comes out of the pipeline must
load cleanly back into the serving layer and produce calibrated
probabilities, with fail-loud behaviour on contract violations.

These tests are intentionally Streamlit-free. They exercise the same code
path that `app.py::load_model` / `app.py::load_gat_model` wrap — just
without the UI error boundary. If CI ever breaks here, the dashboard will
also break; the converse (dashboard breakage isolated to Streamlit UI
logic) is caught by separate smoke tests.

No real data is touched. Synthetic shot graphs are built in-memory with
the v3-psxg layout (27-dim metadata); a tiny model trains for a handful
of steps and is then snapshotted to a temp dir.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from torch_geometric.data import Data

from src.features import GraphSchemaMismatch
from src.model_metadata import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    FeatureSchemaMismatch,
    ModelMetadata,
    save_sidecar,
    sidecar_path_for,
)
from src.models import HybridGATModel, HybridXGModel
from src.serving import (
    load_gat_model,
    load_gcn_model,
    predict_batch,
    predict_one,
    schema_of,
)


# =============================================================================
# Fixtures — synthetic shot graphs in the v3-psxg schema
# =============================================================================

NODE_IN  = 9    # node feature dim — matches src/graph_builder.py
META_DIM = 27   # v3-psxg total


def _make_graph(*, label: float = 0.0, n_nodes: int = 22, seed: int = 0) -> Data:
    """Build a synthetic shot graph with every v3-psxg attribute populated.

    Shapes mirror `scripts/build_shot_graphs.py` so `src.features.build_meta`
    sees a structurally identical object to what production graphs look like.
    """
    g = torch.Generator().manual_seed(seed)

    # Node features + fully-connected edge index (small graph, so brute-force
    # dense edges are fine for a unit test).
    x = torch.randn(n_nodes, NODE_IN, generator=g)
    src = torch.arange(n_nodes).repeat_interleave(n_nodes - 1)
    dst = torch.cat([
        torch.cat([torch.arange(0, i), torch.arange(i + 1, n_nodes)])
        for i in range(n_nodes)
    ])
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr  = torch.randn(edge_index.shape[1], 1, generator=g)

    # One-hot technique (8 dim): pick a random index deterministically.
    tech = torch.zeros(8)
    tech[int(torch.randint(0, 8, (1,), generator=g).item())] = 1.0

    # One-hot placement (9 dim):
    place = torch.zeros(9)
    place[int(torch.randint(0, 9, (1,), generator=g).item())] = 1.0

    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = torch.tensor([label], dtype=torch.float),

        shot_dist    = torch.tensor([float(torch.rand(1, generator=g).item()) * 30 + 1]),
        shot_angle   = torch.tensor([float(torch.rand(1, generator=g).item()) * 1.5]),
        is_header    = torch.tensor([0.0]),
        is_open_play = torch.tensor([1.0]),
        technique    = tech,

        gk_dist            = torch.tensor([float(torch.rand(1, generator=g).item()) * 5 + 0.5]),
        n_def_in_cone      = torch.tensor([float(torch.randint(0, 4, (1,), generator=g).item())]),
        gk_off_centre      = torch.tensor([float(torch.randn(1, generator=g).item())]),
        gk_perp_offset     = torch.tensor([float(torch.randn(1, generator=g).item())]),
        n_def_direct_line  = torch.tensor([float(torch.randint(0, 3, (1,), generator=g).item())]),
        is_right_foot      = torch.tensor([1.0]),

        shot_placement     = place,
    )


@pytest.fixture
def synthetic_graphs():
    """A mixed-label set of 16 synthetic graphs — enough to train a few steps."""
    return [
        _make_graph(label=float(i % 2 == 0), seed=i)
        for i in range(16)
    ]


@pytest.fixture
def tmp_artefacts(tmp_path):
    """Return a dict of paths for the model + temperature scalar + sidecar."""
    return {
        "gcn_ckpt":  tmp_path / "test_gcn.pt",
        "gcn_temp":  tmp_path / "test_gcn_T.pt",
        "gat_ckpt":  tmp_path / "test_gat.pt",
        "gat_temp":  tmp_path / "test_gat_T.pt",
    }


# =============================================================================
# Helpers — micro-train + snapshot
# =============================================================================

def _micro_train_gcn(graphs, *, steps: int = 3) -> HybridXGModel:
    """Train a HybridXGModel for a handful of steps on synthetic data."""
    from torch_geometric.loader import DataLoader
    from src.features import build_meta

    model = HybridXGModel(
        in_channels=NODE_IN,
        hidden_dim=16,
        meta_dim=META_DIM,
        dropout=0.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(graphs, batch_size=8, shuffle=False)

    model.train()
    for _ in range(steps):
        for batch in loader:
            meta = build_meta(batch, schema_version=CURRENT_FEATURE_SCHEMA_VERSION)
            opt.zero_grad()
            logit = model(batch.x, batch.edge_index, batch.batch, meta)
            loss  = torch.nn.functional.binary_cross_entropy_with_logits(
                logit.squeeze(-1), batch.y.squeeze(-1).float()
            )
            loss.backward()
            opt.step()
    model.eval()
    return model


def _micro_train_gat(graphs, *, steps: int = 3) -> HybridGATModel:
    from torch_geometric.loader import DataLoader
    from src.features import build_meta

    model = HybridGATModel(
        node_in=NODE_IN,
        edge_dim=1,
        meta_dim=META_DIM,
        hidden=8,
        heads=2,
        n_layers=2,
        dropout=0.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(graphs, batch_size=8, shuffle=False)

    model.train()
    for _ in range(steps):
        for batch in loader:
            meta = build_meta(batch, schema_version=CURRENT_FEATURE_SCHEMA_VERSION)
            opt.zero_grad()
            logit = model(
                batch.x, batch.edge_index, batch.batch, meta,
                edge_attr=batch.edge_attr,
            )
            loss  = torch.nn.functional.binary_cross_entropy_with_logits(
                logit.squeeze(-1), batch.y.squeeze(-1).float()
            )
            loss.backward()
            opt.step()
    model.eval()
    return model


def _snapshot_gcn(model: HybridXGModel, ckpt_path: Path) -> ModelMetadata:
    """Save state_dict + sidecar for a HybridXGModel."""
    torch.save(model.state_dict(), ckpt_path)
    meta = ModelMetadata.from_hybrid_xg(model, node_in=NODE_IN, edge_dim=0)
    save_sidecar(meta, ckpt_path)
    return meta


def _snapshot_gat(model: HybridGATModel, ckpt_path: Path) -> ModelMetadata:
    torch.save(model.state_dict(), ckpt_path)
    meta = ModelMetadata.from_hybrid_gat(model, node_in=NODE_IN, edge_dim=1)
    save_sidecar(meta, ckpt_path)
    return meta


# =============================================================================
# Tests — HybridXGModel (GCN) round-trip
# =============================================================================

def test_gcn_train_save_load_predict_roundtrip(synthetic_graphs, tmp_artefacts):
    """The core contract: a freshly-trained model reloads bit-identically."""
    trained = _micro_train_gcn(synthetic_graphs)
    _snapshot_gcn(trained, tmp_artefacts["gcn_ckpt"])

    # Reload via the public serving API.
    reloaded = load_gcn_model(tmp_artefacts["gcn_ckpt"], temp_path=None)

    # 1. Wrapper carries authoritative sidecar metadata.
    assert schema_of(reloaded) == CURRENT_FEATURE_SCHEMA_VERSION
    assert reloaded._meta_dim == META_DIM

    # 2. Single-graph inference path returns a valid probability.
    prob = predict_one(reloaded, synthetic_graphs[0])
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

    # 3. Batch inference path — same length as input, same range.
    probs = predict_batch(reloaded, synthetic_graphs)
    assert probs.shape == (len(synthetic_graphs),)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    # 4. Reloaded weights agree with in-memory model to ~FP-noise. The
    #    temperature scaler wraps the base model but with T=1.0 (no temp file
    #    loaded), so outputs should be identical up to numerical round-off.
    from torch_geometric.data import Batch
    from src.features import build_meta
    batch = Batch.from_data_list(synthetic_graphs)
    meta  = build_meta(batch, schema_version=CURRENT_FEATURE_SCHEMA_VERSION)
    with torch.no_grad():
        reloaded.eval()
        trained.eval()
        new_logits = reloaded.model(batch.x, batch.edge_index, batch.batch, meta)
        old_logits = trained(batch.x, batch.edge_index, batch.batch, meta)
    assert torch.allclose(new_logits, old_logits, atol=1e-6)


def test_gcn_predict_batch_equals_predict_one(synthetic_graphs, tmp_artefacts):
    """`predict_one` and `predict_batch` must agree element-wise."""
    trained = _micro_train_gcn(synthetic_graphs)
    _snapshot_gcn(trained, tmp_artefacts["gcn_ckpt"])
    model = load_gcn_model(tmp_artefacts["gcn_ckpt"], temp_path=None)

    batch_probs = predict_batch(model, synthetic_graphs[:4])
    for i in range(4):
        single = predict_one(model, synthetic_graphs[i])
        assert abs(single - float(batch_probs[i].item())) < 1e-5


# =============================================================================
# Tests — HybridGATModel round-trip
# =============================================================================

def test_gat_train_save_load_predict_roundtrip(synthetic_graphs, tmp_artefacts):
    trained = _micro_train_gat(synthetic_graphs)
    _snapshot_gat(trained, tmp_artefacts["gat_ckpt"])

    reloaded = load_gat_model(tmp_artefacts["gat_ckpt"], temp_path=None)
    assert schema_of(reloaded) == CURRENT_FEATURE_SCHEMA_VERSION
    assert reloaded._meta_dim == META_DIM

    probs = predict_batch(reloaded, synthetic_graphs)
    assert probs.shape == (len(synthetic_graphs),)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


# =============================================================================
# Tests — contract violations (must fail loudly, never silently)
# =============================================================================

def test_missing_checkpoint_raises_filenotfound(tmp_artefacts):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_gcn_model(tmp_artefacts["gcn_ckpt"], temp_path=None)


def test_missing_sidecar_raises_filenotfound(synthetic_graphs, tmp_artefacts):
    trained = _micro_train_gcn(synthetic_graphs)
    torch.save(trained.state_dict(), tmp_artefacts["gcn_ckpt"])
    # Deliberately DO NOT save the sidecar.

    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        load_gcn_model(tmp_artefacts["gcn_ckpt"], temp_path=None)


def test_schema_mismatch_raises_feature_schema_mismatch(synthetic_graphs, tmp_artefacts):
    """A sidecar with a schema the serving code doesn't know must fail loudly."""
    trained = _micro_train_gcn(synthetic_graphs)
    torch.save(trained.state_dict(), tmp_artefacts["gcn_ckpt"])
    meta = ModelMetadata.from_hybrid_xg(trained, node_in=NODE_IN, edge_dim=0)
    # Corrupt the schema declaration.
    meta.feature_schema_version = "v0-stone-age"
    save_sidecar(meta, tmp_artefacts["gcn_ckpt"])

    with pytest.raises(FeatureSchemaMismatch):
        load_gcn_model(
            tmp_artefacts["gcn_ckpt"],
            temp_path=None,
            expected_schema=CURRENT_FEATURE_SCHEMA_VERSION,
        )


def test_model_class_mismatch_raises(synthetic_graphs, tmp_artefacts):
    """A GCN checkpoint must not load as a GAT (and vice-versa)."""
    trained = _micro_train_gcn(synthetic_graphs)
    _snapshot_gcn(trained, tmp_artefacts["gcn_ckpt"])

    with pytest.raises(FeatureSchemaMismatch):
        # Intentionally try to load a GCN ckpt via the GAT loader.
        load_gat_model(tmp_artefacts["gcn_ckpt"], temp_path=None)


def test_checksum_mismatch_raises(synthetic_graphs, tmp_artefacts):
    """A torn weights file must be caught by the sidecar checksum gate."""
    trained = _micro_train_gcn(synthetic_graphs)
    _snapshot_gcn(trained, tmp_artefacts["gcn_ckpt"])

    # Overwrite the .pt with different (valid-shape) bytes.
    other = _micro_train_gcn(synthetic_graphs, steps=1)
    torch.save(other.state_dict(), tmp_artefacts["gcn_ckpt"])

    with pytest.raises(FeatureSchemaMismatch, match="sha256"):
        load_gcn_model(
            tmp_artefacts["gcn_ckpt"],
            temp_path=None,
            verify_checksum=True,
        )


def test_missing_graph_attribute_raises_graph_schema_mismatch(
    synthetic_graphs, tmp_artefacts
):
    """A graph missing v3-psxg attributes must surface `GraphSchemaMismatch`,
    not silently substitute defaults (review §3.2)."""
    trained = _micro_train_gcn(synthetic_graphs)
    _snapshot_gcn(trained, tmp_artefacts["gcn_ckpt"])
    model = load_gcn_model(tmp_artefacts["gcn_ckpt"], temp_path=None)

    # Strip a required attribute from one graph.
    broken = _make_graph(seed=999)
    del broken.shot_placement
    with pytest.raises(GraphSchemaMismatch, match="shot_placement"):
        predict_one(model, broken)


# =============================================================================
# Tests — temperature scalar is applied when present
# =============================================================================

def test_temperature_scalar_changes_probability(synthetic_graphs, tmp_artefacts):
    """When a T-file is saved and loaded, predictions should shift toward
    0.5 if T > 1 (smoothing) or away from it if T < 1."""
    trained = _micro_train_gcn(synthetic_graphs)
    _snapshot_gcn(trained, tmp_artefacts["gcn_ckpt"])

    # Save T = 2.0 (heavy smoothing).
    torch.save({"T": 2.0}, tmp_artefacts["gcn_temp"])

    model_flat = load_gcn_model(tmp_artefacts["gcn_ckpt"], temp_path=None)
    model_hot  = load_gcn_model(
        tmp_artefacts["gcn_ckpt"], temp_path=tmp_artefacts["gcn_temp"]
    )

    probs_flat = predict_batch(model_flat, synthetic_graphs)
    probs_hot  = predict_batch(model_hot,  synthetic_graphs)

    # With T=2 the distance from 0.5 should shrink for every sample.
    dist_flat = (probs_flat - 0.5).abs()
    dist_hot  = (probs_hot  - 0.5).abs()
    assert torch.all(dist_hot <= dist_flat + 1e-5), (
        f"T=2.0 did not smooth predictions: flat={probs_flat}, hot={probs_hot}"
    )
