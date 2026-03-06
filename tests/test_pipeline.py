"""
test_pipeline.py
----------------
Smoke tests for the full data → graph → model pipeline.
No real data downloads required — all inputs are synthetic.

Run with:  pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from src.graph_builder import (
    EdgeStrategy,
    build_graph_from_freeze_frame,
    build_graph_from_tracking_frame,
)
from src.features import enrich_graph
from src.models.gcn import FootballGCN
from src.models.gat import FootballGAT


# ---------------------------------------------------------------------------
# Fixtures — synthetic data
# ---------------------------------------------------------------------------

def _mock_freeze_frame(n_teammates: int = 5, n_opponents: int = 6) -> list[dict]:
    """Fake StatsBomb 360 freeze-frame with random positions."""
    rng = np.random.default_rng(42)
    players = []
    for _ in range(n_teammates):
        players.append({
            "location": rng.uniform([0, 0], [120, 80]).tolist(),
            "teammate": True,
            "actor": False,
            "keeper": False,
        })
    for _ in range(n_opponents):
        players.append({
            "location": rng.uniform([0, 0], [120, 80]).tolist(),
            "teammate": False,
            "actor": False,
            "keeper": False,
        })
    return players


def _mock_tracking_frame(n_home: int = 11, n_away: int = 11):
    """Fake Metrica tracking frame with [0,1]² coordinates."""
    rng = np.random.default_rng(7)
    home_pos = rng.uniform(0, 1, (n_home, 2)).astype(np.float32)
    away_pos = rng.uniform(0, 1, (n_away, 2)).astype(np.float32)
    home_vel = rng.uniform(-0.01, 0.01, (n_home, 2)).astype(np.float32)
    away_vel = rng.uniform(-0.01, 0.01, (n_away, 2)).astype(np.float32)
    return home_pos, away_pos, home_vel, away_vel


# ---------------------------------------------------------------------------
# graph_builder tests
# ---------------------------------------------------------------------------

class TestGraphBuilder:

    def test_freeze_frame_returns_data_object(self):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff)
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None

    def test_freeze_frame_node_count(self):
        ff = _mock_freeze_frame(n_teammates=5, n_opponents=6)
        data = build_graph_from_freeze_frame(ff)
        assert data.x.shape[0] == 11

    def test_freeze_frame_node_features_shape(self):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff)
        # [x, y, team] → 3 features
        assert data.x.shape[1] == 3

    def test_freeze_frame_edge_features_shape(self):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff)
        n_edges = data.edge_index.shape[1]
        # [distance, delta_x, delta_y, same_team] → 4 features
        assert data.edge_attr.shape == (n_edges, 4)

    def test_freeze_frame_label(self):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff, label=1.0)
        assert data.y is not None
        assert data.y.item() == pytest.approx(1.0)

    @pytest.mark.parametrize("strategy", list(EdgeStrategy))
    def test_all_edge_strategies(self, strategy):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff, strategy=strategy)
        assert data.edge_index.shape[0] == 2

    def test_tracking_frame_full_squad(self):
        home_pos, away_pos, home_vel, away_vel = _mock_tracking_frame()
        data = build_graph_from_tracking_frame(
            home_pos, away_pos, home_vel, away_vel
        )
        assert data.x.shape[0] == 22
        # [x, y, vx, vy, team] → 5 features
        assert data.x.shape[1] == 5

    def test_tracking_frame_without_velocity(self):
        home_pos, away_pos, _, _ = _mock_tracking_frame()
        data = build_graph_from_tracking_frame(home_pos, away_pos)
        assert data.x.shape[1] == 3  # [x, y, team]

    def test_edge_index_valid_range(self):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff)
        n = data.x.shape[0]
        assert data.edge_index.max().item() < n
        assert data.edge_index.min().item() >= 0


# ---------------------------------------------------------------------------
# features tests
# ---------------------------------------------------------------------------

class TestFeatures:

    def test_enrich_adds_node_features(self):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff)
        original_n_feats = data.x.shape[1]
        data = enrich_graph(data)
        # should add dist_atk, dist_def, angle_atk, pressure → +4
        assert data.x.shape[1] == original_n_feats + 4

    def test_enrich_adds_edge_features(self):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff)
        original_e_feats = data.edge_attr.shape[1]
        data = enrich_graph(data)
        # should add pass_angle → +1
        assert data.edge_attr.shape[1] == original_e_feats + 1

    def test_enrich_with_roles(self):
        n = 11
        ff = _mock_freeze_frame(n_teammates=5, n_opponents=6)
        data = build_graph_from_freeze_frame(ff)
        roles = ["goalkeeper"] + ["center back"] * 4 + ["center midfield"] * 4 + ["center forward"] * 2
        data = enrich_graph(data, roles=roles)
        # role one-hot adds NUM_ROLES=17 columns
        from src.features import NUM_ROLES
        assert data.x.shape[1] == 3 + 4 + NUM_ROLES

    def test_enrich_tracking_adds_velocity_alignment(self):
        home_pos, away_pos, home_vel, away_vel = _mock_tracking_frame()
        data = build_graph_from_tracking_frame(
            home_pos, away_pos, home_vel, away_vel
        )
        original_e_feats = data.edge_attr.shape[1]
        data = enrich_graph(data)
        # pass_angle + velocity_alignment → +2
        assert data.edge_attr.shape[1] == original_e_feats + 2


# ---------------------------------------------------------------------------
# model tests
# ---------------------------------------------------------------------------

def _make_batch_of_graphs(n_graphs: int = 4, strategy=EdgeStrategy.DELAUNAY):
    """Create a PyG Batch from synthetic graphs."""
    graphs = []
    for i in range(n_graphs):
        ff = _mock_freeze_frame()
        data = build_graph_from_freeze_frame(ff, strategy=strategy, label=float(i % 2))
        data = enrich_graph(data)
        graphs.append(data)
    return Batch.from_data_list(graphs)


class TestGCN:

    def test_forward_shape_binary(self):
        batch = _make_batch_of_graphs(4)
        in_ch = batch.x.shape[1]
        model = FootballGCN(in_channels=in_ch, hidden_dim=32, out_channels=1)
        model.eval()
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (4, 1)

    def test_forward_shape_multiclass(self):
        batch = _make_batch_of_graphs(4)
        in_ch = batch.x.shape[1]
        model = FootballGCN(in_channels=in_ch, hidden_dim=32, out_channels=5)
        model.eval()
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (4, 5)

    def test_gradients_flow(self):
        batch = _make_batch_of_graphs(2)
        in_ch = batch.x.shape[1]
        model = FootballGCN(in_channels=in_ch, hidden_dim=32, out_channels=1)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestGAT:

    def test_forward_shape_with_edge_features(self):
        batch = _make_batch_of_graphs(4)
        in_ch = batch.x.shape[1]
        edge_dim = batch.edge_attr.shape[1]
        model = FootballGAT(
            in_channels=in_ch, edge_dim=edge_dim,
            hidden_dim=16, out_channels=1, heads=2
        )
        model.eval()
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch,
                        edge_attr=batch.edge_attr)
        assert out.shape == (4, 1)

    def test_forward_shape_without_edge_features(self):
        batch = _make_batch_of_graphs(4)
        in_ch = batch.x.shape[1]
        model = FootballGAT(in_channels=in_ch, edge_dim=0,
                            hidden_dim=16, out_channels=1, heads=2)
        model.eval()
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (4, 1)

    def test_gradients_flow(self):
        batch = _make_batch_of_graphs(2)
        in_ch = batch.x.shape[1]
        edge_dim = batch.edge_attr.shape[1]
        model = FootballGAT(in_channels=in_ch, edge_dim=edge_dim,
                            hidden_dim=16, out_channels=1, heads=2)
        out = model(batch.x, batch.edge_index, batch.batch,
                    edge_attr=batch.edge_attr)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
