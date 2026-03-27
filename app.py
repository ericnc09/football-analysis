"""
Football GNN Analysis Dashboard
================================
Interactive Streamlit app for exploring HybridGCN xG predictions
vs StatsBomb's industry xG across 8,013 shots from 7 competitions.

Run:
    streamlit run app.py
"""

import sys
import html as _html
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mplsoccer import Pitch, VerticalPitch
import streamlit as st

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import TECHNIQUE_INDEX, NUM_TECHNIQUES
from src.models.hybrid_gat import HybridGATModel
from src.calibration import TemperatureScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football GNN · xG Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS – dark theme ───────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f0f1a; }
  [data-testid="stSidebar"]          { background: #16213e; }
  .metric-card {
    background: #1a1a2e; border-radius: 8px; padding: 12px 14px;
    border-left: 3px solid #4FC3F7; margin-bottom: 8px;
  }
  .metric-card.gold  { border-left-color: #FFD54F; }
  .metric-card.green { border-left-color: #66BB6A; }
  .metric-card.red   { border-left-color: #EF5350; }
  .metric-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .metric-value { font-size: 22px; font-weight: 700; color: #e0e0e0; white-space: nowrap; }
  .metric-sub   { font-size: 11px; color: #aaa; margin-top: 2px; white-space: nowrap; }
  h1, h2, h3   { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PROCESSED      = REPO_ROOT / "data" / "processed"
MODEL_PATH          = PROCESSED / "pool_7comp_hybrid_xg.pt"
TEMP_PATH           = PROCESSED / "pool_7comp_T.pt"              # GCN global temperature
GAT_MODEL_PATH      = PROCESSED / "pool_7comp_hybrid_gat_xg.pt"
GAT_TEMP_PATH       = PROCESSED / "pool_7comp_gat_T.pt"          # GAT global temperature
PER_COMP_T_GCN_PATH = PROCESSED / "pool_7comp_per_comp_T_gcn.pt" # per-competition T (GCN)
PER_COMP_T_GAT_PATH = PROCESSED / "pool_7comp_per_comp_T_gat.pt" # per-competition T (GAT)
META_DIM            = 27  # full feature set (updated each sprint)
SURPRISE_XG_THRESHOLD = 0.15   # goals below this are "worldies"

# Technique index → display name
TECHNIQUE_NAMES: dict[int, str] = {v: k.title() for k, v in TECHNIQUE_INDEX.items()}
TECHNIQUE_NAMES[0] = "Normal"  # unknown bucket shown as Normal

COMPETITIONS = {
    "wc2022":         "🏆 FIFA World Cup 2022",
    "wwc2023":        "🏆 Women's World Cup 2023",
    "euro2020":       "🇪🇺 UEFA Euro 2020",
    "euro2024":       "🇪🇺 UEFA Euro 2024",
    "bundesliga2324": "🇩🇪 Bundesliga 2023/24",
    "weuro2022":      "🇪🇺 Women's Euro 2022",
    "weuro2025":      "🇪🇺 Women's Euro 2025",
}

# Pitch dimensions (StatsBomb 120×80 normalised to metres)
PL, PW = 105.0, 68.0   # pitch length, width
GOAL_X, GOAL_Y = PL, PW / 2


# ── Model definition (mirrors train_xg_hybrid.py) ────────────────────────────
class HybridXGModel(nn.Module):
    def __init__(self, in_channels=9, hidden_dim=64, meta_dim=15, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_dim),
            GCNConv(hidden_dim,  hidden_dim),
            GCNConv(hidden_dim,  hidden_dim),
        ])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch, metadata):
        return self.head(torch.cat([self.encode(x, edge_index, batch), metadata], dim=1))


# ── Gradient saliency ─────────────────────────────────────────────────────────
def compute_node_saliency(model, graph):
    """
    Gradient-based node importance for a single shot graph.

    Runs a forward pass with gradients enabled on node features, backprops
    the predicted xG, and returns per-node importance as the sum of absolute
    input gradients across all node features.

    Returns
    -------
    importance : np.ndarray, shape [n_nodes]
        Importance score per player; higher = more influential.
    """
    # Resolve actual meta_dim: check TemperatureScaler wrapper first, then bare model
    _scaler = load_model()
    meta_dim = getattr(_scaler, "_meta_dim", 15)

    model.eval()
    x = graph.x.clone().detach().float().requires_grad_(True)
    ei = graph.edge_index
    batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    base = torch.tensor([[
        float(graph.shot_dist.item()),
        float(graph.shot_angle.item()),
        float(graph.is_header.item()),
        float(graph.is_open_play.item()),
    ]])                                              # [1, 4]
    tech = graph.technique.unsqueeze(0)              # [1, 8]
    if meta_dim >= 15:
        gk   = torch.tensor([[
            float(graph.gk_dist.item()) if hasattr(graph, "gk_dist") else 20.0,
            float(graph.n_def_in_cone.item()) if hasattr(graph, "n_def_in_cone") else 0.0,
            float(graph.gk_off_centre.item()) if hasattr(graph, "gk_off_centre") else 0.0,
        ]])                                              # [1, 3]
        if meta_dim >= 18:
            new = torch.tensor([[
                float(graph.gk_perp_offset.item()) if hasattr(graph, "gk_perp_offset") else 3.0,
                float(graph.n_def_direct_line.item()) if hasattr(graph, "n_def_direct_line") else 0.0,
                float(graph.is_right_foot.item()) if hasattr(graph, "is_right_foot") else 0.5,
            ]])                                          # [1, 3]
            meta = torch.cat([base, tech, gk, new], dim=1)  # [1, 18]
        else:
            meta = torch.cat([base, tech, gk], dim=1)   # [1, 15]
    else:
        meta = torch.cat([base, tech], dim=1)            # [1, 12]

    logit = model(x, ei, batch, meta)
    pred  = torch.sigmoid(logit).squeeze()
    pred.backward()

    importance = x.grad.abs().sum(dim=1).detach().numpy()
    # Normalise 0→1 for display
    rng = importance.max() - importance.min()
    if rng > 1e-8:
        importance = (importance - importance.min()) / rng
    return importance


# ── GAT attention extraction ──────────────────────────────────────────────────
def compute_gat_attention(gat_model, graph) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Run forward_with_attention on a single graph using the HybridGATModel.

    Returns
    -------
    edge_index : (2, E) int array — src/dst node indices
    alpha      : (E,) float array — scalar importance per directed edge
                 (mean across heads and layers; softmax-normalised within-graph)
    Returns None if GAT model is not available.
    """
    if gat_model is None:
        return None

    # The gat_model is TemperatureScaler-wrapped; access the inner HybridGATModel
    inner = gat_model.model if isinstance(gat_model, TemperatureScaler) else gat_model

    # Infer the meta_dim the model was actually trained with from its head layer.
    # head[0] = Linear(pool_dim + meta_dim, pool_dim)
    actual_meta_dim = inner.head[0].in_features - inner._pool_dim

    inner.eval()
    x        = graph.x.clone().detach().float()
    ei       = graph.edge_index
    batch_v  = torch.zeros(graph.x.shape[0], dtype=torch.long)

    base = torch.tensor([[
        float(graph.shot_dist.item()),
        float(graph.shot_angle.item()),
        float(graph.is_header.item()),
        float(graph.is_open_play.item()),
    ]])                                  # [1, 4]
    tech = graph.technique.unsqueeze(0)  # [1, 8]

    if actual_meta_dim >= 15:
        gk = torch.tensor([[
            float(graph.gk_dist.item()) if hasattr(graph, "gk_dist") else 20.0,
            float(graph.n_def_in_cone.item()) if hasattr(graph, "n_def_in_cone") else 0.0,
            float(graph.gk_off_centre.item()) if hasattr(graph, "gk_off_centre") else 0.0,
        ]])                              # [1, 3]
        if actual_meta_dim >= 18:
            new = torch.tensor([[
                float(graph.gk_perp_offset.item()) if hasattr(graph, "gk_perp_offset") else 3.0,
                float(graph.n_def_direct_line.item()) if hasattr(graph, "n_def_direct_line") else 0.0,
                float(graph.is_right_foot.item()) if hasattr(graph, "is_right_foot") else 0.5,
            ]])                          # [1, 3]
            if actual_meta_dim >= 27:
                plc = (graph.shot_placement.unsqueeze(0)
                       if hasattr(graph, "shot_placement")
                       else torch.zeros(1, 9))   # [1, 9]
                meta = torch.cat([base, tech, gk, new, plc], dim=1)  # [1, 27]
            else:
                meta = torch.cat([base, tech, gk, new], dim=1)       # [1, 18]
        else:
            meta = torch.cat([base, tech, gk], dim=1)                # [1, 15]
    else:
        meta = torch.cat([base, tech], dim=1)                         # [1, 12]

    with torch.no_grad():
        _, alphas = inner.forward_with_attention(x, ei, batch_v, meta)

    # alphas: list of (E, heads) tensors — one per layer
    # Aggregate: mean over layers, mean over heads → (E,)
    alpha_stack = torch.stack(alphas)           # (n_layers, E, heads)
    alpha_E     = alpha_stack.mean(0).mean(-1)  # (E,)
    alpha_E     = alpha_E.cpu().numpy()

    return graph.edge_index.numpy(), alpha_E


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load HybridGCN + temperature scalar. Returns TemperatureScaler-wrapped model.

    Auto-detects meta_dim from the checkpoint so it works regardless of whether
    the model was trained with meta_dim=12 (base+technique) or 15 (+GK features).
    head.0.weight shape = (hidden_dim, hidden_dim + meta_dim)  → hidden_dim = 64
    """
    ckpt = torch.load(MODEL_PATH, weights_only=True, map_location="cpu")
    hidden_dim      = 64   # HybridXGModel default hidden_dim
    actual_meta_dim = int(ckpt["head.0.weight"].shape[1]) - hidden_dim
    base = HybridXGModel(meta_dim=actual_meta_dim)
    base.load_state_dict(ckpt)
    base.eval()
    scaler = TemperatureScaler.load(base, TEMP_PATH)
    scaler.eval()
    # Stash meta_dim so callers don't have to re-derive it
    scaler._meta_dim = actual_meta_dim
    return scaler


@st.cache_resource
def load_gat_model():
    """
    Load HybridGATModel + temperature scalar.
    Returns (model, available:bool).
    """
    if not GAT_MODEL_PATH.exists():
        return None, False
    try:
        # Infer node_in from the saved state dict
        ckpt = torch.load(GAT_MODEL_PATH, weights_only=True, map_location="cpu")
        # convs.0.lin_l.weight shape = (heads*hidden, node_in)
        node_in = ckpt["convs.0.lin_l.weight"].shape[1]
        # edge_dim: convs.0.lin_edge exists only when edge_dim > 0
        edge_dim = (ckpt["convs.0.lin_edge.weight"].shape[1]
                    if "convs.0.lin_edge.weight" in ckpt else 0)
        # Auto-detect meta_dim from saved head weights:
        # head.0 = Linear(pool_dim + meta_dim, pool_dim); pool_dim = hidden = 32
        _pool_dim = 32
        actual_meta_dim = int(ckpt["head.0.weight"].shape[1]) - _pool_dim
        gat = HybridGATModel(node_in=node_in, edge_dim=edge_dim,
                             meta_dim=actual_meta_dim, hidden=32, heads=4,
                             n_layers=3, dropout=0.3)
        gat.load_state_dict(ckpt)
        gat.eval()
        scaler = TemperatureScaler.load(gat, GAT_TEMP_PATH)
        scaler.eval()
        return scaler, True
    except Exception as e:
        return None, False


@st.cache_resource
def load_competition(key: str):
    # Path is constructed entirely from our own PROCESSED directory — never from
    # user input — so weights_only=False is safe here.  PyG Data objects cannot
    # be serialised with weights_only=True because they contain custom classes.
    path = (PROCESSED / f"statsbomb_{key}_shot_graphs.pt").resolve()
    if not str(path).startswith(str(PROCESSED.resolve())):
        raise ValueError(f"Path traversal detected: {path}")
    if not path.exists():
        return []
    return torch.load(path, weights_only=False)  # nosec: trusted internal data only


def _build_meta(batch, meta_dim: int = 27) -> torch.Tensor:
    """Build metadata tensor for a PyG batch.

    Handles all historical meta_dim sizes gracefully:
      12 = base(4) + technique(8)
      15 = 12 + gk_original(3)
      18 = 15 + gk_precision(3)
      27 = 18 + shot_placement(9)   ← current target (PSxG)
    Falls back gracefully when newer attributes are missing from old graphs.
    """
    n = batch.shot_dist.shape[0]

    base = torch.stack([
        batch.shot_dist.squeeze(),
        batch.shot_angle.squeeze(),
        batch.is_header.squeeze().float(),
        batch.is_open_play.squeeze().float(),
    ], dim=1)                              # [n, 4]
    tech = batch.technique.view(-1, 8)    # [n, 8]

    if meta_dim < 15:
        return torch.cat([base, tech], dim=1)   # [n, 12]

    gk = torch.stack([
        batch.gk_dist.squeeze(),
        batch.n_def_in_cone.squeeze(),
        batch.gk_off_centre.squeeze(),
    ], dim=1)                              # [n, 3]

    if meta_dim < 18:
        return torch.cat([base, tech, gk], dim=1)   # [n, 15]

    def _safe(attr, default):
        if hasattr(batch, attr):
            return getattr(batch, attr).squeeze()
        return torch.full((n,), default)

    new = torch.stack([
        _safe("gk_perp_offset",    3.0),   # metres; 3.0 = GK slightly off line
        _safe("n_def_direct_line", 0.0),   # count
        _safe("is_right_foot",     0.5),   # unknown → neutral 0.5
    ], dim=1)                              # [n, 3]

    if meta_dim < 27:
        return torch.cat([base, tech, gk, new], dim=1)   # [n, 18]

    # PSxG placement: 9-dim one-hot goal-face zone (safe fallback for old graphs)
    if hasattr(batch, "shot_placement"):
        plc = batch.shot_placement.view(-1, 9)   # [n, 9]
    else:
        plc = torch.zeros(n, 9)                  # [n, 9] — unknown zone

    return torch.cat([base, tech, gk, new, plc], dim=1)   # [n, 27]


@st.cache_resource
def get_predictions(key: str):
    graphs = load_competition(key)
    if not graphs:
        return np.array([])
    model    = load_model()   # TemperatureScaler-wrapped; forward() applies T
    meta_dim = getattr(model, "_meta_dim", 15)
    loader   = DataLoader(graphs, batch_size=256, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            meta   = _build_meta(batch, meta_dim)
            logits = model(batch.x, batch.edge_index, batch.batch, meta)
            probs.extend(torch.sigmoid(logits.squeeze()).tolist())
    return np.array(probs)


@st.cache_resource
def build_match_index(key: str) -> dict:
    """
    Group graphs by match_id.
    Returns {match_id: {graphs, home_team, away_team, label}}, sorted by match_id.
    """
    graphs = load_competition(key)
    index: dict[int, dict] = {}
    for g in graphs:
        mid = int(g.match_id.item())
        if mid not in index:
            index[mid] = {"graphs": [], "home_team": g.home_team, "away_team": ""}
        index[mid]["graphs"].append(g)
        # Determine away team: first team_name that differs from home_team
        if not index[mid]["away_team"] and g.team_name != g.home_team:
            index[mid]["away_team"] = g.team_name

    # Build display labels
    for mid, entry in index.items():
        n = len(entry["graphs"])
        home = entry["home_team"] or "Home"
        away = entry["away_team"] or "Away"
        entry["label"] = f"{home} vs {away}  ({n} shots)"

    return dict(sorted(index.items()))


def get_match_predictions(match_graphs: list) -> np.ndarray:
    """Run inference on a match-scoped list of graphs. Returns probs array (N,)."""
    model    = load_model()   # T-scaled
    meta_dim = getattr(model, "_meta_dim", 15)
    loader   = DataLoader(match_graphs, batch_size=len(match_graphs), shuffle=False)
    with torch.no_grad():
        batch  = next(iter(loader))
        meta   = _build_meta(batch, meta_dim)
        logits = model(batch.x, batch.edge_index, batch.batch, meta)
        probs  = torch.sigmoid(logits.squeeze()).numpy()
    return np.atleast_1d(probs)


def _match_kpis(graphs: list, probs: np.ndarray) -> dict:
    """Compute {shots, goals, xG_hybrid, xG_sb} for a list of graphs."""
    return {
        "shots":     len(graphs),
        "goals":     int(sum(int(g.y.item()) for g in graphs)),
        "xG_hybrid": float(probs.sum()) if len(probs) else 0.0,
        "xG_sb":     float(sum(g.sb_xg.item() for g in graphs)),
    }


def _draw_shot_map_on_ax(ax, graphs: list, probs: np.ndarray, title: str) -> None:
    """Draw a half-pitch shot map on an existing Axes."""
    pitch = VerticalPitch(
        pitch_type="custom", pitch_length=PL, pitch_width=PW,
        pitch_color=PANEL_BG, line_color="#555", half=True,
        goal_type="box", pad_top=2, pad_bottom=2,
    )
    pitch.draw(ax=ax)

    goals_mask = np.array([int(g.y.item()) for g in graphs]) == 1
    shooter_x  = np.array([g.x[g.x[:, 3] == 1][0, 0].item() for g in graphs])
    shooter_y  = np.array([g.x[g.x[:, 3] == 1][0, 1].item() for g in graphs])

    if (~goals_mask).any():
        pitch.scatter(shooter_x[~goals_mask], shooter_y[~goals_mask], ax=ax,
                      c=probs[~goals_mask], cmap="RdYlGn", vmin=0, vmax=1,
                      s=55, alpha=0.65, zorder=3, marker="o")
    if goals_mask.any():
        pitch.scatter(shooter_x[goals_mask], shooter_y[goals_mask], ax=ax,
                      c=probs[goals_mask], cmap="RdYlGn", vmin=0, vmax=1,
                      s=180, alpha=0.95, zorder=5, marker="*",
                      edgecolors="white", linewidths=0.8)

    ax.set_title(title, color=TEXT_COLOR, fontsize=10, fontweight="bold")


def _draw_cumulative_xg(ax, match_graphs: list, match_probs: np.ndarray,
                         home_team: str, away_team: str) -> None:
    """Draw step-function cumulative xG timeline, one line per team."""
    home_mask = np.array([g.team_name == home_team for g in match_graphs])
    away_mask = ~home_mask

    def _cum_series(mask):
        subset = [(int(match_graphs[i].minute.item()), float(match_probs[i]))
                  for i in range(len(match_graphs)) if mask[i]]
        subset.sort(key=lambda t: t[0])
        if not subset:
            return [0], [0.0]
        mins, xgs = zip(*subset)
        cumxg = np.cumsum(xgs).tolist()
        # Prepend 0 at minute 0 for clean step start
        return [0] + list(mins), [0.0] + cumxg

    home_mins, home_cum = _cum_series(home_mask)
    away_mins, away_cum = _cum_series(away_mask)

    ax.step(home_mins, home_cum, where="post", color="#4FC3F7", lw=2.2,
            label=f"{home_team or 'Home'}  ({home_cum[-1]:.2f} xG)")
    ax.step(away_mins, away_cum, where="post", color="#EF5350", lw=2.2,
            label=f"{away_team or 'Away'}  ({away_cum[-1]:.2f} xG)")

    # Mark goals with vertical dashed lines
    for i, g in enumerate(match_graphs):
        if int(g.y.item()) == 1:
            min_ = int(g.minute.item())
            col  = "#4FC3F7" if home_mask[i] else "#EF5350"
            ax.axvline(min_, color=col, ls="--", lw=0.8, alpha=0.55)
            ax.text(min_ + 0.4, ax.get_ylim()[1] * 0.92, "⚽",
                    fontsize=7, color=col, alpha=0.8)

    ax.axhline(1.0, color="#555", ls=":", lw=0.8, alpha=0.5)
    ax.set(xlabel="Minute", ylabel="Cumulative xG",
           title="Cumulative xG Timeline")
    ax.title.set_color(TEXT_COLOR)
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, edgecolor="#444")
    ax.grid(alpha=0.2)


def match_report_figure(match_graphs: list, match_probs: np.ndarray,
                         home_team: str, away_team: str) -> plt.Figure:
    """
    4-panel match report:
      Row 0: [home shot map] [away shot map] [KPI text panel]
      Row 1: [cumulative xG timeline — full width]
    """
    home_mask  = np.array([g.team_name == home_team for g in match_graphs])
    away_mask  = ~home_mask
    home_graphs = [g for g, m in zip(match_graphs, home_mask) if m]
    away_graphs = [g for g, m in zip(match_graphs, away_mask) if m]
    home_probs  = match_probs[home_mask]
    away_probs  = match_probs[away_mask]

    home_kpi = _match_kpis(home_graphs, home_probs)
    away_kpi = _match_kpis(away_graphs, away_probs)

    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.0], hspace=0.38, wspace=0.28)
    ax_home     = fig.add_subplot(gs[0, 0])
    ax_away     = fig.add_subplot(gs[0, 1])
    ax_kpi      = fig.add_subplot(gs[0, 2])
    ax_timeline = fig.add_subplot(gs[1, :])

    # Shot maps
    _draw_shot_map_on_ax(ax_home, home_graphs, home_probs,
                          f"{home_team or 'Home'} shots")
    _draw_shot_map_on_ax(ax_away, away_graphs, away_probs,
                          f"{away_team or 'Away'} shots")

    # KPI panel
    ax_kpi.set_facecolor(PANEL_BG)
    ax_kpi.axis("off")
    lines = [
        ("", "#aaa"),
        (f"{'Metric':<14}  {'Home':>8}  {'Away':>8}", "#888"),
        ("─" * 34, "#555"),
        (f"{'Shots':<14}  {home_kpi['shots']:>8}  {away_kpi['shots']:>8}", TEXT_COLOR),
        (f"{'Goals':<14}  {home_kpi['goals']:>8}  {away_kpi['goals']:>8}", "#66BB6A"),
        (f"{'xG (model)':<14}  {home_kpi['xG_hybrid']:>8.2f}  {away_kpi['xG_hybrid']:>8.2f}", "#4FC3F7"),
        (f"{'xG (SB)':<14}  {home_kpi['xG_sb']:>8.2f}  {away_kpi['xG_sb']:>8.2f}", "#FFD54F"),
    ]
    for i, (line, color) in enumerate(lines):
        ax_kpi.text(0.05, 0.88 - i * 0.13, line, transform=ax_kpi.transAxes,
                    fontsize=9.5, color=color, fontfamily="monospace",
                    verticalalignment="top")

    # Timeline
    ax_timeline.set_facecolor(PANEL_BG)
    _draw_cumulative_xg(ax_timeline, match_graphs, match_probs, home_team, away_team)

    return fig


# ── Drawing helpers ───────────────────────────────────────────────────────────
DARK_BG    = "#0f0f1a"
PANEL_BG   = "#16213e"
TEXT_COLOR = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
    "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "axes.edgecolor": "#333",
})


def shot_map_figure(graphs, hybrid_probs, title=""):
    """All shots on the attacking half, coloured by HybridGCN xG."""
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=DARK_BG)
    pitch = VerticalPitch(
        pitch_type="custom", pitch_length=PL, pitch_width=PW,
        pitch_color=PANEL_BG, line_color="#555", half=True,
        goal_type="box", pad_top=2, pad_bottom=2,
    )
    pitch.draw(ax=ax)

    goals_mask  = np.array([int(g.y.item()) for g in graphs]) == 1
    misses_mask = ~goals_mask

    # Extract shooter position (actor node) for each graph
    # x[:,0] = x_m (pitch length 0-105), x[:,1] = y_m (pitch width 0-68)
    # VerticalPitch.scatter(x, y) → x=length coord, y=width coord
    shooter_x = np.array([g.x[g.x[:, 3] == 1][0, 0].item() for g in graphs])  # length (0-105)
    shooter_y = np.array([g.x[g.x[:, 3] == 1][0, 1].item() for g in graphs])  # width  (0-68)

    # Misses
    sc = pitch.scatter(
        shooter_x[misses_mask], shooter_y[misses_mask], ax=ax,
        c=hybrid_probs[misses_mask], cmap="RdYlGn", vmin=0, vmax=1,
        s=55, alpha=0.65, zorder=3, marker="o",
    )

    # Goals – outlined star
    pitch.scatter(
        shooter_x[goals_mask], shooter_y[goals_mask], ax=ax,
        c=hybrid_probs[goals_mask], cmap="RdYlGn", vmin=0, vmax=1,
        s=180, alpha=0.95, zorder=5, marker="*",
        edgecolors="white", linewidths=0.8,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("HybridGCN xG", color=TEXT_COLOR, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=8)

    legend = [
        mpatches.Patch(color="#66BB6A", label=f"Goal ({goals_mask.sum()})"),
        mpatches.Patch(color="#888",    label=f"Miss ({misses_mask.sum()})"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="lower left",
              facecolor=PANEL_BG, labelcolor=TEXT_COLOR, edgecolor="#444")
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def freeze_frame_figure(graph, hybrid_xg, shot_index, total,
                         node_importance=None, show_saliency=True,
                         attn_data=None, show_attention=False):
    """
    Full pitch showing all visible players at the moment of a shot.
    Actor = shooter (gold star), teammates (green), defenders (red), keeper (orange).

    If node_importance is provided and show_saliency is True, draws gradient-saliency
    edges from the shooter to the top-5 most influential players.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG,
                             gridspec_kw={"width_ratios": [2.2, 1]})

    # ── Left: freeze frame on pitch ─────────────────────────────────────────
    ax = axes[0]
    pitch = Pitch(
        pitch_type="custom", pitch_length=PL, pitch_width=PW,
        pitch_color=PANEL_BG, line_color="#555", goal_type="box",
    )
    pitch.draw(ax=ax)

    nodes = graph.x.numpy()   # [n, 9]: x_m, y_m, teammate, actor, keeper, ...
    xs, ys = nodes[:, 0], nodes[:, 1]
    teammate = nodes[:, 2].astype(bool)
    actor    = nodes[:, 3].astype(bool)
    keeper   = nodes[:, 4].astype(bool)
    opponent = (~teammate) & (~actor) & (~keeper)

    # Attacking teammates
    if teammate.any():
        pitch.scatter(xs[teammate], ys[teammate], ax=ax,
                      color="#66BB6A", s=200, zorder=5, edgecolors="white",
                      linewidths=0.8, label="Attacker")

    # Outfield defenders
    if opponent.any():
        pitch.scatter(xs[opponent], ys[opponent], ax=ax,
                      color="#EF5350", s=200, zorder=5, edgecolors="white",
                      linewidths=0.8, label="Defender")

    # Keeper
    if keeper.any():
        pitch.scatter(xs[keeper], ys[keeper], ax=ax,
                      color="#FF9800", s=250, zorder=6, edgecolors="white",
                      linewidths=1.0, marker="D", label="Goalkeeper")

    # Shooter
    if actor.any():
        sx, sy = xs[actor][0], ys[actor][0]
        pitch.scatter(sx, sy, ax=ax,
                      color="#FFD54F", s=350, zorder=7, edgecolors="white",
                      linewidths=1.2, marker="*", label="Shooter")
        # Arrow to goal centre
        pitch.arrows(sx, sy, GOAL_X, GOAL_Y, ax=ax,
                     color="#FFD54F", width=1.2, headwidth=4, headlength=4,
                     alpha=0.7, zorder=4)

        # ── Gradient-saliency overlay ─────────────────────────────────────
        if show_saliency and node_importance is not None and len(node_importance) > 0:
            # Zero out the shooter's own importance so we rank other players
            imp = node_importance.copy()
            actor_idx = np.where(actor)[0]
            if len(actor_idx):
                imp[actor_idx[0]] = 0.0

            # Top-5 most influential other players
            top_k = min(5, (imp > 0).sum())
            if top_k > 0:
                top_nodes = np.argsort(imp)[::-1][:top_k]
                imp_vals  = imp[top_nodes]

                saliency_cmap = plt.cm.get_cmap("cool")
                for rank, (ni, iv) in enumerate(zip(top_nodes, imp_vals)):
                    if iv < 0.05:
                        continue
                    tx, ty = xs[ni], ys[ni]
                    col   = saliency_cmap(0.4 + 0.6 * iv)
                    lw    = 0.8 + 3.5 * iv
                    alpha = 0.35 + 0.55 * iv
                    ax.plot([sx, tx], [sy, ty], color=col, lw=lw, alpha=alpha,
                            zorder=3, solid_capstyle="round",
                            linestyle=(0, (3, 2)) if rank > 1 else "-")
                    # Importance score label near the player
                    ax.text(tx + 0.8, ty + 0.5, f"{iv:.2f}",
                            fontsize=6.5, color=col, alpha=0.85, zorder=8,
                            ha="left", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.2", fc="#0f0f1a", ec="none", alpha=0.6))

        # ── GAT Attention overlay ─────────────────────────────────────────
        if show_attention and attn_data is not None:
            edge_index_np, alpha_E = attn_data
            # Filter edges that originate from or connect to the shooter
            src_nodes, dst_nodes = edge_index_np[0], edge_index_np[1]
            actor_idx = int(np.where(actor)[0][0]) if actor.any() else -1

            # Rank all edges by attention weight; take top-3
            top_k     = min(3, len(alpha_E))
            top_edges = np.argsort(alpha_E)[::-1][:top_k]

            attn_cmap = plt.cm.get_cmap("plasma")
            # Normalise within top-k for colour mapping
            top_alphas = alpha_E[top_edges]
            a_max = top_alphas.max() if top_alphas.max() > 1e-8 else 1.0

            for rank, eidx in enumerate(top_edges):
                src_i, dst_i = int(src_nodes[eidx]), int(dst_nodes[eidx])
                a_val = float(alpha_E[eidx])
                a_norm = a_val / a_max

                x0, y0 = xs[src_i], ys[src_i]
                x1, y1 = xs[dst_i], ys[dst_i]
                col   = attn_cmap(0.3 + 0.7 * a_norm)
                lw    = 1.0 + 4.0 * a_norm
                alpha = 0.45 + 0.45 * a_norm

                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color=col,
                        lw=lw, alpha=alpha,
                        mutation_scale=10 + 8 * a_norm,
                        connectionstyle="arc3,rad=0.15",
                    ),
                    zorder=8,
                )
                # Label: rank + alpha value near midpoint
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(mid_x + 0.5, mid_y + 0.5,
                        f"#{rank+1} α={a_val:.3f}",
                        fontsize=6, color=col, alpha=0.9, zorder=9,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="#0f0f1a", ec="none", alpha=0.55))

    ax.legend(fontsize=8, loc="upper left", facecolor=PANEL_BG,
              labelcolor=TEXT_COLOR, edgecolor="#444")

    # Legend note
    if show_saliency and node_importance is not None:
        ax.text(0.01, 0.01,
                "── gradient saliency edges: thicker = higher model influence",
                transform=ax.transAxes, fontsize=6.5, color="#aaa",
                alpha=0.75, va="bottom")
    elif show_attention and attn_data is not None:
        ax.text(0.01, 0.01,
                "── GAT attention: top-3 player pairs model focused on",
                transform=ax.transAxes, fontsize=6.5, color="#e040fb",
                alpha=0.85, va="bottom")

    goal = int(graph.y.item())
    outcome_str = "⚽ GOAL" if goal else "✗ No goal"
    outcome_col = "#66BB6A" if goal else "#EF5350"
    ax.set_title(
        f"Shot {shot_index + 1} of {total}  ·  {outcome_str}",
        color=outcome_col, fontsize=11, fontweight="bold",
    )

    # ── Right: xG comparison panel ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)

    sb_xg   = graph.sb_xg.item()
    dist_m  = graph.shot_dist.item()
    angle_d = np.degrees(graph.shot_angle.item())
    header  = bool(graph.is_header.item())
    op      = bool(graph.is_open_play.item())
    tech_label = TECHNIQUE_NAMES.get(int(graph.technique.argmax().item()), "Normal")

    models  = ["StatsBomb xG\n(industry)", "HybridGCN\n(our model)"]
    values  = [sb_xg, hybrid_xg]
    colors  = ["#FFD54F", "#4FC3F7"]
    bars = ax2.barh(models, values, color=colors, alpha=0.85,
                    height=0.45, edgecolor="#333")
    for bar, val in zip(bars, values):
        ax2.text(min(val + 0.015, 0.92), bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", ha="left",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax2.set_xlim(0, 1.0)
    ax2.axvline(0.5, color="#555", ls="--", lw=0.8, alpha=0.6)
    ax2.set_xlabel("xG (probability of goal)", color=TEXT_COLOR, fontsize=9)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax2.set_title("xG Comparison", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax2.grid(axis="x", alpha=0.2)

    # Metadata text below bars
    meta_lines = [
        f"Distance  {dist_m:.1f} m",
        f"Angle     {angle_d:.1f}°",
        f"Type      {'Header' if header else 'Foot'}",
        f"Technique {tech_label}",
        f"Situation {'Open play' if op else 'Set piece'}",
        f"n players {graph.x.shape[0]}",
    ]
    meta_txt = "\n".join(meta_lines)
    ax2.text(0.03, -0.32, meta_txt, transform=ax2.transAxes,
             fontsize=8.5, color="#aaa", fontfamily="monospace",
             verticalalignment="top", clip_on=False)

    plt.tight_layout()
    return fig


def reliability_diagram_figure(graphs, probs_cal, n_bins=10):
    """
    Reliability diagram (calibration curve) + Brier score bar chart.

    Left panel  — predicted xG bucket vs actual conversion rate.
                  A perfectly calibrated model hugs the diagonal.
    Right panel — Brier score comparison: HybridGCN+T vs StatsBomb xG.
    """
    from sklearn.metrics import brier_score_loss

    labels_np = np.array([int(g.y.item()) for g in graphs])
    sb_probs  = np.array([g.sb_xg.item() for g in graphs])
    bins      = np.linspace(0, 1, n_bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=DARK_BG)

    # ── Left: calibration curves ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)

    for probs, name, color, marker in [
        (probs_cal, "HybridGCN + T-scaling", "#4FC3F7", "o"),
        (sb_probs,  "StatsBomb xG",          "#FFD54F", "s"),
    ]:
        centers, fracs, counts = [], [], []
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() >= 5:
                centers.append((bins[i] + bins[i + 1]) / 2)
                fracs.append(float(labels_np[mask].mean()))
                counts.append(int(mask.sum()))

        if centers:
            ax.plot(centers, fracs, f"{marker}-", color=color, lw=2.2, ms=7,
                    label=name, zorder=4)
            # Histogram silhouette at bottom showing sample density
            ax.bar(centers, [c / max(counts) * 0.07 for c in counts],
                   width=0.07, bottom=0.0, color=color, alpha=0.18,
                   zorder=2, align="center")

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.3,
            label="Perfect calibration", zorder=3)
    ax.fill_between([0, 1], [0, 1], 1.05,
                    alpha=0.06, color="#EF5350", label="Over-confident zone")
    ax.fill_between([0, 1], 0, [0, 1],
                    alpha=0.06, color="#4FC3F7", label="Under-confident zone")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set(xlabel="Predicted xG (model output)",
           ylabel="Actual conversion rate",
           title="Reliability Diagram — Calibration Curve")
    ax.title.set_color(TEXT_COLOR)
    ax.legend(fontsize=8.5, facecolor=PANEL_BG, labelcolor=TEXT_COLOR,
              edgecolor="#444", loc="upper left")
    ax.grid(alpha=0.2)
    ax.tick_params(colors=TEXT_COLOR)

    # ── Right: Brier score bar chart ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)

    model_names = ["HybridGCN + T", "StatsBomb xG"]
    brier_vals  = [
        brier_score_loss(labels_np, probs_cal),
        brier_score_loss(labels_np, sb_probs),
    ]
    bar_colors = ["#4FC3F7", "#FFD54F"]

    bars = ax2.barh(model_names, brier_vals, color=bar_colors,
                    alpha=0.85, height=0.4, edgecolor="#333")
    for bar, val in zip(bars, brier_vals):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", ha="left",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    ax2.axvline(0.10, color="#66BB6A", ls="--", lw=1.2, alpha=0.8,
                label="Target ≤ 0.100")
    ax2.set_xlim(0, max(brier_vals) * 1.3)
    ax2.set(xlabel="Brier Score  (lower = better, max = 1.0)",
            title="Calibration Quality — Brier Score")
    ax2.title.set_color(TEXT_COLOR)
    ax2.legend(fontsize=8.5, facecolor=PANEL_BG, labelcolor=TEXT_COLOR,
               edgecolor="#444")
    ax2.grid(axis="x", alpha=0.2)
    ax2.tick_params(colors=TEXT_COLOR)

    fig.suptitle("Model Calibration", color=TEXT_COLOR,
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def xg_distribution_figure(graphs, hybrid_probs):
    """Side-by-side histograms: Hybrid vs StatsBomb xG, goals vs misses."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=DARK_BG)
    labels = np.array([int(g.y.item()) for g in graphs])
    sb_all = np.array([g.sb_xg.item() for g in graphs])
    bins   = np.linspace(0, 1, 21)

    for ax, probs, name, c in [
        (axes[0], hybrid_probs, "HybridGCN", "#4FC3F7"),
        (axes[1], sb_all,       "StatsBomb xG", "#FFD54F"),
    ]:
        ax.hist(probs[labels == 0], bins=bins, alpha=0.6, color="#EF5350",
                label="Miss", density=True)
        ax.hist(probs[labels == 1], bins=bins, alpha=0.75, color="#66BB6A",
                label="Goal", density=True)
        ax.axvline(probs[labels == 1].mean(), color="#66BB6A", ls="--", lw=1.5,
                   label=f"Goal mean {probs[labels==1].mean():.2f}")
        ax.axvline(probs[labels == 0].mean(), color="#EF5350", ls="--", lw=1.5,
                   label=f"Miss mean {probs[labels==0].mean():.2f}")
        ax.set(title=name, xlabel="Predicted xG", ylabel="Density")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)
        ax.tick_params(colors=TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)

    fig.suptitle("xG Distributions — Goals vs Misses", color=TEXT_COLOR,
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Football GNN")
    st.markdown("**HybridGCN xG** trained on 8,013 shots  \nfrom 7 StatsBomb 360 competitions")
    st.markdown("---")

    comp_key = st.selectbox(
        "Competition",
        options=list(COMPETITIONS.keys()),
        format_func=lambda k: COMPETITIONS[k],
    )

    st.markdown("---")
    view = st.radio("View", ["📍 Shot Map", "🔬 Shot Inspector", "📊 xG Distributions", "📋 Match Report", "🌟 Surprise Goals", "👤 Player Profile"])

    st.markdown("---")

    # Match selector — only shown in Match Report view
    match_id_sel = None
    if view == "📋 Match Report":
        st.markdown("**Match**")
        _match_index_placeholder = st.empty()  # filled after data loads

    st.markdown("---")
    _scaler_sb = load_model()
    _T_sb = _scaler_sb.temperature if isinstance(_scaler_sb, TemperatureScaler) else 1.0
    _gat_sb, _gat_avail_sb = load_gat_model()
    _T_gat = _gat_sb.temperature if (_gat_avail_sb and isinstance(_gat_sb, TemperatureScaler)) else "—"
    st.markdown(
        "**Model:** HybridGAT+T (GAT + metadata + placement)  \n"
        "**Test AUC:** 0.763  ·  **Brier:** 0.159  \n"
        "**vs StatsBomb xG:** 0.794 AUC  \n"
        "**Data:** 326 matches · 7 competitions"
    )
    st.markdown("---")
    _T_gat_str = f"{_T_gat:.3f}" if isinstance(_T_gat, float) else str(_T_gat)
    st.markdown(
        f"🌡️ **Temperature scaling (global)**  \n"
        f"GCN  T = `{_T_sb:.3f}`  \n"
        f"GAT  T = `{_T_gat_str}`"
    )
    # Per-competition T breakdown
    _per_comp_T_gat = {}
    if PER_COMP_T_GAT_PATH.exists():
        try:
            _per_comp_T_gat = torch.load(PER_COMP_T_GAT_PATH, weights_only=True)
        except Exception:
            pass
    if _per_comp_T_gat:
        with st.expander("Per-competition T (GAT)", expanded=False):
            for _cl, _ct in sorted(_per_comp_T_gat.items()):
                st.markdown(f"`{_cl}` → **{_ct:.3f}**")


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {COMPETITIONS[comp_key]}…"):
    graphs       = load_competition(comp_key)
    hybrid_probs = get_predictions(comp_key)

if not graphs:
    st.error(f"No data found for {comp_key}. Run `build_shot_graphs.py` first.")
    st.stop()

labels  = np.array([int(g.y.item()) for g in graphs])
sb_xgs  = np.array([g.sb_xg.item()  for g in graphs])
n_goals = int(labels.sum())
n_total = len(graphs)

# ── Header KPIs ───────────────────────────────────────────────────────────────
st.markdown(f"# {COMPETITIONS[comp_key]}")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total shots</div>
        <div class="metric-value">{n_total:,}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card green">
        <div class="metric-label">Goals</div>
        <div class="metric-value">{n_goals}</div>
        <div class="metric-sub">{100*n_goals/n_total:.1f}% conversion</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">HybridGCN mean xG</div>
        <div class="metric-value">{hybrid_probs.mean():.3f}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card gold">
        <div class="metric-label">StatsBomb mean xG</div>
        <div class="metric-value">{sb_xgs.mean():.3f}</div>
    </div>""", unsafe_allow_html=True)
with c5:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, hybrid_probs) if len(np.unique(labels)) > 1 else 0.5
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">HybridGCN AUC</div>
        <div class="metric-value">{auc:.3f}</div>
        <div class="metric-sub">SB ref: {roc_auc_score(labels, sb_xgs):.3f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ── Views ─────────────────────────────────────────────────────────────────────

if view == "📍 Shot Map":
    st.markdown("### Shot Map — HybridGCN xG")

    # ── Filters row ───────────────────────────────────────────────────────────
    sm_f1, sm_f2, sm_f3 = st.columns([1, 1.6, 2.4])

    with sm_f1:
        sm_outcome = st.selectbox(
            "Outcome", ["All shots", "Goals only", "Misses only"],
            key="shotmap_outcome",
        )

    # Build team list from the loaded competition
    _all_teams = sorted(set(g.team_name for g in graphs if g.team_name))
    with sm_f2:
        sm_team = st.selectbox(
            "Team / Country",
            options=["All teams"] + _all_teams,
            key="shotmap_team",
        )

    # Apply outcome filter first
    if sm_outcome == "Goals only":
        sm_mask = labels == 1
    elif sm_outcome == "Misses only":
        sm_mask = labels == 0
    else:
        sm_mask = np.ones(len(graphs), dtype=bool)

    # Apply team filter on top
    if sm_team != "All teams":
        team_mask = np.array([g.team_name == sm_team for g in graphs])
        sm_mask   = sm_mask & team_mask

    sm_graphs = [g for g, m in zip(graphs, sm_mask) if m]
    sm_probs  = hybrid_probs[sm_mask]
    sm_labels = labels[sm_mask]

    sm_n = int(sm_mask.sum())
    sm_g = int((sm_labels == 1).sum())

    with sm_f3:
        map_title = (
            f"{sm_team} — {COMPETITIONS[comp_key]}"
            if sm_team != "All teams" else COMPETITIONS[comp_key]
        )
        st.caption(
            f"Showing **{sm_n}** shots  ·  **{sm_g}** goals  ·  "
            "★ = goal  ·  ● = miss  ·  colour = HybridGCN xG  ·  green=high, red=low"
        )

    col_l, col_r = st.columns([3, 1])
    with col_l:
        fig = shot_map_figure(sm_graphs, sm_probs, title=map_title)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        # CSV export of currently filtered shots
        if len(sm_graphs) > 0:
            import pandas as pd
            export_rows = []
            for g, p in zip(sm_graphs, sm_probs):
                tech_idx = int(g.technique.argmax().item())
                export_rows.append({
                    "player":    g.player_name or "Unknown",
                    "team":      g.team_name   or "—",
                    "minute":    int(g.minute.item()) if hasattr(g, "minute") else 0,
                    "outcome":   "Goal" if int(g.y.item()) == 1 else "Miss",
                    "xG_model":  round(float(p), 4),
                    "xG_sb":     round(float(g.sb_xg.item()), 4),
                    "dist_m":    round(float(g.shot_dist.item()), 2),
                    "angle_rad": round(float(g.shot_angle.item()), 4),
                    "is_header": int(g.is_header.item()),
                    "technique": TECHNIQUE_NAMES.get(tech_idx, "Normal"),
                })
            csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download shots CSV",
                data=csv_bytes,
                file_name=f"shots_{comp_key}_{sm_team.replace(' ','_')}.csv",
                mime="text/csv",
                key="shotmap_csv_dl",
            )

    with col_r:
        sm_labels = labels[sm_mask]
        # ── Team KPI card (shown only when a specific team is selected) ──────
        if sm_team != "All teams" and len(sm_graphs) > 0:
            conv_rate = sm_g / sm_n * 100 if sm_n else 0
            xg_total  = float(sm_probs.sum())
            xg_sb     = float(sum(g.sb_xg.item() for g in sm_graphs))
            avg_dist  = float(np.mean([g.shot_dist.item() for g in sm_graphs]))
            xg_diff   = sm_g - xg_total
            diff_col  = "#66BB6A" if xg_diff >= 0 else "#EF5350"
            diff_lbl  = f"+{xg_diff:.2f}" if xg_diff >= 0 else f"{xg_diff:.2f}"
            st.markdown(
                f"<div style='background:#1a1a2e;border-radius:8px;padding:12px 14px;"
                f"border-left:4px solid #4FC3F7;margin-bottom:10px;font-size:12px;color:#ccc'>"
                f"<b style='color:#e0e0e0;font-size:13px'>{sm_team}</b><br>"
                f"<span style='color:#888'>Shots</span> <b style='color:#e0e0e0'>{sm_n}</b> &nbsp;·&nbsp; "
                f"<span style='color:#888'>Goals</span> <b style='color:#66BB6A'>{sm_g}</b> &nbsp;·&nbsp; "
                f"<span style='color:#888'>Conv.</span> <b style='color:#e0e0e0'>{conv_rate:.1f}%</b><br>"
                f"<span style='color:#888'>xG model</span> <b style='color:#4FC3F7'>{xg_total:.2f}</b> &nbsp;·&nbsp; "
                f"<span style='color:#888'>xG SB</span> <b style='color:#FFD54F'>{xg_sb:.2f}</b><br>"
                f"<span style='color:#888'>Goals − xG</span> <b style='color:{diff_col}'>{diff_lbl}</b> &nbsp;·&nbsp; "
                f"<span style='color:#888'>Avg dist</span> <b style='color:#e0e0e0'>{avg_dist:.1f} m</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin:6px 0;border-top:1px solid #333'></div>",
                        unsafe_allow_html=True)

        # ── Top 10 highest xG shots ───────────────────────────────────────────
        st.markdown(
            "<p style='font-size:13px;font-weight:700;color:#e0e0e0;"
            "margin-bottom:6px'>🎯 Top 10 highest xG shots</p>",
            unsafe_allow_html=True,
        )
        top_idx = np.argsort(sm_probs)[::-1][:10]
        for rank, i in enumerate(top_idx):
            g     = sm_graphs[i]
            icon  = "⚽" if sm_labels[i] else "<span style='color:#EF5350'>✗</span>"
            dist  = g.shot_dist.item()
            xg    = sm_probs[i]
            # Show team name when in "All teams" mode
            team_tag = (
                f"<span style='font-size:9px;color:#666;min-width:60px;overflow:hidden;"
                f"text-overflow:ellipsis;white-space:nowrap'>{_html.escape(g.team_name or '')}</span>"
                if sm_team == "All teams" else ""
            )
            bar_w = int(xg * 60)
            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:5px;gap:6px'>"
                f"<span style='color:#888;font-size:10px;width:18px'>#{rank+1}</span>"
                f"<span style='font-size:13px'>{icon}</span>"
                f"{team_tag}"
                f"<div style='flex:1;background:#1a1a2e;border-radius:3px;height:14px'>"
                f"<div style='width:{bar_w}px;background:#4FC3F7;border-radius:3px;height:14px'></div></div>"
                f"<span style='font-size:12px;font-weight:700;color:#e0e0e0;min-width:36px'>{xg:.3f}</span>"
                f"<span style='font-size:10px;color:#888;min-width:30px'>{dist:.0f}m</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Surprising goals (only when goals visible) ────────────────────────
        goal_idxs_sm = np.where(sm_labels == 1)[0]
        if len(goal_idxs_sm) > 0:
            st.markdown("<div style='margin:10px 0;border-top:1px solid #333'></div>",
                        unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:13px;font-weight:700;color:#e0e0e0;"
                "margin-bottom:4px'>😮 Most surprising goals</p>"
                "<p style='font-size:10px;color:#888;margin-bottom:6px'>Goals model rated lowest</p>",
                unsafe_allow_html=True,
            )
            surprise = goal_idxs_sm[np.argsort(sm_probs[goal_idxs_sm])[:8]]
            for i in surprise:
                g    = sm_graphs[i]
                dist = g.shot_dist.item()
                xg   = sm_probs[i]
                team_tag = (
                    f"<span style='font-size:9px;color:#666;min-width:60px;overflow:hidden;"
                    f"text-overflow:ellipsis;white-space:nowrap'>{_html.escape(g.team_name or '')}</span>"
                    if sm_team == "All teams" else ""
                )
                st.markdown(
                    f"<div style='display:flex;align-items:center;margin-bottom:5px;gap:6px'>"
                    f"<span style='font-size:13px'>⚽</span>"
                    f"{team_tag}"
                    f"<div style='flex:1;background:#1a1a2e;border-radius:3px;height:14px'>"
                    f"<div style='width:{int(xg*60)}px;background:#EF5350;border-radius:3px;height:14px'></div></div>"
                    f"<span style='font-size:12px;font-weight:700;color:#e0e0e0;min-width:36px'>{xg:.3f}</span>"
                    f"<span style='font-size:10px;color:#888;min-width:30px'>{dist:.0f}m</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


elif view == "🔬 Shot Inspector":
    st.markdown("### Shot Inspector — Freeze Frame")

    # Filters
    fil1, fil2, fil3 = st.columns([1, 1, 2])
    with fil1:
        outcome_filter = st.selectbox("Outcome", ["All", "Goals only", "Misses only"])
    with fil2:
        type_filter = st.selectbox("Type", ["All", "Open play", "Set piece", "Headers"])

    # Build filtered index
    mask = np.ones(len(graphs), dtype=bool)
    if outcome_filter == "Goals only":
        mask &= labels == 1
    elif outcome_filter == "Misses only":
        mask &= labels == 0
    if type_filter == "Open play":
        mask &= np.array([bool(g.is_open_play.item()) for g in graphs])
    elif type_filter == "Set piece":
        mask &= ~np.array([bool(g.is_open_play.item()) for g in graphs])
    elif type_filter == "Headers":
        mask &= np.array([bool(g.is_header.item()) for g in graphs])

    filtered_idx = np.where(mask)[0]

    if len(filtered_idx) == 0:
        st.warning("No shots match the current filters.")
    else:
        with fil3:
            shot_rank = st.slider(
                f"Shot (1 – {len(filtered_idx)})",
                min_value=1, max_value=len(filtered_idx), value=1,
            )

        selected_idx = filtered_idx[shot_rank - 1]
        graph        = graphs[selected_idx]
        hybrid_xg    = float(hybrid_probs[selected_idx])

        gat_model, gat_available = load_gat_model()
        overlay_options = ["Gradient Saliency", "GAT Attention (top-3 pairs)", "None"]
        if not gat_available:
            overlay_options = ["Gradient Saliency", "None",
                               "GAT Attention — train HybridGAT first"]

        overlay = st.sidebar.radio(
            "Overlay",
            options=overlay_options[:3] if gat_available else ["Gradient Saliency", "None"],
            help=(
                "Gradient Saliency: which players most affect the GCN's xG prediction "
                "(backprop through node features).\n\n"
                "GAT Attention: which player pairs the GATv2 model attends to "
                "during message passing (learned α weights)."
            ),
        )

        show_sal  = overlay == "Gradient Saliency"
        show_attn = overlay == "GAT Attention (top-3 pairs)" and gat_available

        node_imp   = compute_node_saliency(load_model().model, graph) if show_sal else None
        attn_data  = compute_gat_attention(gat_model, graph) if show_attn else None

        fig = freeze_frame_figure(
            graph, hybrid_xg, shot_rank - 1, len(filtered_idx),
            node_importance=node_imp, show_saliency=show_sal,
            attn_data=attn_data,     show_attention=show_attn,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── GAT attention table ──────────────────────────────────────────
        if show_attn and attn_data is not None:
            edge_index_np, alpha_E = attn_data
            top_edges = np.argsort(alpha_E)[::-1][:3]
            st.markdown("**Top-3 attention pairs**")
            rows = []
            for rank, eidx in enumerate(top_edges):
                src_i = int(edge_index_np[0][eidx])
                dst_i = int(edge_index_np[1][eidx])
                nodes_np = graph.x.numpy()
                def _role(ni):
                    if nodes_np[ni, 3]:  return "🌟 Shooter"
                    if nodes_np[ni, 4]:  return "🧤 Keeper"
                    if nodes_np[ni, 2]:  return "🟢 Attacker"
                    return "🔴 Defender"
                rows.append({
                    "Rank":   f"#{rank+1}",
                    "From":   _role(src_i),
                    "To":     _role(dst_i),
                    "α weight": f"{alpha_E[eidx]:.4f}",
                    "Bar":    "█" * int(alpha_E[eidx] / alpha_E[top_edges[0]] * 12),
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).set_index("Rank"),
                         use_container_width=True)

        # Quick nav
        nc1, nc2 = st.columns(2)
        with nc1:
            if st.button("⬅ Previous") and shot_rank > 1:
                st.session_state["_rank"] = shot_rank - 1
        with nc2:
            if st.button("Next ➡") and shot_rank < len(filtered_idx):
                st.session_state["_rank"] = shot_rank + 1


elif view == "📋 Match Report":
    import pandas as pd

    match_index = build_match_index(comp_key)
    if not match_index:
        st.warning("No match data found. Ensure graphs were built with the latest build_shot_graphs.py.")
    else:
        with _match_index_placeholder:
            match_id_sel = st.selectbox(
                "Match",
                options=list(match_index.keys()),
                format_func=lambda mid: match_index[mid]["label"],
                key="match_selector",
            )
        entry     = match_index[match_id_sel]
        home_team = entry["home_team"]
        away_team = entry["away_team"]
        mg        = entry["graphs"]

        st.markdown(f"### {home_team or 'Home'} vs {away_team or 'Away'}")

        with st.spinner("Generating match report…"):
            m_probs = get_match_predictions(mg)

        # ── Visual 4-panel ────────────────────────────────────────────────────
        fig = match_report_figure(mg, m_probs, home_team, away_team)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")

        # ── Analyst narrative section ─────────────────────────────────────────
        home_mask = np.array([g.team_name == home_team for g in mg])
        away_mask = ~home_mask

        def _team_stats(mask):
            gs   = [mg[i] for i in range(len(mg)) if mask[i]]
            prbs = m_probs[mask]
            goals = [g for g in gs if int(g.y.item()) == 1]
            return {
                "shots":    len(gs),
                "goals":    len(goals),
                "xG":       float(prbs.sum()),
                "xG_sb":    float(sum(g.sb_xg.item() for g in gs)),
                "on_target": sum(1 for g in gs if int(g.y.item()) == 1
                                 or g.sb_xg.item() > 0.05),
                "avg_dist": float(np.mean([g.shot_dist.item() for g in gs])) if gs else 0,
                "avg_gk_d": float(np.mean([g.gk_dist.item() for g in gs
                                            if hasattr(g, "gk_dist")])) if gs else float("nan"),
                "surprise_goals": [
                    (g, p) for g, p in zip(gs, prbs)
                    if int(g.y.item()) == 1 and p < SURPRISE_XG_THRESHOLD
                ],
                "graphs":   gs,
                "probs":    prbs,
            }

        hs = _team_stats(home_mask)
        as_ = _team_stats(away_mask)

        # Determine performance narrative
        def _perf_narrative(st_dict):
            diff = st_dict["goals"] - st_dict["xG"]
            if diff > 0.5:   return "over-performed their xG"
            if diff < -0.5:  return "under-performed their xG"
            return "finished in line with expected goals"

        # Shot quality: high-quality = xG > 0.2
        def _hq_shots(st_dict):
            return sum(1 for g, p in zip(st_dict["graphs"], st_dict["probs"]) if p > 0.20)

        st.markdown("#### 📋 Analyst Summary")

        # Executive summary box
        total_xg_home = hs["xG"]
        total_xg_away = as_["xG"]
        xg_winner     = _html.escape(home_team if total_xg_home > total_xg_away else away_team)
        xg_loser      = _html.escape(away_team if total_xg_home > total_xg_away else home_team)
        _home_esc     = _html.escape(home_team or "Home")
        _away_esc     = _html.escape(away_team or "Away")

        _surprise_line = ""
        if hs['surprise_goals'] or as_['surprise_goals']:
            names = ", ".join(
                f"{_html.escape(sg[0].player_name or 'Unknown')} ({sg[1]:.1%} xG)"
                for sg in hs['surprise_goals'] + as_['surprise_goals']
            )
            _surprise_line = f"<br><br>⚡ <b style='color:#FFD54F'>Surprise goal(s) flagged</b> — {names}."

        st.markdown(f"""
<div style='background:#1a1a2e;border-radius:8px;padding:16px 20px;
            border-left:4px solid #4FC3F7;margin-bottom:16px;font-size:13px;color:#ccc;
            line-height:1.7'>
<b style='color:#e0e0e0;font-size:15px'>Executive Summary</b><br><br>
{xg_winner} dominated the xG battle ({max(total_xg_home, total_xg_away):.2f}) vs
{xg_loser} ({min(total_xg_home, total_xg_away):.2f}).
<b style='color:#4FC3F7'>{_home_esc}</b> took {hs['shots']} shots
(avg distance {hs['avg_dist']:.1f} m) and {_perf_narrative(hs)}.
<b style='color:#EF5350'>{_away_esc}</b> took {as_['shots']} shots
(avg distance {as_['avg_dist']:.1f} m) and {_perf_narrative(as_)}.
{_surprise_line}
</div>
""", unsafe_allow_html=True)

        # Side-by-side team stats
        mc1, mc2 = st.columns(2)
        for col, team, st_d, colour in [
            (mc1, home_team or "Home", hs, "#4FC3F7"),
            (mc2, away_team or "Away", as_, "#EF5350"),
        ]:
            with col:
                st.markdown(
                    f"<div style='font-size:14px;font-weight:700;color:{colour};"
                    f"margin-bottom:8px'>{_html.escape(team)}</div>",
                    unsafe_allow_html=True,
                )
                rows = [
                    ("Shots",               st_d["shots"]),
                    ("Goals",               st_d["goals"]),
                    ("xG (model)",          f"{st_d['xG']:.2f}"),
                    ("xG (StatsBomb)",      f"{st_d['xG_sb']:.2f}"),
                    ("High-quality shots",  _hq_shots(st_d)),
                    ("Avg shot distance",   f"{st_d['avg_dist']:.1f} m"),
                    ("Avg GK distance",     f"{st_d['avg_gk_d']:.1f} m" if not np.isnan(st_d['avg_gk_d']) else "—"),
                    ("Surprise goals",      len(st_d["surprise_goals"])),
                ]
                df_team = pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")
                st.dataframe(df_team, use_container_width=True)

        # Shot-by-shot log
        st.markdown("#### 📄 Shot-by-Shot Log")
        shot_rows = []
        for i, (g, p) in enumerate(zip(mg, m_probs)):
            tech_idx = int(g.technique.argmax().item())
            gk_d = g.gk_dist.item() if hasattr(g, "gk_dist") else float("nan")
            n_def = int(g.n_def_in_cone.item()) if hasattr(g, "n_def_in_cone") else 0
            surprise_flag = "⭐" if (int(g.y.item()) == 1 and p < SURPRISE_XG_THRESHOLD) else ""
            shot_rows.append({
                "Min":        int(g.minute.item()),
                "Team":       g.team_name or "—",
                "Player":     g.player_name or "—",
                "Outcome":    "⚽ Goal" if int(g.y.item()) == 1 else "✗ Miss",
                "xG (model)": f"{p:.3f}",
                "xG (SB)":    f"{g.sb_xg.item():.3f}",
                "Dist (m)":   f"{g.shot_dist.item():.1f}",
                "Technique":  TECHNIQUE_NAMES.get(tech_idx, "Normal"),
                "GK dist":    f"{gk_d:.1f}" if not np.isnan(gk_d) else "—",
                "Def in cone": n_def,
                "Flag":       surprise_flag,
            })
        shot_rows.sort(key=lambda r: r["Min"])
        st.dataframe(
            pd.DataFrame(shot_rows),
            use_container_width=True,
            height=min(40 + 35 * len(shot_rows), 500),
        )

elif view == "📊 xG Distributions":
    st.markdown("### xG Distributions — Goals vs Misses")
    fig = xg_distribution_figure(graphs, hybrid_probs)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.markdown("#### 🌡️ Calibration — Reliability Diagram")
    st.caption(
        "Left: predicted xG buckets vs actual conversion rate — a well-calibrated model "
        "hugs the diagonal. Right: Brier score (lower = better)."
    )
    fig_cal = reliability_diagram_figure(graphs, hybrid_probs)
    st.pyplot(fig_cal, use_container_width=True)
    plt.close(fig_cal)

    st.markdown("---")
    # Summary stats table
    st.markdown("#### Model comparison on this competition")
    from sklearn.metrics import average_precision_score, brier_score_loss

    rows = []
    for name, probs in [("HybridGCN", hybrid_probs), ("StatsBomb xG", sb_xgs)]:
        auc_v = roc_auc_score(labels, probs)
        ap_v  = average_precision_score(labels, probs)
        br_v  = brier_score_loss(labels, probs)
        rows.append({"Model": name, "AUC": f"{auc_v:.3f}", "Avg Precision": f"{ap_v:.3f}", "Brier": f"{br_v:.3f}"})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("---")
    # Read the current temperature T from the loaded scaler
    _scaler = load_model()
    _T = _scaler.temperature if isinstance(_scaler, TemperatureScaler) else 1.0
    _calibrated = _T != 1.0

    if _calibrated:
        st.success(
            f"✅ **Temperature scaling applied** (T = {_T:.3f}).  "
            f"Raw logits divided by T before sigmoid — over-confidence above 20% xG is suppressed.  \n"
            f"Goal mean xG: model={hybrid_probs[labels==1].mean():.3f}  ·  "
            f"StatsBomb={sb_xgs[labels==1].mean():.3f}"
        )
    else:
        st.info(
            "**Note on calibration:** No temperature scalar found. Run `train_xg_hybrid.py` "
            "to generate `pool_7comp_T.pt` and fix over-confidence.  \n"
            f"Goal mean xG: model={hybrid_probs[labels==1].mean():.3f}  ·  "
            f"StatsBomb={sb_xgs[labels==1].mean():.3f}"
        )


# ── 🌟 Surprise Goals ─────────────────────────────────────────────────────────
elif view == "🌟 Surprise Goals":
    import pandas as pd

    st.markdown("### 🌟 Surprise Goals — Worldies the Model Didn't See Coming")
    st.caption(
        f"Goals where HybridGCN xG < {SURPRISE_XG_THRESHOLD:.0%} — "
        "long-range efforts, deflections, and moments of individual brilliance "
        "that the model rated as unlikely."
    )

    # ── Find surprise goals ────────────────────────────────────────────────────
    goal_mask     = labels == 1
    surprise_mask = goal_mask & (hybrid_probs < SURPRISE_XG_THRESHOLD)
    surprise_idx  = np.where(surprise_mask)[0]

    if len(surprise_idx) == 0:
        st.info(f"No goals below {SURPRISE_XG_THRESHOLD:.0%} xG in this competition. "
                "Try a different competition or lower the threshold.")
    else:
        surprise_idx_sorted = surprise_idx[np.argsort(hybrid_probs[surprise_idx])]

        # ── Summary KPIs ──────────────────────────────────────────────────────
        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            st.markdown(f"""<div class="metric-card red">
                <div class="metric-label">Surprise Goals</div>
                <div class="metric-value">{len(surprise_idx)}</div>
                <div class="metric-sub">of {n_goals} total goals ({100*len(surprise_idx)/max(n_goals,1):.0f}%)</div>
            </div>""", unsafe_allow_html=True)
        with kc2:
            avg_xg = hybrid_probs[surprise_idx].mean()
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Avg model xG</div>
                <div class="metric-value">{avg_xg:.3f}</div>
                <div class="metric-sub">SB avg: {sb_xgs[surprise_idx].mean():.3f}</div>
            </div>""", unsafe_allow_html=True)
        with kc3:
            avg_dist = np.array([graphs[i].shot_dist.item() for i in surprise_idx]).mean()
            st.markdown(f"""<div class="metric-card gold">
                <div class="metric-label">Avg shot distance</div>
                <div class="metric-value">{avg_dist:.1f} m</div>
                <div class="metric-sub">vs overall avg {np.array([g.shot_dist.item() for g in graphs]).mean():.1f} m</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        col_pitch, col_table = st.columns([1.4, 1])

        # ── Pitch map ─────────────────────────────────────────────────────────
        with col_pitch:
            fig_s, ax_s = plt.subplots(figsize=(8, 7), facecolor=DARK_BG)
            pitch_s = VerticalPitch(
                pitch_type="custom", pitch_length=PL, pitch_width=PW,
                pitch_color=PANEL_BG, line_color="#555", half=True,
                goal_type="box", pad_top=2, pad_bottom=2,
            )
            pitch_s.draw(ax=ax_s)

            s_probs = hybrid_probs[surprise_idx_sorted]
            s_x = np.array([graphs[i].x[graphs[i].x[:, 3] == 1][0, 0].item()
                             for i in surprise_idx_sorted])
            s_y = np.array([graphs[i].x[graphs[i].x[:, 3] == 1][0, 1].item()
                             for i in surprise_idx_sorted])

            sc = pitch_s.scatter(s_x, s_y, ax=ax_s,
                                 c=s_probs, cmap="plasma_r", vmin=0, vmax=SURPRISE_XG_THRESHOLD,
                                 s=250, alpha=0.9, zorder=5, marker="*",
                                 edgecolors="white", linewidths=0.8)

            # Label each with rank
            for rank, (xi, yi) in enumerate(zip(s_x, s_y)):
                ax_s.text(xi + 1.0, yi, f"#{rank+1}",
                          fontsize=6.5, color="#FFD54F", alpha=0.9,
                          fontweight="bold", zorder=7)

            cbar = fig_s.colorbar(sc, ax=ax_s, fraction=0.03, pad=0.02)
            cbar.set_label("Model xG (lower = more surprising)", color=TEXT_COLOR, fontsize=8)
            cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=7)
            ax_s.set_title(
                f"Surprise Goals (xG < {SURPRISE_XG_THRESHOLD:.0%})  ·  "
                f"{COMPETITIONS[comp_key]}",
                color="#FFD54F", fontsize=10, fontweight="bold",
            )
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)

        # ── Ranked table ──────────────────────────────────────────────────────
        with col_table:
            st.markdown(
                "<p style='font-size:13px;font-weight:700;color:#FFD54F;"
                "margin-bottom:6px'>Ranked by model xG ↑ (most surprising first)</p>",
                unsafe_allow_html=True,
            )
            rows = []
            for rank, i in enumerate(surprise_idx_sorted):
                g = graphs[i]
                tech_idx  = int(g.technique.argmax().item())
                tech_name = TECHNIQUE_NAMES.get(tech_idx, "Normal")
                gk_d      = g.gk_dist.item() if hasattr(g, "gk_dist") else float("nan")
                rows.append({
                    "Rank":        f"#{rank+1}",
                    "Player":      g.player_name or "Unknown",
                    "Team":        g.team_name   or "—",
                    "Min":         int(g.minute.item()),
                    "xG (model)":  f"{hybrid_probs[i]:.3f}",
                    "xG (SB)":     f"{sb_xgs[i]:.3f}",
                    "Dist (m)":    f"{g.shot_dist.item():.1f}",
                    "Technique":   tech_name,
                    "GK dist (m)": f"{gk_d:.1f}" if not np.isnan(gk_d) else "—",
                })
            df = pd.DataFrame(rows).set_index("Rank")
            st.dataframe(df, use_container_width=True, height=min(40 + 35 * len(rows), 600))

            # Analyst note
            if len(surprise_idx_sorted) > 0:
                most_surprising = graphs[surprise_idx_sorted[0]]
                _ms_player = _html.escape(most_surprising.player_name or "Unknown")
                _ms_team   = _html.escape(most_surprising.team_name or "—")
                st.markdown(
                    f"<div style='margin-top:14px;padding:10px 12px;"
                    f"background:#1a1a2e;border-left:3px solid #FFD54F;"
                    f"border-radius:4px;font-size:12px;color:#ccc'>"
                    f"<b style='color:#FFD54F'>⚡ Most surprising:</b> "
                    f"{_ms_player} "
                    f"({_ms_team}) "
                    f"at min. {most_surprising.minute.item()!s} — "
                    f"model gave only <b style='color:#FFD54F'>"
                    f"{hybrid_probs[surprise_idx_sorted[0]]:.1%}</b> chance of scoring "
                    f"from {most_surprising.shot_dist.item():.1f} m."
                    f"</div>",
                    unsafe_allow_html=True,
                )


elif view == "👤 Player Profile":
    import pandas as pd

    st.markdown("### 👤 Player xG Profile — Individual Shot Quality")
    st.caption(
        "Per-player aggregated stats across all shots in this competition. "
        "xG overperformance = Goals − model xG (positive = beating the model)."
    )

    # ── Build per-player stats ─────────────────────────────────────────────────
    player_stats: dict[str, dict] = {}
    for g, p in zip(graphs, hybrid_probs):
        name = g.player_name or "Unknown"
        team = g.team_name   or "—"
        if name not in player_stats:
            player_stats[name] = {
                "team": team, "shots": 0, "goals": 0,
                "xG_model": 0.0, "xG_sb": 0.0, "dist_sum": 0.0,
            }
        s = player_stats[name]
        s["shots"]    += 1
        s["goals"]    += int(g.y.item())
        s["xG_model"] += float(p)
        s["xG_sb"]    += float(g.sb_xg.item())
        s["dist_sum"] += float(g.shot_dist.item())

    rows = []
    for name, s in player_stats.items():
        if s["shots"] == 0:
            continue
        overperf = s["goals"] - s["xG_model"]
        rows.append({
            "Player":           name,
            "Team":             s["team"],
            "Shots":            s["shots"],
            "Goals":            s["goals"],
            "xG (model)":       round(s["xG_model"], 2),
            "xG (SB)":          round(s["xG_sb"],    2),
            "Goals − xG":       round(overperf,       2),
            "Avg dist (m)":     round(s["dist_sum"] / s["shots"], 1),
            "Conv %":           round(100 * s["goals"] / s["shots"], 1),
        })

    df_players = pd.DataFrame(rows)
    if df_players.empty:
        st.info("No player data available for this competition.")
    else:
        # ── Sort & filter controls ─────────────────────────────────────────────
        pc1, pc2, pc3 = st.columns([1, 1, 1])
        with pc1:
            sort_col = st.selectbox(
                "Sort by",
                ["Goals − xG", "xG (model)", "Goals", "Shots", "Avg dist (m)", "Conv %"],
                key="player_sort",
            )
        with pc2:
            min_shots = st.slider("Min shots", 1, 10, 3, key="player_min_shots")
        with pc3:
            team_filter = st.selectbox(
                "Filter team",
                ["All teams"] + sorted(df_players["Team"].unique().tolist()),
                key="player_team_filter",
            )

        df_show = df_players[df_players["Shots"] >= min_shots].copy()
        if team_filter != "All teams":
            df_show = df_show[df_show["Team"] == team_filter]
        df_show = df_show.sort_values(sort_col, ascending=False).reset_index(drop=True)

        if df_show.empty:
            st.warning("No players match the current filters.")
        else:
            # ── Summary KPIs ───────────────────────────────────────────────────
            pk1, pk2, pk3 = st.columns(3)
            with pk1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Players shown</div>
                    <div class="metric-value">{len(df_show)}</div>
                </div>""", unsafe_allow_html=True)
            with pk2:
                top_scorer = df_show.iloc[0]
                st.markdown(f"""<div class="metric-card green">
                    <div class="metric-label">Top by {_html.escape(str(sort_col))}</div>
                    <div class="metric-value" style="font-size:14px">{_html.escape(str(top_scorer['Player']))}</div>
                    <div class="metric-sub">{top_scorer[sort_col]}</div>
                </div>""", unsafe_allow_html=True)
            with pk3:
                over_count = int((df_show["Goals − xG"] > 0).sum())
                st.markdown(f"""<div class="metric-card gold">
                    <div class="metric-label">Overperforming model</div>
                    <div class="metric-value">{over_count}</div>
                    <div class="metric-sub">of {len(df_show)} players</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            col_chart, col_table = st.columns([1.2, 1])

            # ── Scatter: xG vs Goals, coloured by overperformance ─────────────
            with col_chart:
                top_n = min(20, len(df_show))
                df_plot = df_show.head(top_n)

                fig_p, ax_p = plt.subplots(figsize=(7, 5.5), facecolor=DARK_BG)
                ax_p.set_facecolor(PANEL_BG)

                overperf_vals = df_plot["Goals − xG"].values
                norm_op = plt.Normalize(overperf_vals.min(), overperf_vals.max())
                colors_p = plt.cm.RdYlGn(norm_op(overperf_vals))

                ax_p.scatter(df_plot["xG (model)"], df_plot["Goals"],
                             c=colors_p, s=110, alpha=0.85, zorder=4, edgecolors="#333", lw=0.5)

                # Diagonal = perfect calibration (Goals == xG)
                max_val = max(df_plot["xG (model)"].max(), df_plot["Goals"].max()) * 1.1
                ax_p.plot([0, max_val], [0, max_val], "--", color="#666", lw=1.2,
                          label="Goals = xG", zorder=2)

                # Label top-5 overperformers and top-5 underperformers
                label_mask = (df_plot["Goals − xG"].rank(ascending=False) <= 3) | \
                             (df_plot["Goals − xG"].rank(ascending=True)  <= 3)
                for _, row in df_plot[label_mask].iterrows():
                    ax_p.annotate(
                        row["Player"].split()[-1],  # surname only
                        (row["xG (model)"], row["Goals"]),
                        xytext=(5, 3), textcoords="offset points",
                        fontsize=6.5, color=TEXT_COLOR, alpha=0.85,
                        bbox=dict(boxstyle="round,pad=0.15", fc=DARK_BG, ec="none", alpha=0.6),
                    )

                ax_p.set(xlabel="Total xG (model)", ylabel="Goals scored",
                         title=f"Goals vs xG  (top {top_n} players by {sort_col})")
                ax_p.title.set_color(TEXT_COLOR)
                ax_p.tick_params(colors=TEXT_COLOR)
                ax_p.grid(alpha=0.2)
                ax_p.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COLOR,
                            edgecolor="#444")

                sm_p = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm_op)
                sm_p.set_array([])
                cbar_p = fig_p.colorbar(sm_p, ax=ax_p, fraction=0.03, pad=0.02)
                cbar_p.set_label("Goals − xG", color=TEXT_COLOR, fontsize=8)
                cbar_p.ax.yaxis.set_tick_params(color=TEXT_COLOR)
                plt.setp(cbar_p.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=7)

                plt.tight_layout()
                st.pyplot(fig_p, use_container_width=True)
                plt.close(fig_p)

            # ── Sortable data table with download ─────────────────────────────
            with col_table:
                st.markdown(
                    "<p style='font-size:13px;font-weight:700;color:#e0e0e0;"
                    "margin-bottom:6px'>Player stats table</p>",
                    unsafe_allow_html=True,
                )

                # Colour-code Goals − xG column by styling
                def _style_overperf(val):
                    try:
                        v = float(val)
                        if v > 0.5:   return "color: #66BB6A; font-weight: bold"
                        if v < -0.5:  return "color: #EF5350; font-weight: bold"
                        return "color: #e0e0e0"
                    except Exception:
                        return ""

                df_styled = df_show[["Player","Team","Shots","Goals",
                                     "xG (model)","Goals − xG","Avg dist (m)"]].style.applymap(
                    _style_overperf, subset=["Goals − xG"]
                )
                st.dataframe(df_styled, use_container_width=True,
                             height=min(40 + 35 * len(df_show), 550))

                # ── CSV download ───────────────────────────────────────────────
                csv_bytes = df_show.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download player stats CSV",
                    data=csv_bytes,
                    file_name=f"player_xg_{comp_key}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
