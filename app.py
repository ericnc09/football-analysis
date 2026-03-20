"""
Football GNN Analysis Dashboard
================================
Interactive Streamlit app for exploring HybridGCN xG predictions
vs StatsBomb's industry xG across 8,013 shots from 7 competitions.

Run:
    streamlit run app.py
"""

import sys
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
MODEL_PATH     = PROCESSED / "pool_7comp_hybrid_xg.pt"
TEMP_PATH      = PROCESSED / "pool_7comp_T.pt"          # GCN temperature scalar
GAT_MODEL_PATH = PROCESSED / "pool_7comp_hybrid_gat_xg.pt"
GAT_TEMP_PATH  = PROCESSED / "pool_7comp_gat_T.pt"      # GAT temperature scalar
META_DIM       = 12  # shot_dist, shot_angle, is_header, is_open_play + technique (8-dim one-hot)

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
    def __init__(self, in_channels=9, hidden_dim=64, meta_dim=META_DIM, dropout=0.3):
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
    model.eval()
    x = graph.x.clone().detach().float().requires_grad_(True)
    ei = graph.edge_index
    batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    base = torch.tensor([[
        float(graph.shot_dist.item()),
        float(graph.shot_angle.item()),
        float(graph.is_header.item()),
        float(graph.is_open_play.item()),
    ]])                                        # [1, 4]
    tech = graph.technique.unsqueeze(0)        # [1, 8]
    meta = torch.cat([base, tech], dim=1)      # [1, 12]

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

    inner.eval()
    x        = graph.x.clone().detach().float()
    ei       = graph.edge_index
    batch_v  = torch.zeros(graph.x.shape[0], dtype=torch.long)
    meta     = torch.cat([
        torch.tensor([[
            float(graph.shot_dist.item()),
            float(graph.shot_angle.item()),
            float(graph.is_header.item()),
            float(graph.is_open_play.item()),
        ]]),
        graph.technique.unsqueeze(0),   # [1, 8]
    ], dim=1)                           # [1, 12]

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
    """Load HybridGCN + temperature scalar. Returns TemperatureScaler-wrapped model."""
    base = HybridXGModel()
    base.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
    base.eval()
    # Wrap with temperature scaler (falls back to T=1.0 if no T file yet)
    scaler = TemperatureScaler.load(base, TEMP_PATH)
    scaler.eval()
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
        gat = HybridGATModel(node_in=node_in, edge_dim=edge_dim,
                             meta_dim=META_DIM, hidden=32, heads=4,
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
    path = PROCESSED / f"statsbomb_{key}_shot_graphs.pt"
    if not path.exists():
        return []
    return torch.load(path, weights_only=False)


def _build_meta(batch) -> torch.Tensor:
    """Build 12-dim metadata tensor for a PyG batch."""
    base = torch.stack([
        batch.shot_dist.squeeze(),
        batch.shot_angle.squeeze(),
        batch.is_header.squeeze().float(),
        batch.is_open_play.squeeze().float(),
    ], dim=1)                          # [n, 4]
    tech = batch.technique.view(-1, 8) # [n, 8]
    return torch.cat([base, tech], dim=1)  # [n, 12]


@st.cache_resource
def get_predictions(key: str):
    graphs = load_competition(key)
    if not graphs:
        return np.array([])
    model = load_model()   # TemperatureScaler-wrapped; forward() applies T
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            meta   = _build_meta(batch)
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
    model  = load_model()   # T-scaled
    loader = DataLoader(match_graphs, batch_size=len(match_graphs), shuffle=False)
    with torch.no_grad():
        batch  = next(iter(loader))
        meta   = _build_meta(batch)
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
    view = st.radio("View", ["📍 Shot Map", "🔬 Shot Inspector", "📊 xG Distributions", "📋 Match Report"])

    st.markdown("---")

    # Match selector — only shown in Match Report view
    match_id_sel = None
    if view == "📋 Match Report":
        st.markdown("**Match**")
        _match_index_placeholder = st.empty()  # filled after data loads

    st.markdown("---")
    st.markdown(
        "**Model:** HybridGCN (GCN + dist/angle/header/technique)  \n"
        "**Val AUC:** 0.790  ·  **Test AUC:** 0.751  \n"
        "**vs StatsBomb xG:** 0.773  \n"
        "**Data:** 326 matches · 7 competitions"
    )


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
    st.caption("★ = goal  ·  ● = miss  ·  colour = HybridGCN predicted xG  ·  green=high, red=low")

    col_l, col_r = st.columns([3, 1])
    with col_l:
        fig = shot_map_figure(graphs, hybrid_probs, title=COMPETITIONS[comp_key])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        st.markdown(
            "<p style='font-size:13px;font-weight:700;color:#e0e0e0;"
            "margin-bottom:6px'>🎯 Top 10 highest xG shots</p>",
            unsafe_allow_html=True,
        )
        top_idx = np.argsort(hybrid_probs)[::-1][:10]
        for rank, i in enumerate(top_idx):
            g    = graphs[i]
            icon = "⚽" if labels[i] else "<span style='color:#EF5350'>✗</span>"
            dist = g.shot_dist.item()
            xg   = hybrid_probs[i]
            bar_w = int(xg * 60)
            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:5px;gap:6px'>"
                f"<span style='color:#888;font-size:10px;width:18px'>#{rank+1}</span>"
                f"<span style='font-size:13px'>{icon}</span>"
                f"<div style='flex:1;background:#1a1a2e;border-radius:3px;height:14px'>"
                f"<div style='width:{bar_w}px;background:#4FC3F7;border-radius:3px;height:14px'></div></div>"
                f"<span style='font-size:12px;font-weight:700;color:#e0e0e0;min-width:36px'>{xg:.3f}</span>"
                f"<span style='font-size:10px;color:#888;min-width:30px'>{dist:.0f}m</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='margin:10px 0;border-top:1px solid #333'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:13px;font-weight:700;color:#e0e0e0;"
            "margin-bottom:4px'>😮 Most surprising goals</p>"
            "<p style='font-size:10px;color:#888;margin-bottom:6px'>Goals model rated lowest</p>",
            unsafe_allow_html=True,
        )
        goal_idxs = np.where(labels == 1)[0]
        surprise  = goal_idxs[np.argsort(hybrid_probs[goal_idxs])[:8]]
        for i in surprise:
            dist = graphs[i].shot_dist.item()
            xg   = hybrid_probs[i]
            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:5px;gap:6px'>"
                f"<span style='font-size:13px'>⚽</span>"
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
        entry        = match_index[match_id_sel]
        home_team    = entry["home_team"]
        away_team    = entry["away_team"]

        st.markdown(f"### {home_team or 'Home'} vs {away_team or 'Away'}")

        with st.spinner("Generating match report…"):
            m_probs = get_match_predictions(entry["graphs"])

        fig = match_report_figure(entry["graphs"], m_probs, home_team, away_team)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

elif view == "📊 xG Distributions":
    st.markdown("### xG Distributions — Goals vs Misses")
    fig = xg_distribution_figure(graphs, hybrid_probs)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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
