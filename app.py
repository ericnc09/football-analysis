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
PROCESSED  = REPO_ROOT / "data" / "processed"
MODEL_PATH = PROCESSED / "pool_7comp_hybrid_xg.pt"
META_DIM   = 4

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
    meta = torch.tensor([[
        float(graph.shot_dist.item()),
        float(graph.shot_angle.item()),
        float(graph.is_header.item()),
        float(graph.is_open_play.item()),
    ]])

    logit = model(x, ei, batch, meta)
    pred  = torch.sigmoid(logit).squeeze()
    pred.backward()

    importance = x.grad.abs().sum(dim=1).detach().numpy()
    # Normalise 0→1 for display
    rng = importance.max() - importance.min()
    if rng > 1e-8:
        importance = (importance - importance.min()) / rng
    return importance


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = HybridXGModel()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_competition(key: str):
    path = PROCESSED / f"statsbomb_{key}_shot_graphs.pt"
    if not path.exists():
        return []
    return torch.load(path, weights_only=False)


@st.cache_resource
def get_predictions(key: str):
    graphs = load_competition(key)
    if not graphs:
        return np.array([])
    model = load_model()
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            meta = torch.stack([
                batch.shot_dist.squeeze(),
                batch.shot_angle.squeeze(),
                batch.is_header.squeeze().float(),
                batch.is_open_play.squeeze().float(),
            ], dim=1)
            logits = model(batch.x, batch.edge_index, batch.batch, meta)
            probs.extend(torch.sigmoid(logits.squeeze()).tolist())
    return np.array(probs)


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
                         node_importance=None, show_saliency=True):
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

    ax.legend(fontsize=8, loc="upper left", facecolor=PANEL_BG,
              labelcolor=TEXT_COLOR, edgecolor="#444")

    # Saliency legend note
    if show_saliency and node_importance is not None:
        ax.text(0.01, 0.01,
                "── gradient saliency edges: thicker = higher model influence",
                transform=ax.transAxes, fontsize=6.5, color="#aaa",
                alpha=0.75, va="bottom")

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
    view = st.radio("View", ["📍 Shot Map", "🔬 Shot Inspector", "📊 xG Distributions"])

    st.markdown("---")
    st.markdown(
        "**Model:** HybridGCN (GCN encoder + shot metadata)  \n"
        "**AUC:** 0.752 pooled · 0.760 cross-gender  \n"
        "**vs StatsBomb xG:** 0.794  \n"
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

        show_sal = st.sidebar.checkbox("Show gradient saliency", value=True,
                                       help="Draws edges from shooter to the players "
                                            "that most influenced the xG prediction "
                                            "(gradient magnitude w.r.t. node features).")

        node_imp = compute_node_saliency(load_model(), graph) if show_sal else None
        fig = freeze_frame_figure(graph, hybrid_xg, shot_rank - 1, len(filtered_idx),
                                  node_importance=node_imp, show_saliency=show_sal)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Quick nav
        nc1, nc2 = st.columns(2)
        with nc1:
            if st.button("⬅ Previous") and shot_rank > 1:
                st.session_state["_rank"] = shot_rank - 1
        with nc2:
            if st.button("Next ➡") and shot_rank < len(filtered_idx):
                st.session_state["_rank"] = shot_rank + 1


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
    st.markdown(
        "**Note on calibration:** HybridGCN over-assigns probability to goals (mean xG for goals = "
        f"{hybrid_probs[labels==1].mean():.2f} vs StatsBomb's {sb_xgs[labels==1].mean():.2f}). "
        "This is a known limitation: without shot technique data, the model compensates with "
        "higher spatial confidence. Adding `shot_technique` as a feature is the next experiment."
    )
