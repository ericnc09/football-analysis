#!/usr/bin/env python3
"""
generate_match_report.py
------------------------
Generates a professional analyst-quality match report for a given match_id.
Outputs a self-contained HTML file styled like a real scouting report.

Usage
-----
    python scripts/generate_match_report.py --match_id 3869685 --competition wc2022
    # → reports/wc2022_3869685_Argentina_vs_France.html

    python scripts/generate_match_report.py --list wc2022
    # → lists all available matches with their IDs
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mplsoccer import VerticalPitch, Pitch
import base64, io, datetime

from src.calibration import TemperatureScaler
from src.features import TECHNIQUE_INDEX

PROCESSED   = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

META_DIM = 15  # updated after GK pressure rebuild
META_DIM_FALLBACK = 12   # old graphs without gk features

TECHNIQUE_NAMES = {v: k.title() for k, v in TECHNIQUE_INDEX.items()}
TECHNIQUE_NAMES[0] = "Normal"
SURPRISE_THRESHOLD = 0.15

DARK_BG   = "#0f0f1a"
PANEL_BG  = "#16213e"
TEXT_COL  = "#e0e0e0"
BLUE      = "#4FC3F7"
RED       = "#EF5350"
GREEN     = "#66BB6A"
GOLD      = "#FFD54F"

PL, PW = 105.0, 68.0
GOAL_X, GOAL_Y = PL, PW / 2


# ── Model ─────────────────────────────────────────────────────────────────────
# Canonical HybridXGModel lives in src/models/hybrid_gcn.py — imported here
# so the match-report script can never drift against app.py or the training
# script.
from src.models.hybrid_gcn import HybridXGModel  # noqa: E402


def _build_meta_single(g, meta_dim):
    """Build meta tensor for a single graph."""
    base = torch.tensor([[
        float(g.shot_dist.item()),
        float(g.shot_angle.item()),
        float(g.is_header.item()),
        float(g.is_open_play.item()),
    ]])
    tech = g.technique.unsqueeze(0)
    if meta_dim == 15:
        gk = torch.tensor([[
            float(g.gk_dist.item()) if hasattr(g, "gk_dist") else 20.0,
            float(g.n_def_in_cone.item()) if hasattr(g, "n_def_in_cone") else 0.0,
            float(g.gk_off_centre.item()) if hasattr(g, "gk_off_centre") else 0.0,
        ]])
        return torch.cat([base, tech, gk], dim=1)
    return torch.cat([base, tech], dim=1)


def load_model():
    """Load HybridXGModel wrapped in TemperatureScaler."""
    ckpt = torch.load(PROCESSED / "pool_7comp_hybrid_xg.pt",
                      weights_only=True, map_location="cpu")
    # Detect meta_dim from the head weight shape
    head_in = ckpt["head.0.weight"].shape[1]
    in_ch   = ckpt["convs.0.bias"].shape[0]  # hidden dim actually
    # Reconstruct: head.0 in = hidden_dim + meta_dim, hidden_dim = 64
    meta_dim_actual = head_in - 64
    model = HybridXGModel(in_channels=9, hidden_dim=64,
                          meta_dim=meta_dim_actual, dropout=0.3)
    model.load_state_dict(ckpt)
    model.eval()
    scaler = TemperatureScaler.load(model, PROCESSED / "pool_7comp_T.pt")
    scaler.eval()
    return scaler, meta_dim_actual


def run_inference(model, graphs, meta_dim):
    """Run inference on a list of shot graphs."""
    probs = []
    with torch.no_grad():
        for g in graphs:
            meta  = _build_meta_single(g, meta_dim)
            batch = torch.zeros(g.x.shape[0], dtype=torch.long)
            logit = model(g.x, g.edge_index, batch, meta)
            probs.append(float(torch.sigmoid(logit).item()))
    return np.array(probs)


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_shot_map(graphs, probs, team, colour):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=DARK_BG)
    pitch = VerticalPitch(
        pitch_type="custom", pitch_length=PL, pitch_width=PW,
        pitch_color=PANEL_BG, line_color="#555", half=True,
        goal_type="box", pad_top=2,
    )
    pitch.draw(ax=ax)
    goal_mask = np.array([int(g.y.item()) for g in graphs]) == 1
    sx = np.array([g.x[g.x[:, 3] == 1][0, 0].item() for g in graphs])
    sy = np.array([g.x[g.x[:, 3] == 1][0, 1].item() for g in graphs])
    if (~goal_mask).any():
        pitch.scatter(sx[~goal_mask], sy[~goal_mask], ax=ax,
                      c=probs[~goal_mask], cmap="RdYlGn", vmin=0, vmax=1,
                      s=60, alpha=0.65, zorder=3)
    if goal_mask.any():
        pitch.scatter(sx[goal_mask], sy[goal_mask], ax=ax,
                      c=probs[goal_mask], cmap="RdYlGn", vmin=0, vmax=1,
                      s=200, alpha=0.95, zorder=5, marker="*",
                      edgecolors="white", linewidths=0.8)
    ax.set_title(team, color=colour, fontsize=12, fontweight="bold")
    return _fig_to_b64(fig)


def make_timeline(graphs, probs, home_team, away_team):
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    home_mask = np.array([g.team_name == home_team for g in graphs])
    away_mask = ~home_mask

    def cum_series(mask):
        pairs = [(int(graphs[i].minute.item()), float(probs[i]))
                 for i in range(len(graphs)) if mask[i]]
        pairs.sort(key=lambda t: t[0])
        if not pairs:
            return [0], [0.0]
        mins, xgs = zip(*pairs)
        return [0] + list(mins), [0.0] + np.cumsum(xgs).tolist()

    hm, hc = cum_series(home_mask)
    am, ac = cum_series(away_mask)
    ax.step(hm, hc, where="post", color=BLUE, lw=2.5,
            label=f"{home_team}  ({hc[-1]:.2f} xG)")
    ax.step(am, ac, where="post", color=RED, lw=2.5,
            label=f"{away_team}  ({ac[-1]:.2f} xG)")

    for i, g in enumerate(graphs):
        if int(g.y.item()) == 1:
            col = BLUE if home_mask[i] else RED
            ax.axvline(int(g.minute.item()), color=col, ls="--", lw=0.8, alpha=0.55)
            ax.text(int(g.minute.item()) + 0.5, ax.get_ylim()[1] * 0.88,
                    "⚽", fontsize=8, color=col, alpha=0.85)

    ax.axhline(1.0, color="#444", ls=":", lw=0.8)
    ax.set(xlabel="Minute", ylabel="Cumulative xG",
           title="Cumulative xG Timeline", xlim=(0, 120))
    ax.title.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL); ax.yaxis.label.set_color(TEXT_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT_COL, edgecolor="#333")
    ax.grid(alpha=0.2)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    return _fig_to_b64(fig)


# ── HTML report builder ────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #0f0f1a;
          color: #e0e0e0; max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  h1   {{ font-size: 28px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }}
  h2   {{ font-size: 18px; font-weight: 700; color: #4FC3F7; margin: 28px 0 10px; }}
  h3   {{ font-size: 14px; font-weight: 700; color: #aaa; margin: 18px 0 6px; }}
  .header {{ border-bottom: 2px solid #4FC3F7; padding-bottom: 16px; margin-bottom: 24px; }}
  .subtitle {{ font-size: 13px; color: #888; margin-top: 4px; letter-spacing: 0.5px;
               text-transform: uppercase; }}
  .kpi-row {{ display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }}
  .kpi {{ background: #1a1a2e; border-radius: 8px; padding: 14px 18px;
          border-left: 3px solid #4FC3F7; flex: 1; min-width: 140px; }}
  .kpi.home {{ border-left-color: #4FC3F7; }}
  .kpi.away {{ border-left-color: #EF5350; }}
  .kpi.gold {{ border-left-color: #FFD54F; }}
  .kpi.green {{ border-left-color: #66BB6A; }}
  .kpi-label {{ font-size: 10px; color: #888; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 4px; }}
  .kpi-value {{ font-size: 24px; font-weight: 800; color: #fff; }}
  .kpi-sub   {{ font-size: 11px; color: #aaa; margin-top: 3px; }}
  .section {{ background: #1a1a2e; border-radius: 10px; padding: 20px 24px;
              margin: 16px 0; }}
  .exec-summary {{ border-left: 4px solid #4FC3F7; padding: 16px 20px;
                   background: #161a2e; border-radius: 6px; line-height: 1.8;
                   font-size: 14px; color: #ccc; margin: 16px 0; }}
  .exec-summary b {{ color: #fff; }}
  .maps {{ display: flex; gap: 16px; margin: 16px 0; }}
  .maps img {{ border-radius: 8px; width: 48%; }}
  .timeline img {{ border-radius: 8px; width: 100%; margin: 12px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  th {{ background: #0f0f1a; color: #888; font-weight: 600; text-align: left;
        padding: 8px 10px; border-bottom: 1px solid #333; text-transform: uppercase;
        font-size: 10px; letter-spacing: 0.8px; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1a1a2e; color: #ccc; }}
  tr:hover td {{ background: #1e2440; }}
  .goal-row td {{ color: #66BB6A; font-weight: 600; }}
  .surprise-row td {{ color: #FFD54F; font-weight: 700; }}
  .badge {{ display: inline-block; border-radius: 4px; padding: 2px 7px;
            font-size: 10px; font-weight: 700; }}
  .badge-home {{ background: #1a3050; color: #4FC3F7; }}
  .badge-away {{ background: #301a1a; color: #EF5350; }}
  .badge-goal {{ background: #1a3020; color: #66BB6A; }}
  .badge-surprise {{ background: #302010; color: #FFD54F; }}
  .team-stat-grid {{ display: flex; gap: 16px; }}
  .team-stat-col {{ flex: 1; }}
  .stat-row {{ display: flex; justify-content: space-between; padding: 6px 0;
               border-bottom: 1px solid #2a2a3e; font-size: 13px; }}
  .stat-label {{ color: #888; }}
  .stat-value {{ color: #e0e0e0; font-weight: 600; }}
  .surprise-box {{ background: #201a00; border: 1px solid #FFD54F33;
                   border-radius: 8px; padding: 14px 18px; margin: 10px 0; }}
  .surprise-box h3 {{ color: #FFD54F; }}
  .surprise-item {{ padding: 8px 0; border-bottom: 1px solid #2a2000;
                    font-size: 13px; }}
  .surprise-item:last-child {{ border-bottom: none; }}
  .footer {{ color: #444; font-size: 11px; text-align: center; margin-top: 32px;
             padding-top: 16px; border-top: 1px solid #1a1a2e; }}
  .model-note {{ color: #555; font-size: 11px; font-style: italic; margin-top: 6px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def build_html_report(match_id, graphs, probs, home_team, away_team, comp_label):
    home_mask = np.array([g.team_name == home_team for g in graphs])
    away_mask = ~home_mask

    def team_data(mask):
        gs    = [graphs[i] for i in range(len(graphs)) if mask[i]]
        ps    = probs[mask]
        goals = [(g, p) for g, p in zip(gs, ps) if int(g.y.item()) == 1]
        surp  = [(g, p) for g, p in goals if p < SURPRISE_THRESHOLD]
        gk_ds = [g.gk_dist.item() for g in gs if hasattr(g, "gk_dist")]
        n_def = [g.n_def_in_cone.item() for g in gs if hasattr(g, "n_def_in_cone")]
        return {
            "gs": gs, "ps": ps, "goals": goals, "surprise": surp,
            "n_shots": len(gs), "n_goals": len(goals), "xG": float(ps.sum()),
            "xG_sb": float(sum(g.sb_xg.item() for g in gs)),
            "avg_dist": float(np.mean([g.shot_dist.item() for g in gs])) if gs else 0,
            "hq": sum(1 for p in ps if p > 0.20),
            "avg_gk_d": float(np.mean(gk_ds)) if gk_ds else float("nan"),
            "avg_def_cone": float(np.mean(n_def)) if n_def else 0.0,
        }

    hd = team_data(home_mask)
    ad = team_data(away_mask)

    # figures
    b64_home_map = make_shot_map(hd["gs"], hd["ps"], home_team, BLUE)
    b64_away_map = make_shot_map(ad["gs"], ad["ps"], away_team, RED)
    b64_timeline = make_timeline(graphs, probs, home_team, away_team)

    # Executive summary narrative
    xg_winner = home_team if hd["xG"] >= ad["xG"] else away_team
    xg_margin = abs(hd["xG"] - ad["xG"])
    h_result  = f"{hd['n_goals']}–{ad['n_goals']}"

    def perf_note(d, name):
        diff = d["n_goals"] - d["xG"]
        if diff >  0.4: return f"{name} over-performed their model xG by {diff:.2f}"
        if diff < -0.4: return f"{name} under-performed their model xG by {-diff:.2f}"
        return f"{name} finished in line with their model xG"

    def hq_note(d):
        pct = 100 * d["hq"] / max(d["n_shots"], 1)
        return f"{d['hq']} of {d['n_shots']} shots ({pct:.0f}%) rated above 20% xG"

    surprise_items_html = ""
    all_surprises = [(g, p, home_team) for g, p in hd["surprise"]] + \
                    [(g, p, away_team) for g, p in ad["surprise"]]
    all_surprises.sort(key=lambda x: x[1])

    for g, p, team in all_surprises:
        tech = TECHNIQUE_NAMES.get(int(g.technique.argmax().item()), "Normal")
        gk_d_str = f"{g.gk_dist.item():.1f} m" if hasattr(g, "gk_dist") else "—"
        surprise_items_html += f"""
        <div class="surprise-item">
          <span class="badge badge-surprise">⭐ {p:.1%} xG</span>
          &nbsp;<b style="color:#fff">{g.player_name or 'Unknown'}</b>
          <span style="color:#888">({team}) ·</span>
          min. <b>{g.minute.item()!s}</b> ·
          {g.shot_dist.item():.1f} m · {tech} · GK was {gk_d_str} away
          <div style="font-size:11px;color:#888;margin-top:3px">
            StatsBomb xG: {g.sb_xg.item():.3f} — model also flagged as low-probability
          </div>
        </div>"""

    if not surprise_items_html:
        surprise_items_html = "<div style='color:#666;font-size:13px'>No surprise goals in this match (all goals ≥ 15% xG).</div>"

    # Shot-by-shot table
    shot_rows_html = ""
    shots_sorted   = sorted(range(len(graphs)), key=lambda i: graphs[i].minute.item())
    for i in shots_sorted:
        g    = graphs[i]
        p    = probs[i]
        is_goal    = int(g.y.item()) == 1
        is_surprise = is_goal and p < SURPRISE_THRESHOLD
        is_home    = g.team_name == home_team
        row_class  = "surprise-row" if is_surprise else ("goal-row" if is_goal else "")
        outcome_badge = f'<span class="badge badge-{"surprise" if is_surprise else "goal"}">{"⭐ Goal" if is_surprise else ("⚽ Goal" if is_goal else "")}</span>' if is_goal else "✗"
        team_badge    = f'<span class="badge badge-{"home" if is_home else "away"}">{g.team_name or "—"}</span>'
        tech = TECHNIQUE_NAMES.get(int(g.technique.argmax().item()), "Normal")
        gk_d = f"{g.gk_dist.item():.1f}" if hasattr(g, "gk_dist") else "—"
        n_d  = int(g.n_def_in_cone.item()) if hasattr(g, "n_def_in_cone") else "—"
        shot_rows_html += f"""
        <tr class="{row_class}">
          <td>{int(g.minute.item())}'</td>
          <td>{team_badge}</td>
          <td>{g.player_name or "—"}</td>
          <td>{outcome_badge}</td>
          <td><b>{p:.3f}</b></td>
          <td>{g.sb_xg.item():.3f}</td>
          <td>{g.shot_dist.item():.1f}</td>
          <td>{tech}</td>
          <td>{gk_d}</td>
          <td>{n_d}</td>
        </tr>"""

    def stat_row(label, hval, aval):
        return f"""
        <div class="stat-row">
          <span class="stat-value" style="color:{BLUE}">{hval}</span>
          <span class="stat-label">{label}</span>
          <span class="stat-value" style="color:{RED}">{aval}</span>
        </div>"""

    gk_avg_h = f"{hd['avg_gk_d']:.1f} m" if not np.isnan(hd['avg_gk_d']) else "—"
    gk_avg_a = f"{ad['avg_gk_d']:.1f} m" if not np.isnan(ad['avg_gk_d']) else "—"

    body = f"""
<div class="header">
  <h1>⚽ Match Report</h1>
  <div class="subtitle">{comp_label} · Match ID {match_id} · Generated {datetime.date.today()}</div>
</div>

<h2 style="font-size:26px;color:#fff;text-align:center;margin:0 0 6px">
  <span style="color:{BLUE}">{home_team}</span>
  &nbsp;<span style="color:#555">vs</span>&nbsp;
  <span style="color:{RED}">{away_team}</span>
</h2>
<div style="text-align:center;color:#888;font-size:13px;margin-bottom:24px">
  Final score: <b style="color:#fff">{h_result}</b>
  &nbsp;·&nbsp; Total shots: {len(graphs)}
  &nbsp;·&nbsp; Total xG: {hd["xG"] + ad["xG"]:.2f}
</div>

<!-- KPI row -->
<div class="kpi-row">
  <div class="kpi home">
    <div class="kpi-label">xG (model)</div>
    <div class="kpi-value">{hd['xG']:.2f}</div>
    <div class="kpi-sub">SB: {hd['xG_sb']:.2f} · {home_team}</div>
  </div>
  <div class="kpi home">
    <div class="kpi-label">Shots / Goals</div>
    <div class="kpi-value">{hd['n_shots']} / {hd['n_goals']}</div>
    <div class="kpi-sub">{hd['hq']} high-quality (xG&gt;20%)</div>
  </div>
  <div class="kpi gold">
    <div class="kpi-label">xG Winner</div>
    <div class="kpi-value" style="font-size:16px">{xg_winner}</div>
    <div class="kpi-sub">by {xg_margin:.2f} xG</div>
  </div>
  <div class="kpi away">
    <div class="kpi-label">Shots / Goals</div>
    <div class="kpi-value">{ad['n_shots']} / {ad['n_goals']}</div>
    <div class="kpi-sub">{ad['hq']} high-quality (xG&gt;20%)</div>
  </div>
  <div class="kpi away">
    <div class="kpi-label">xG (model)</div>
    <div class="kpi-value">{ad['xG']:.2f}</div>
    <div class="kpi-sub">SB: {ad['xG_sb']:.2f} · {away_team}</div>
  </div>
</div>

<!-- Executive summary -->
<div class="exec-summary">
  <b>📋 Executive Summary</b><br><br>
  <b style="color:{BLUE}">{home_team}</b> and
  <b style="color:{RED}">{away_team}</b> produced a combined
  <b>{hd['xG'] + ad['xG']:.2f} xG</b> from <b>{len(graphs)} shots</b>.
  {xg_winner} dominated the xG battle ({max(hd['xG'], ad['xG']):.2f} vs {min(hd['xG'], ad['xG']):.2f}),
  suggesting the scoreline {"reflected" if abs(hd['n_goals'] - ad['n_goals']) <= 1 else "did not fully reflect"} the underlying shot quality.<br><br>
  {perf_note(hd, home_team)}. {perf_note(ad, away_team)}.<br><br>
  <b>Shot quality:</b> {home_team}: {hq_note(hd)}.
  {away_team}: {hq_note(ad)}.<br><br>
  <b>GK pressure:</b> {home_team} shots faced an average GK distance of
  <b>{gk_avg_h}</b> with <b>{hd['avg_def_cone']:.1f}</b> defenders per shot in the shooting cone.
  {away_team} faced <b>{gk_avg_a}</b> GK distance with
  <b>{ad['avg_def_cone']:.1f}</b> defenders per shot in the cone.
  {"Higher pressure correlates with lower xG — confirming the defensive effort was effective." if max(hd['avg_def_cone'], ad['avg_def_cone']) > 0.5 else "Both teams had relatively clear shooting lanes on average."}
  <div class="model-note">Model: HybridGCN (GCN + dist/angle/technique/GK pressure) · Temperature-calibrated · 7,946-shot training set</div>
</div>

<!-- Shot maps -->
<h2>Shot Maps</h2>
<div class="maps">
  <img src="data:image/png;base64,{b64_home_map}" alt="{home_team} shots"/>
  <img src="data:image/png;base64,{b64_away_map}" alt="{away_team} shots"/>
</div>

<!-- Timeline -->
<h2>Cumulative xG Timeline</h2>
<div class="timeline">
  <img src="data:image/png;base64,{b64_timeline}" alt="xG timeline"/>
</div>

<!-- Head-to-head comparison -->
<h2>Head-to-Head Stats</h2>
<div class="section" style="max-width:650px">
  <div style="display:flex;justify-content:space-between;margin-bottom:12px">
    <span style="color:{BLUE};font-weight:700">{home_team}</span>
    <span style="color:#555">Metric</span>
    <span style="color:{RED};font-weight:700">{away_team}</span>
  </div>
  {stat_row("Shots",           hd["n_shots"],                       ad["n_shots"])}
  {stat_row("Goals",           hd["n_goals"],                       ad["n_goals"])}
  {stat_row("xG (model)",      f"{hd['xG']:.2f}",                   f"{ad['xG']:.2f}")}
  {stat_row("xG (StatsBomb)",  f"{hd['xG_sb']:.2f}",               f"{ad['xG_sb']:.2f}")}
  {stat_row("High-qual shots", hd["hq"],                            ad["hq"])}
  {stat_row("Avg distance",    f"{hd['avg_dist']:.1f} m",           f"{ad['avg_dist']:.1f} m")}
  {stat_row("Avg GK dist",     gk_avg_h,                            gk_avg_a)}
  {stat_row("Avg def in cone", f"{hd['avg_def_cone']:.1f}",         f"{ad['avg_def_cone']:.1f}")}
  {stat_row("Surprise goals",  len(hd["surprise"]),                 len(ad["surprise"]))}
</div>

<!-- Surprise goals -->
<h2>⭐ Surprise Goals (xG &lt; {SURPRISE_THRESHOLD:.0%})</h2>
<div class="surprise-box">
  <h3>Goals the model rated as low-probability</h3>
  {surprise_items_html}
</div>

<!-- Shot log -->
<h2>Shot-by-Shot Log</h2>
<div class="section">
<table>
  <thead>
    <tr>
      <th>Min</th><th>Team</th><th>Player</th><th>Outcome</th>
      <th>xG (model)</th><th>xG (SB)</th><th>Dist</th>
      <th>Technique</th><th>GK dist</th><th>Def in cone</th>
    </tr>
  </thead>
  <tbody>
    {shot_rows_html}
  </tbody>
</table>
</div>

<div class="footer">
  Generated by Football GNN · HybridGCN xG Model · {datetime.date.today()}<br>
  <span style="color:#333">Training data: 7,946 shots · 7 StatsBomb 360 competitions · Val AUC 0.790</span>
</div>
"""
    return HTML_TEMPLATE.format(title=f"{home_team} vs {away_team} — xG Report", body=body)


# ── CLI ────────────────────────────────────────────────────────────────────────

def list_matches(comp_key):
    path = PROCESSED / f"statsbomb_{comp_key}_shot_graphs.pt"
    if not path.exists():
        print(f"No data found for {comp_key}. Run build_shot_graphs.py first.")
        return
    graphs = torch.load(path, weights_only=False)
    matches = defaultdict(lambda: {"teams": set(), "shots": 0, "goals": 0})
    for g in graphs:
        mid = int(g.match_id.item())
        matches[mid]["teams"].add(g.team_name)
        matches[mid]["shots"] += 1
        matches[mid]["goals"] += int(g.y.item())
    print(f"\n{'match_id':>10}  {'match':45}  shots  goals")
    print("-" * 75)
    for mid, info in sorted(matches.items()):
        teams = sorted(info["teams"])
        label = " vs ".join(teams) if len(teams) == 2 else str(teams)
        print(f"{mid:>10}  {label:45}  {info['shots']:5}  {info['goals']:5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_id",     type=int, default=None)
    parser.add_argument("--competition",  type=str, default="wc2022")
    parser.add_argument("--list",         type=str, default=None,
                        help="List all matches for a competition key")
    args = parser.parse_args()

    if args.list:
        list_matches(args.list)
        return

    if args.match_id is None:
        print("Provide --match_id or use --list <comp_key> to see available matches.")
        sys.exit(1)

    comp_key = args.competition
    path = PROCESSED / f"statsbomb_{comp_key}_shot_graphs.pt"
    if not path.exists():
        print(f"No data found: {path}")
        sys.exit(1)

    print(f"Loading {comp_key} graphs…")
    all_graphs = torch.load(path, weights_only=False)
    mg = [g for g in all_graphs if int(g.match_id.item()) == args.match_id]
    if not mg:
        print(f"No shots found for match_id={args.match_id}")
        sys.exit(1)

    home_team = mg[0].home_team
    teams     = {g.team_name for g in mg}
    away_team = next((t for t in teams if t != home_team), "Away")

    print(f"Match: {home_team} vs {away_team}  ({len(mg)} shots)")
    print("Loading model…")
    model, meta_dim = load_model()

    print("Running inference…")
    probs = run_inference(model, mg, meta_dim)

    goals  = sum(int(g.y.item()) for g in mg)
    home_g = sum(int(g.y.item()) for g in mg if g.team_name == home_team)
    away_g = goals - home_g
    print(f"Score: {home_team} {home_g}–{away_g} {away_team}")
    print(f"Model xG: {home_team} {sum(probs[i] for i, g in enumerate(mg) if g.team_name == home_team):.2f}"
          f" vs {away_team} {sum(probs[i] for i, g in enumerate(mg) if g.team_name != home_team):.2f}")

    comp_labels = {
        "wc2022": "FIFA World Cup 2022",
        "wwc2023": "Women's World Cup 2023",
        "euro2020": "UEFA Euro 2020",
        "euro2024": "UEFA Euro 2024",
        "bundesliga2324": "Bundesliga 2023/24",
        "weuro2022": "Women's Euro 2022",
        "weuro2025": "Women's Euro 2025",
    }
    comp_label = comp_labels.get(comp_key, comp_key)

    print("Building HTML report…")
    html = build_html_report(args.match_id, mg, probs, home_team, away_team, comp_label)

    safe_home = home_team.replace(" ", "_")
    safe_away = away_team.replace(" ", "_")
    out_path  = REPORTS_DIR / f"{comp_key}_{args.match_id}_{safe_home}_vs_{safe_away}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Report saved → {out_path}")
    print(f"   Open in browser: open '{out_path}'")


if __name__ == "__main__":
    main()
