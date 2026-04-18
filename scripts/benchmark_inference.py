#!/usr/bin/env python3
"""
benchmark_inference.py
----------------------
Measure latency and throughput of the served HybridGCN / HybridGAT xG models.

Addresses reviews/04_ml_engineer_review.md §7.5 — "a hiring manager may ask
'can this serve 1000 rps?' — having a 30-second answer is a force multiplier."

What it measures
----------------
1. **Model load time** — cold-start cost (read `.pt` + sidecar + build model
   + apply temperature scaler). This is the worst-case latency the first user
   sees after a Space wakes up from scale-to-zero.
2. **Single-graph latency** — p50 / p95 / p99 over N warm predictions. This is
   what the Streamlit UI experiences on each shot click.
3. **Batch throughput** — shots/sec at a range of batch sizes. This is what
   matters for bulk-scoring a full competition.
4. **Model footprint** — parameter count and on-disk checkpoint size.

Example
-------
    python scripts/benchmark_inference.py                    # GCN, 500 warm iters
    python scripts/benchmark_inference.py --model gat --n 1000
    python scripts/benchmark_inference.py --batch-sizes 1 8 32 128 512
    python scripts/benchmark_inference.py --json results/bench_gcn.json

Output
------
Prints a human-readable table and (optionally) writes a JSON file for the
README benchmark section.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from src.serving import load_gat_model, load_gcn_model, predict_batch  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"

DEFAULT_PATHS: dict[str, dict[str, Path]] = {
    "gcn": {
        "model": PROCESSED / "pool_7comp_hybrid_xg.pt",
        "temp":  PROCESSED / "pool_7comp_T.pt",
    },
    "gat": {
        "model": PROCESSED / "pool_7comp_hybrid_gat_xg.pt",
        "temp":  PROCESSED / "pool_7comp_gat_T.pt",
    },
}


# =============================================================================
# Result record
# =============================================================================

@dataclass
class BenchResult:
    model: str
    device: str
    n_parameters: int
    checkpoint_bytes: int
    load_time_s: float
    warm_iters: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    batch_throughput: dict[int, float]  # batch_size -> shots/sec

    def to_json(self) -> dict:
        d = asdict(self)
        # JSON wants string keys
        d["batch_throughput"] = {str(k): v for k, v in d["batch_throughput"].items()}
        return d


# =============================================================================
# Data helpers — load real graphs if present, otherwise synthesise
# =============================================================================

def _load_real_graphs(max_graphs: int) -> list:
    """Prefer real shot graphs from `data/processed/`. Falls back to synthetic."""
    paths = sorted(PROCESSED.glob("statsbomb_*_shot_graphs.pt"))
    if not paths:
        return []
    graphs = []
    for p in paths:
        gs = torch.load(p, weights_only=False)
        graphs.extend(gs)
        if len(graphs) >= max_graphs:
            break
    return graphs[:max_graphs]


def _synth_graphs(n: int, seed: int = 0) -> list:
    """Build synthetic shot-graph fixtures if real ones aren't downloaded.

    Mirrors the v3-psxg shape used in tests/test_serving.py so `predict_batch`
    doesn't trip on a missing attribute.
    """
    from torch_geometric.data import Data
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        n_nodes = 22
        x = torch.randn((n_nodes, 5), generator=g)
        ei = torch.randint(0, n_nodes, (2, 60), generator=g)
        ea = torch.randn((ei.shape[1], 3), generator=g)
        tech = torch.zeros(8); tech[torch.randint(0, 8, (1,), generator=g)] = 1
        plc  = torch.zeros(9); plc[torch.randint(0, 9, (1,), generator=g)] = 1
        d = Data(
            x=x, edge_index=ei, edge_attr=ea,
            y=torch.tensor([0], dtype=torch.float),
            shot_dist=torch.tensor([15.0]),
            shot_angle=torch.tensor([0.35]),
            is_header=torch.tensor([0.0]),
            is_open_play=torch.tensor([1.0]),
            technique=tech.unsqueeze(0),
            gk_dist=torch.tensor([10.0]),
            n_def_in_cone=torch.tensor([1.0]),
            gk_off_centre=torch.tensor([0.2]),
            gk_perp_offset=torch.tensor([0.5]),
            n_def_direct_line=torch.tensor([0.0]),
            is_right_foot=torch.tensor([1.0]),
            shot_placement=plc.unsqueeze(0),
        )
        out.append(d)
    return out


# =============================================================================
# Core benchmark
# =============================================================================

def measure_load(
    model_name: str,
    paths: dict[str, Path],
    device: str,
) -> tuple[object, float]:
    """Time a cold load: read → build → calibrate → ready for inference."""
    t0 = time.perf_counter()
    if model_name == "gcn":
        m = load_gcn_model(paths["model"], paths["temp"], device=device)
    elif model_name == "gat":
        m = load_gat_model(paths["model"], paths["temp"], device=device)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
    dt = time.perf_counter() - t0
    return m, dt


def measure_single_latency(
    model,
    graphs: list,
    n_iters: int,
    warmup: int,
    device: str,
) -> dict[str, float]:
    """p50/p95/p99 single-graph latency over N iterations after W warmups."""
    # Warmup — cache kernels, resolve lazy loads.
    for _ in range(warmup):
        predict_batch(model, graphs[:1], device=device)

    samples_ms: list[float] = []
    n_pool = len(graphs)
    for i in range(n_iters):
        g = graphs[i % n_pool]
        t0 = time.perf_counter()
        predict_batch(model, [g], device=device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    return {
        "p50":  statistics.median(samples_ms),
        "p95":  samples_ms[int(0.95 * len(samples_ms))],
        "p99":  samples_ms[int(0.99 * len(samples_ms))],
        "mean": statistics.fmean(samples_ms),
    }


def measure_batch_throughput(
    model,
    graphs: list,
    batch_sizes: list[int],
    device: str,
    total_shots_per_bs: int,
) -> dict[int, float]:
    """Shots-per-second at each batch size.

    We score `total_shots_per_bs` shots in batches of size `bs` and divide
    elapsed by total shots. Using the same total across batch sizes makes
    the comparison fair (longer wall-clock for bs=1, same work done).
    """
    throughput: dict[int, float] = {}
    pool = graphs * (1 + total_shots_per_bs // max(1, len(graphs)))
    for bs in batch_sizes:
        n_batches = max(1, total_shots_per_bs // bs)
        # warmup
        predict_batch(model, pool[:bs], device=device)
        t0 = time.perf_counter()
        for b in range(n_batches):
            start = (b * bs) % (len(pool) - bs + 1)
            predict_batch(model, pool[start:start + bs], device=device)
        dt = time.perf_counter() - t0
        shots_done = n_batches * bs
        throughput[bs] = shots_done / dt
    return throughput


# =============================================================================
# Reporting
# =============================================================================

def _fmt_ms(x: float) -> str:
    return f"{x:.2f} ms"


def _fmt_sps(x: float) -> str:
    return f"{x:,.1f} shots/s"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def print_report(r: BenchResult) -> None:
    print("=" * 62)
    print(f"  Inference benchmark — {r.model.upper()}  (device={r.device})")
    print("=" * 62)
    print(f"  Parameters         : {r.n_parameters:,}")
    print(f"  Checkpoint on disk : {_fmt_bytes(r.checkpoint_bytes)}")
    print(f"  Cold load time     : {r.load_time_s:.3f} s")
    print()
    print(f"  Single-graph latency (n={r.warm_iters:,} warm iters)")
    print(f"    p50  : {_fmt_ms(r.latency_p50_ms)}")
    print(f"    p95  : {_fmt_ms(r.latency_p95_ms)}")
    print(f"    p99  : {_fmt_ms(r.latency_p99_ms)}")
    print(f"    mean : {_fmt_ms(r.latency_mean_ms)}")
    print()
    print(f"  Batch throughput")
    print(f"    {'bs':>5}  {'shots/sec':>14}")
    for bs, sps in sorted(r.batch_throughput.items()):
        print(f"    {bs:>5}  {_fmt_sps(sps):>14}")
    print("=" * 62)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark HybridGCN / HybridGAT inference latency + throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["gcn", "gat"], default="gcn",
                        help="Which model to benchmark")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Override model checkpoint path")
    parser.add_argument("--temp-path",  type=Path, default=None,
                        help="Override temperature scaler path")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Inference device")
    parser.add_argument("--n", type=int, default=500,
                        help="Single-graph latency iterations (after warmup)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations before timing")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 8, 32, 128],
                        help="Batch sizes to benchmark throughput at")
    parser.add_argument("--shots-per-bs", type=int, default=1024,
                        help="Approx. shots scored at each batch size")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write results to this JSON file (e.g. results/bench_gcn.json)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic graphs even if real ones are on disk")
    args = parser.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────
    defaults = DEFAULT_PATHS[args.model]
    model_path = args.model_path or defaults["model"]
    temp_path  = args.temp_path  or defaults["temp"]
    if not model_path.exists():
        print(f"ERROR: model checkpoint not found: {model_path}", file=sys.stderr)
        print("Hint: run scripts/train_xg_hybrid.py, or pull from HF Hub first.",
              file=sys.stderr)
        return 2

    # ── Load model ─────────────────────────────────────────────────────
    print(f"[load] {args.model.upper()} from {model_path.name} (+ {temp_path.name})")
    model, load_dt = measure_load(args.model, {"model": model_path, "temp": temp_path},
                                  device=args.device)
    n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad) \
        if hasattr(model, "model") else \
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_size = model_path.stat().st_size

    # ── Data ───────────────────────────────────────────────────────────
    graphs = [] if args.synthetic else _load_real_graphs(max_graphs=1024)
    if not graphs:
        print("[data] No real shot graphs found — using synthetic fixtures.")
        graphs = _synth_graphs(1024)
    else:
        print(f"[data] Loaded {len(graphs)} real shot graphs for benchmarking.")

    # ── Timings ────────────────────────────────────────────────────────
    print(f"[warmup] {args.warmup} iters; [latency] {args.n} iters")
    lat = measure_single_latency(model, graphs, args.n, args.warmup, args.device)
    print(f"[throughput] batch sizes: {args.batch_sizes}  (~{args.shots_per_bs} shots/bs)")
    tput = measure_batch_throughput(model, graphs, args.batch_sizes, args.device,
                                    total_shots_per_bs=args.shots_per_bs)

    # ── Report ─────────────────────────────────────────────────────────
    result = BenchResult(
        model=args.model,
        device=args.device,
        n_parameters=n_params,
        checkpoint_bytes=ckpt_size,
        load_time_s=load_dt,
        warm_iters=args.n,
        latency_p50_ms=lat["p50"],
        latency_p95_ms=lat["p95"],
        latency_p99_ms=lat["p99"],
        latency_mean_ms=lat["mean"],
        batch_throughput=tput,
    )
    print()
    print_report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n")
        print(f"\n[json] wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
