"""
reproducibility.py
------------------
Utilities for reproducible ML runs. Addresses reviews/04_ml_engineer_review.md
§7.1 (no seed discipline) and §7.2 (headline numbers stranded in print
statements, unrecoverable without the exact config + git sha).

Two public surfaces:

1. `set_seed(seed)` — one call seeds python `random`, NumPy, torch CPU+CUDA,
   and (best-effort) deterministic cuDNN. Returns a dict you can log.

2. `RunLogger(path)` — append-only JSONL experiment log. Every training
   script should `RunLogger("results/runs.jsonl").log(run_config, metrics)`
   on exit. Zero-ops overhead; grep-able; survives crashes (line-buffered).

Non-goals
---------
We deliberately don't wrap MLflow / Weights & Biases. Those are great tools
but introduce server-side state and auth. A JSONL file on disk is enough to
recover "what produced AUC 0.760?" six months from now, which was the
actual pain point in the review.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

# NumPy / torch imports are deferred into the functions that need them so
# this module remains importable in a thin (NumPy-only) test sandbox.


# =============================================================================
# Seeding
# =============================================================================

def set_seed(seed: int, *, deterministic_cudnn: bool = True) -> dict[str, Any]:
    """Seed every RNG a training run is likely to touch.

    Parameters
    ----------
    seed : int
        The base seed. Python's `random`, `hashlib`-style environment hash
        (`PYTHONHASHSEED`), NumPy, and torch (CPU + all CUDA devices) are
        seeded with this value.
    deterministic_cudnn : bool
        If True and cuDNN is in use, disable its non-deterministic
        heuristics. This trades ~5-10% throughput for bitwise-identical
        training runs.

    Returns
    -------
    dict
        Summary of what was seeded — embed in the RunLogger config block so
        reruns have provenance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    info: dict[str, Any] = {"seed": seed, "pythonhashseed": str(seed)}

    try:
        import numpy as np  # noqa: WPS433
        np.random.seed(seed)
        info["numpy"] = True
    except ImportError:
        info["numpy"] = False

    try:
        import torch  # noqa: WPS433
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            info["cuda_seeded"] = True
        if deterministic_cudnn:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                info["cudnn_deterministic"] = True
            except Exception:
                # cuDNN not available (CPU-only wheel) — not a failure.
                info["cudnn_deterministic"] = False
        info["torch"] = True
    except ImportError:
        info["torch"] = False

    return info


# =============================================================================
# Provenance helpers
# =============================================================================

def git_sha(short: bool = True) -> str | None:
    """Return the current git HEAD SHA, or `None` if not a git repo.

    Never raises — if `git` isn't on PATH or the workdir is dirty, we simply
    return None and let the caller decide. Used inside `RunLogger` so a
    missing git install never blocks a training run from recording metrics.
    """
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short \
        else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(
            cmd, cwd=_repo_root(), stderr=subprocess.DEVNULL
        )
        return out.decode().strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def git_dirty() -> bool | None:
    """Return True if the working tree has uncommitted changes, else False.

    Returns None if git isn't available — callers should not treat that as
    "clean"; it's "unknown".
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_repo_root(),
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _repo_root() -> Path:
    """Guess the repo root from this file's location."""
    return Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# =============================================================================
# Experiment log
# =============================================================================

class RunLogger:
    """Append-only JSONL log of training runs.

    Usage
    -----
        logger = RunLogger("results/runs.jsonl")
        logger.log(
            script="scripts/train_xg_hybrid.py",
            config={"lr": 1e-3, "hidden": 64, "seed": 42},
            metrics={"val_auc": 0.760, "val_brier": 0.18},
            model_path="data/processed/pool_7comp_hybrid_xg.pt",
        )

    Output (one line per call):
        {
          "timestamp": "2026-04-17T12:03:41+00:00",
          "git_sha": "a1b2c3d",
          "git_dirty": false,
          "script": "scripts/train_xg_hybrid.py",
          "python_version": "3.11.7",
          "config": {...},
          "metrics": {...},
          "model_path": "data/processed/pool_7comp_hybrid_xg.pt"
        }

    Six months from now you can recover the config that produced any
    published metric with a one-liner:
        jq 'select(.metrics.val_auc > 0.75) | .config' results/runs.jsonl
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        script: str,
        config: Mapping[str, Any],
        metrics: Mapping[str, Any],
        model_path: str | Path | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one run record. Returns the record dict (useful for logging)."""
        record: dict[str, Any] = {
            "timestamp":      _utc_now_iso(),
            "git_sha":        git_sha(short=True),
            "git_dirty":      git_dirty(),
            "script":         script,
            "python_version": _python_version(),
            "config":         dict(config),
            "metrics":        dict(metrics),
            "model_path":     str(model_path) if model_path is not None else None,
        }
        if extra:
            record["extra"] = dict(extra)

        # Append mode with line buffering: a crash mid-training still
        # flushes the last successful record to disk.
        with self.path.open("a", buffering=1, encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=_json_fallback))
            f.write("\n")
        return record


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _json_fallback(obj: Any) -> Any:
    """JSON fallback for non-serialisable types (torch tensors, numpy arrays, paths)."""
    # Torch / numpy duck-typing without importing either module at serialise time.
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


__all__ = [
    "RunLogger",
    "git_dirty",
    "git_sha",
    "set_seed",
]
