"""
model_metadata.py
-----------------
Training ↔ serving contract for every model checkpoint.

Problem this solves
-------------------
Historically, `app.py` reverse-engineered `meta_dim`, `node_in`, and `edge_dim`
from the shape of `head.0.weight` at load time:

    hidden_dim      = 64
    actual_meta_dim = int(ckpt["head.0.weight"].shape[1]) - hidden_dim

That works today because training and serving are written by the same author
using the same hidden-dim constant. It is a **time bomb**: any silent feature
schema change in `build_shot_graphs.py` or a hidden-dim bump in training
produces a *running* dashboard with *wrong numbers*. No exception. No alert.

Solution
--------
At training time, every `.pt` checkpoint gets a companion `.meta.json` sidecar
describing the architecture and the feature schema it was trained against. At
serving time, the loader reads the sidecar and **fails loudly** if the
serving-side schema doesn't match.

Sidecar layout (see §3.1 of 04_ml_engineer_review.md for rationale):

    {
      "model_class": "HybridGATModel",
      "hidden_dim": 32,
      "heads": 4,
      "n_layers": 3,
      "dropout": 0.3,
      "node_in": 9,
      "edge_dim": 4,
      "meta_dim": 27,
      "feature_schema_version": "v3-psxg",
      "feature_order": ["shot_dist", "shot_angle", ...],
      "trained_on": "pool_7comp_v1",
      "trained_commit": "<git-sha>",
      "trained_at": "2026-04-12T14:00:00Z",
      "val_auc": 0.760,
      "val_brier": 0.148,
      "val_ece": 0.042,
      "temperature_per_competition": {"wc2022": 1.14, ...}
    }

Usage
-----
    from src.model_metadata import (
        ModelMetadata, save_sidecar, load_sidecar, assert_compatible,
        sidecar_path_for,
    )

    # At train time — after torch.save(model.state_dict(), ckpt):
    meta = ModelMetadata.from_hybrid_gat(
        model, val_auc=0.760, val_brier=0.148, trained_on="pool_7comp_v1")
    save_sidecar(meta, ckpt)

    # At serve time — before constructing the model:
    meta = load_sidecar(ckpt)
    assert_compatible(meta, expected_schema="v3-psxg", expected_meta_dim=27)
    model = build_model_from_meta(meta)
    model.load_state_dict(torch.load(ckpt, weights_only=True))

Design notes
------------
* The sidecar lives *next to* the .pt so it moves with it through
  upload/download. Do NOT embed in the state_dict: state_dict is torch-pickle,
  JSON is diff-friendly, grep-friendly, and machine-readable.
* `feature_schema_version` is the *single* source of truth for feature layout.
  Bumping it is a breaking change — the loader will reject any graph file
  that doesn't declare the same version.
* Everything optional (val_*, trained_commit) uses sentinel nulls so missing
  fields don't crash deserialisation.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# ─── Schema constants ───────────────────────────────────────────────────────
# Bump this any time you change the feature layout in _build_meta / _default_meta.
# Bumping it is a breaking change: every checkpoint sidecar must reference one of
# the versions listed in KNOWN_SCHEMAS, and the loader rejects unknown versions.
CURRENT_FEATURE_SCHEMA_VERSION = "v3-psxg"

# Feature order must mirror src/calibration.py::_default_meta and
# app.py::_build_meta. These modules share src/features.py::build_meta as a
# single implementation once §8 of the review is complete.
FEATURE_ORDER_V3_PSXG = [
    # [0:4]   base
    "shot_dist", "shot_angle", "is_header", "is_open_play",
    # [4:12]  technique one-hot (NUM_TECHNIQUES=8)
    "technique[0]", "technique[1]", "technique[2]", "technique[3]",
    "technique[4]", "technique[5]", "technique[6]", "technique[7]",
    # [12:15] gk-original
    "gk_dist", "n_def_in_cone", "gk_off_centre",
    # [15:18] gk-precision
    "gk_perp_offset", "n_def_direct_line", "is_right_foot",
    # [18:27] PSxG shot-placement one-hot (NUM_PLACEMENTS=9)
    "shot_placement[0]", "shot_placement[1]", "shot_placement[2]",
    "shot_placement[3]", "shot_placement[4]", "shot_placement[5]",
    "shot_placement[6]", "shot_placement[7]", "shot_placement[8]",
]

# Mapping of every supported schema version → its expected total dimensionality
# + ordered feature name list. New versions extend this map; old versions stay
# for back-compat so yesterday's checkpoints still load.
KNOWN_SCHEMAS: dict[str, dict[str, Any]] = {
    "v1-base": {
        "meta_dim": 12,
        "feature_order": FEATURE_ORDER_V3_PSXG[:12],
    },
    "v2-gk": {
        "meta_dim": 18,
        "feature_order": FEATURE_ORDER_V3_PSXG[:18],
    },
    "v3-psxg": {
        "meta_dim": 27,
        "feature_order": FEATURE_ORDER_V3_PSXG[:27],
    },
}

# Sidecar file suffix — a single `.meta.json` next to `<name>.pt`.
SIDECAR_SUFFIX = ".meta.json"

# Version of the sidecar file format itself (not the feature schema).
# Bump on breaking changes to the sidecar layout.
METADATA_FORMAT_VERSION = 1


# ─── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class ModelMetadata:
    """
    Companion metadata for a single `.pt` checkpoint.

    Every field serialises to JSON. Optional fields are `None` when unknown
    (e.g. `val_auc` may be unset for a checkpoint written mid-training).
    """
    # ── Architecture ────────────────────────────────────────────────────────
    model_class: str          # "HybridXGModel" | "HybridGATModel" | ...
    hidden_dim: int           # hidden channels per GCN layer or per GAT head
    node_in: int              # node feature dimensionality at layer 0
    edge_dim: int             # edge feature dimensionality; 0 if unused
    meta_dim: int             # shot-metadata tensor width (must equal the
                              # `meta_dim` the head Linear layer expects)
    # GAT-specific (ignored for pure GCN but always serialised)
    heads: int = 1
    n_layers: int = 3
    dropout: float = 0.3

    # ── Feature-schema contract ─────────────────────────────────────────────
    feature_schema_version: str = CURRENT_FEATURE_SCHEMA_VERSION
    feature_order: list[str] = field(default_factory=lambda: list(FEATURE_ORDER_V3_PSXG))

    # ── Training lineage ────────────────────────────────────────────────────
    trained_on: Optional[str] = None          # dataset id, e.g. "pool_7comp_v1"
    trained_commit: Optional[str] = None      # git sha (short) at train time
    trained_at: Optional[str] = None          # ISO-8601 timestamp
    torch_version: Optional[str] = None       # torch.__version__ at train time
    pyg_version: Optional[str] = None         # torch_geometric.__version__

    # ── Validation metrics ──────────────────────────────────────────────────
    val_auc: Optional[float] = None
    val_brier: Optional[float] = None
    val_ece: Optional[float] = None

    # ── Temperature scaling (per-competition calibration) ───────────────────
    temperature_global: Optional[float] = None
    temperature_per_competition: Optional[dict[str, float]] = None

    # ── Integrity ───────────────────────────────────────────────────────────
    # sha256 of the companion `.pt` file at the time this metadata was written.
    # Verified at load time — mismatch indicates a torn upload/download.
    checkpoint_sha256: Optional[str] = None

    # ── Internal ────────────────────────────────────────────────────────────
    metadata_format_version: int = METADATA_FORMAT_VERSION

    # ------------------------------------------------------------------ ctors

    @classmethod
    def from_hybrid_xg(
        cls,
        model: Any,
        *,
        node_in: int,
        edge_dim: int = 0,
        trained_on: Optional[str] = None,
        val_auc: Optional[float] = None,
        val_brier: Optional[float] = None,
        val_ece: Optional[float] = None,
        temperature_global: Optional[float] = None,
        temperature_per_competition: Optional[dict[str, float]] = None,
    ) -> "ModelMetadata":
        """Build metadata for a HybridXGModel (GCN-backed hybrid)."""
        # Pull hidden_dim + meta_dim from the actual head layer so sidecar and
        # weights cannot drift — the head IS the contract.
        head_in = int(model.head[0].in_features)           # hidden_dim + meta_dim
        head_out = int(model.head[0].out_features)         # hidden_dim
        hidden_dim = head_out
        meta_dim = head_in - hidden_dim

        return cls(
            model_class="HybridXGModel",
            hidden_dim=hidden_dim,
            node_in=node_in,
            edge_dim=edge_dim,
            meta_dim=meta_dim,
            heads=1,
            n_layers=len(getattr(model, "convs", [])) or 3,
            dropout=float(getattr(model, "dropout", 0.3)),
            feature_schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
            feature_order=list(_feature_order_for(meta_dim)),
            trained_on=trained_on,
            trained_commit=_safe_git_sha(),
            trained_at=_utc_now_iso(),
            torch_version=_safe_torch_version(),
            pyg_version=_safe_pyg_version(),
            val_auc=val_auc,
            val_brier=val_brier,
            val_ece=val_ece,
            temperature_global=temperature_global,
            temperature_per_competition=temperature_per_competition,
        )

    @classmethod
    def from_hybrid_gat(
        cls,
        model: Any,
        *,
        node_in: int,
        edge_dim: int = 0,
        trained_on: Optional[str] = None,
        val_auc: Optional[float] = None,
        val_brier: Optional[float] = None,
        val_ece: Optional[float] = None,
        temperature_global: Optional[float] = None,
        temperature_per_competition: Optional[dict[str, float]] = None,
    ) -> "ModelMetadata":
        """Build metadata for a HybridGATModel."""
        head_in = int(model.head[0].in_features)    # pool_dim + meta_dim
        pool_dim = int(getattr(model, "_pool_dim", model.head[0].out_features))
        meta_dim = head_in - pool_dim
        hidden_dim = pool_dim   # for HybridGATModel with n_layers>1, pool_dim == hidden

        return cls(
            model_class="HybridGATModel",
            hidden_dim=hidden_dim,
            node_in=node_in,
            edge_dim=edge_dim,
            meta_dim=meta_dim,
            heads=int(getattr(model.convs[0], "heads", 1)),
            n_layers=int(getattr(model, "n_layers", len(model.convs))),
            dropout=float(getattr(model, "dropout", 0.3)),
            feature_schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
            feature_order=list(_feature_order_for(meta_dim)),
            trained_on=trained_on,
            trained_commit=_safe_git_sha(),
            trained_at=_utc_now_iso(),
            torch_version=_safe_torch_version(),
            pyg_version=_safe_pyg_version(),
            val_auc=val_auc,
            val_brier=val_brier,
            val_ece=val_ece,
            temperature_global=temperature_global,
            temperature_per_competition=temperature_per_competition,
        )

    # ----------------------------------------------------------------- shape

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelMetadata":
        # Forward-compatibility: drop unknown keys, fill missing ones with
        # dataclass defaults. This lets future versions add fields without
        # breaking the current loader.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ─── Schema-compatibility gate ──────────────────────────────────────────────

class FeatureSchemaMismatch(RuntimeError):
    """
    Raised when a checkpoint's declared feature schema does not match what
    the serving code is prepared to build. Fail loudly — do NOT try to
    degrade silently by dropping or padding features.
    """


def assert_compatible(
    meta: ModelMetadata,
    *,
    expected_schema: str = CURRENT_FEATURE_SCHEMA_VERSION,
    expected_meta_dim: Optional[int] = None,
    expected_model_class: Optional[str] = None,
) -> None:
    """
    Hard gate the checkpoint against the running codebase.

    Parameters
    ----------
    meta : ModelMetadata
        Sidecar freshly loaded next to a .pt.
    expected_schema : str
        The feature-schema version the serving code currently implements.
        Usually `CURRENT_FEATURE_SCHEMA_VERSION`.
    expected_meta_dim : int, optional
        If passed, also check that the sidecar's meta_dim matches. Useful
        for the `assert model.meta == ckpt.meta` double-check.
    expected_model_class : str, optional
        If passed, require the sidecar declare this model class. Used by
        `load_gat_model` to refuse a `HybridXGModel` checkpoint handed to it.

    Raises
    ------
    FeatureSchemaMismatch
        On any mismatch. Message includes every relevant field so the
        operator knows which artefact is stale.
    """
    # 1. Feature schema version must match exactly.
    if meta.feature_schema_version != expected_schema:
        known = ", ".join(KNOWN_SCHEMAS.keys())
        raise FeatureSchemaMismatch(
            f"Checkpoint declares feature_schema_version="
            f"{meta.feature_schema_version!r} but serving code implements "
            f"{expected_schema!r}. Rebuild graphs with scripts/build_shot_graphs.py "
            f"at the matching schema, or retrain the model against the current schema. "
            f"(Known schemas: {known}.)"
        )

    # 2. The schema version must actually be one we know how to handle.
    if meta.feature_schema_version not in KNOWN_SCHEMAS:
        raise FeatureSchemaMismatch(
            f"Unknown feature_schema_version={meta.feature_schema_version!r}. "
            f"Known: {list(KNOWN_SCHEMAS.keys())}. "
            f"Either bump src/model_metadata.py::KNOWN_SCHEMAS, or retrain."
        )

    # 3. Declared meta_dim must match what the schema's feature order says.
    expected_for_schema = KNOWN_SCHEMAS[meta.feature_schema_version]["meta_dim"]
    if meta.meta_dim != expected_for_schema:
        raise FeatureSchemaMismatch(
            f"Checkpoint meta_dim={meta.meta_dim} but schema "
            f"{meta.feature_schema_version!r} requires {expected_for_schema}. "
            f"Sidecar is internally inconsistent — regenerate it."
        )

    # 4. Caller-supplied overrides.
    if expected_meta_dim is not None and meta.meta_dim != expected_meta_dim:
        raise FeatureSchemaMismatch(
            f"Checkpoint meta_dim={meta.meta_dim} but caller expected "
            f"{expected_meta_dim}."
        )
    if expected_model_class is not None and meta.model_class != expected_model_class:
        raise FeatureSchemaMismatch(
            f"Checkpoint model_class={meta.model_class!r} but caller expected "
            f"{expected_model_class!r}."
        )


# ─── Disk I/O ───────────────────────────────────────────────────────────────

def sidecar_path_for(ckpt_path: Path | str) -> Path:
    """Return the conventional sidecar path next to a checkpoint."""
    p = Path(ckpt_path)
    return p.with_suffix(p.suffix + SIDECAR_SUFFIX) if p.suffix != ".pt" \
        else p.with_name(p.stem + SIDECAR_SUFFIX)


def save_sidecar(meta: ModelMetadata, ckpt_path: Path | str) -> Path:
    """
    Write the metadata JSON next to `ckpt_path` and return the sidecar path.

    If `ckpt_path` exists, also compute and store its sha256 under
    `meta.checkpoint_sha256` so downstream loaders can verify integrity.
    """
    ckpt = Path(ckpt_path)
    if ckpt.exists() and meta.checkpoint_sha256 is None:
        meta.checkpoint_sha256 = _sha256_file(ckpt)

    sidecar = sidecar_path_for(ckpt)
    sidecar.write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=True))
    return sidecar


def load_sidecar(
    ckpt_path: Path | str,
    *,
    verify_checksum: bool = True,
) -> ModelMetadata:
    """
    Read metadata JSON for `ckpt_path`.

    Parameters
    ----------
    ckpt_path : path to the `.pt` checkpoint; sidecar is discovered adjacent.
    verify_checksum : if True and the sidecar records a `checkpoint_sha256`,
        recompute the file hash and fail loudly on mismatch.

    Raises
    ------
    FileNotFoundError
        If the sidecar does not exist. Missing sidecar means the checkpoint
        pre-dates the metadata system; the caller should refuse to serve it.
    FeatureSchemaMismatch
        On checksum mismatch.
    """
    ckpt = Path(ckpt_path)
    sidecar = sidecar_path_for(ckpt)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"No metadata sidecar at {sidecar}. Every checkpoint must have a "
            f"companion {SIDECAR_SUFFIX} — re-export via scripts/train_xg_hybrid.py "
            f"or regenerate with scripts/backfill_metadata.py."
        )
    data = json.loads(sidecar.read_text())
    meta = ModelMetadata.from_dict(data)

    if verify_checksum and meta.checkpoint_sha256 and ckpt.exists():
        actual = _sha256_file(ckpt)
        if actual != meta.checkpoint_sha256:
            raise FeatureSchemaMismatch(
                f"Checkpoint {ckpt.name} sha256 mismatch: "
                f"sidecar expects {meta.checkpoint_sha256[:12]}…, "
                f"file on disk is {actual[:12]}…. "
                f"Torn upload/download — redownload from HF Hub."
            )
    return meta


# ─── Helpers ────────────────────────────────────────────────────────────────

def _feature_order_for(meta_dim: int) -> list[str]:
    """Return the feature name list for a given meta_dim, trimmed if needed."""
    for _, spec in KNOWN_SCHEMAS.items():
        if spec["meta_dim"] == meta_dim:
            return list(spec["feature_order"])
    # Default to the full v3 list truncated — caller is in uncharted territory
    # but we don't want to crash sidecar creation over it.
    return FEATURE_ORDER_V3_PSXG[:meta_dim]


def _sha256_file(path: Path, *, block_size: int = 1 << 20) -> str:
    """Streaming sha256 — safe for multi-GB checkpoints without blowing RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_git_sha() -> Optional[str]:
    """Return short git sha if available; None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, OSError):
        return None


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with second precision, suffixed 'Z'."""
    # `timezone.utc` + .isoformat() produces e.g. "2026-04-17T09:00:00+00:00"
    # — we normalise to the "...Z" form widely used in JSON model cards.
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_torch_version() -> Optional[str]:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def _safe_pyg_version() -> Optional[str]:
    try:
        import torch_geometric
        return torch_geometric.__version__
    except Exception:
        return None
