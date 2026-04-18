"""
serving.py
----------
Pure-Python serving layer for the HybridXGModel / HybridGATModel + temperature
scalers. Extracted from `app.py` so the full load-predict round-trip can be
exercised without importing Streamlit.

Why this exists
---------------
Before this module, `app.py` contained:
  1. a redefinition of `HybridXGModel` (drifting copy #4 of the same class),
  2. a Streamlit-coupled `load_model()` that called `st.error`/`st.stop` on
     sidecar failures,
  3. inline model-building that read hyperparameters from tensor shapes
     instead of the sidecar.

Tests, CLI tools, and notebooks couldn't touch any of that without pulling
in Streamlit. `src/serving.py` is the Streamlit-free equivalent:

    from src.serving import load_gcn_model, load_gat_model, predict_one
    model = load_gcn_model(Path("data/processed/pool_7comp_hybrid_xg.pt"),
                           Path("data/processed/pool_7comp_T.pt"))
    prob  = predict_one(model, graph)

Contract
--------
`load_gcn_model` and `load_gat_model` are **fail-loud**. They raise
`FileNotFoundError` / `FeatureSchemaMismatch` / `RuntimeError` on any
contract violation — the caller is expected to wrap them in a UI-specific
error boundary (`app.py` uses `st.error` + `st.stop`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch, Data

from src.calibration import TemperatureScaler
from src.features import build_meta
from src.model_metadata import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    FeatureSchemaMismatch,
    ModelMetadata,
    assert_compatible,
    load_sidecar,
)
from src.models import HybridGATModel, HybridXGModel


# =============================================================================
# Load helpers
# =============================================================================

def load_gcn_model(
    model_path: Path | str,
    temp_path:  Path | str | None = None,
    *,
    expected_schema: str = CURRENT_FEATURE_SCHEMA_VERSION,
    verify_checksum: bool = True,
    device:     str = "cpu",
) -> TemperatureScaler:
    """
    Load a HybridXGModel checkpoint + its temperature scalar into a
    `TemperatureScaler` wrapper and return it in `eval()` mode.

    Parameters
    ----------
    model_path : Path
        `.pt` file holding the `state_dict` for HybridXGModel. A companion
        `.meta.json` sidecar MUST exist next to it (see
        `src/model_metadata.py`) — this function refuses to silently
        construct a model from a weights file alone.
    temp_path : Path | None
        Optional `.pt` file holding the fitted temperature `T`. Absent or
        `None` → no-op scaler (`T = 1.0`).
    expected_schema : str
        Feature schema version the serving code is built against. Mismatch
        between this and `meta.feature_schema_version` raises
        `FeatureSchemaMismatch`.
    verify_checksum : bool
        If True and the sidecar records `checkpoint_sha256`, recompute
        the hash on disk and raise on mismatch.
    device : str
        Torch device string.

    Raises
    ------
    FileNotFoundError
        If `model_path` or its sidecar is missing.
    FeatureSchemaMismatch
        If the sidecar's schema doesn't match `expected_schema`, or the
        declared `model_class` is not `HybridXGModel`.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"GCN checkpoint {model_path} not found. Run "
            f"`python scripts/train_xg_hybrid.py` to produce it."
        )

    # 1. Sidecar is the authoritative source for architecture & schema.
    meta = load_sidecar(model_path, verify_checksum=verify_checksum)
    assert_compatible(
        meta,
        expected_schema=expected_schema,
        expected_model_class="HybridXGModel",
    )

    # 2. Build the model using sidecar-declared dimensions (no shape peeking).
    base = HybridXGModel(
        in_channels=meta.node_in,
        hidden_dim=meta.hidden_dim,
        meta_dim=meta.meta_dim,
        dropout=meta.dropout,
    )
    ckpt = torch.load(model_path, weights_only=True, map_location=device)
    base.load_state_dict(ckpt)
    base.eval()

    # 3. Wrap with temperature scaler.
    if temp_path is not None and Path(temp_path).exists():
        scaler = TemperatureScaler.load(base, Path(temp_path))
    else:
        scaler = TemperatureScaler(base, init_T=1.0)
    scaler.eval()

    # 4. Stash authoritative metadata for consumers (e.g. `_schema_for`).
    _attach_meta(scaler, meta)
    return scaler


def load_gat_model(
    model_path: Path | str,
    temp_path:  Path | str | None = None,
    *,
    expected_schema: str = CURRENT_FEATURE_SCHEMA_VERSION,
    verify_checksum: bool = True,
    device:     str = "cpu",
) -> TemperatureScaler:
    """
    Load a HybridGATModel checkpoint + its temperature scalar.

    Parameters mirror `load_gcn_model`. See that function for the contract.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"GAT checkpoint {model_path} not found. Run "
            f"`python scripts/calibrate_and_train_gat.py` to produce it."
        )

    meta = load_sidecar(model_path, verify_checksum=verify_checksum)
    assert_compatible(
        meta,
        expected_schema=expected_schema,
        expected_model_class="HybridGATModel",
    )

    gat = HybridGATModel(
        node_in=meta.node_in,
        edge_dim=meta.edge_dim,
        meta_dim=meta.meta_dim,
        hidden=meta.hidden_dim,
        heads=meta.heads,
        n_layers=meta.n_layers,
        dropout=meta.dropout,
    )
    ckpt = torch.load(model_path, weights_only=True, map_location=device)
    gat.load_state_dict(ckpt)
    gat.eval()

    if temp_path is not None and Path(temp_path).exists():
        scaler = TemperatureScaler.load(gat, Path(temp_path))
    else:
        scaler = TemperatureScaler(gat, init_T=1.0)
    scaler.eval()

    _attach_meta(scaler, meta)
    return scaler


# =============================================================================
# Inference helpers
# =============================================================================

def schema_of(model: Any) -> str:
    """Return the `feature_schema_version` stamped on the model wrapper.

    Falls back to `CURRENT_FEATURE_SCHEMA_VERSION` only if the wrapper has
    no sidecar-derived metadata (shouldn't happen in production — every
    model produced by this module stashes one).
    """
    meta = getattr(model, "_model_meta", None)
    if meta is not None and getattr(meta, "feature_schema_version", None):
        return meta.feature_schema_version
    return CURRENT_FEATURE_SCHEMA_VERSION


def predict_one(model: Any, graph: Data, *, device: str = "cpu") -> float:
    """
    Run one forward pass on a single shot graph and return the calibrated
    probability in `[0, 1]`.

    The metadata tensor is built via the canonical `features.build_meta`,
    using the schema version declared on the model's sidecar. Missing graph
    attributes raise `GraphSchemaMismatch` — we do NOT impute defaults.
    """
    schema = schema_of(model)
    batch  = Batch.from_data_list([graph]).to(device)
    meta   = build_meta(batch, schema_version=schema).to(device)

    with torch.no_grad():
        logit = model(batch.x, batch.edge_index, batch.batch, meta)
    return float(torch.sigmoid(logit).squeeze().item())


def predict_batch(
    model: Any,
    graphs: list[Data],
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run one forward pass over a batch of graphs and return calibrated
    probabilities as a `(len(graphs),)` float tensor.
    """
    schema = schema_of(model)
    batch  = Batch.from_data_list(list(graphs)).to(device)
    meta   = build_meta(batch, schema_version=schema).to(device)

    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch, meta)
    return torch.sigmoid(logits).squeeze(-1).cpu()


# =============================================================================
# Internals
# =============================================================================

def _attach_meta(scaler: TemperatureScaler, meta: ModelMetadata) -> None:
    """Stash authoritative metadata on the wrapper so downstream code can
    read schema, meta_dim, etc. without re-peeking at sidecar files."""
    scaler._meta_dim   = meta.meta_dim       # noqa: SLF001
    scaler._model_meta = meta                # noqa: SLF001


__all__ = [
    "load_gcn_model",
    "load_gat_model",
    "predict_one",
    "predict_batch",
    "schema_of",
]
