"""
observability.py
----------------
Logging, error boundaries, and per-session tracing for the Streamlit app and
training scripts. Addresses reviews/04_ml_engineer_review.md §8 "Before
launch" — specifically:

  - structured logging (timestamp + level + logger name)
  - Streamlit per-view error boundary (one view's bug doesn't blank the app)
  - request tracing via st.session_state.session_id

Import-order discipline: this module must not pull in Streamlit at import
time. Training scripts call `configure_logging()` happily without ever
touching `st`. The Streamlit-specific helpers defer their `streamlit` import
to call time so a CLI `python scripts/train_xg_hybrid.py` never imports
Streamlit.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from typing import Any, Iterator


# =============================================================================
# Logging
# =============================================================================

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"


def configure_logging(
    level: str | int = "INFO",
    *,
    stream=None,
    force: bool = False,
) -> None:
    """One-shot logging config. Call from `main()` / `app.py` entry point.

    Idempotent by design: if a handler is already attached (e.g. Streamlit's
    bootstrap installed its own), we keep the first config unless `force=True`.
    The review's "minimum viable" is that every training log line carries
    `YYYY-MM-DD HH:MM:SS LEVEL name:` so grep-able post-hoc reconstruction is
    trivial.
    """
    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        level = env_level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stream or sys.stderr,
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Convenience pass-through that matches `logging.getLogger` signature.

    Exists so call sites can `from src.observability import get_logger` and
    stay agnostic about whether logging has been configured yet.
    """
    return logging.getLogger(name)


# =============================================================================
# Session tracing (Streamlit)
# =============================================================================

_SESSION_KEY = "_session_id"


def get_session_id() -> str:
    """Return (creating if needed) the per-browser-session UUID.

    Stored in `st.session_state` so the id is stable across reruns within one
    Streamlit session, and different across tabs / incognito windows. On HF
    Spaces this shows up in the log feed and lets an operator correlate "user
    X crashed on tab Y at 12:03Z" from otherwise-anonymous logs.

    Outside of Streamlit (CLI scripts, tests), returns a fresh UUID that is
    *not* memoised — there's no session to bind it to.
    """
    try:
        import streamlit as st  # noqa: WPS433 — deferred import, see module docstring
    except ImportError:
        return str(uuid.uuid4())

    try:
        state = st.session_state
    except Exception:
        return str(uuid.uuid4())

    if _SESSION_KEY not in state:
        state[_SESSION_KEY] = str(uuid.uuid4())
    return state[_SESSION_KEY]


def log_event(event: str, **fields: Any) -> None:
    """Emit a structured event line. Session-id auto-attached inside Streamlit.

    Format is a single log line with a JSON payload so downstream log shippers
    (HF Spaces, GCP, anything that splits on newlines) can parse it cleanly
    without multiline joins.

        2026-04-17 12:03:41 INFO  app.event: view_selected {"view":"shot-map","session":"…"}
    """
    payload: dict[str, Any] = {"session": get_session_id(), **fields}
    try:
        msg = json.dumps(payload, default=str, sort_keys=True)
    except (TypeError, ValueError):
        # Shouldn't happen with default=str, but never let logging crash the app.
        msg = repr(payload)
    logging.getLogger("app.event").info("%s %s", event, msg)


# =============================================================================
# Streamlit error boundary
# =============================================================================

@contextmanager
def view_boundary(name: str) -> Iterator[None]:
    """Context manager: catch any Exception inside a Streamlit view body.

    Usage (inside `app.py`):

        if view == "📍 Shot Map":
            with view_boundary("shot-map"):
                render_shot_map(...)      # any exception → friendly st.error

    Behaviour on exception:
      - log at ERROR with session id + view name + traceback
      - render a `st.error` box so the rest of the app (sidebar, header,
        other collapsed views) remains usable
      - swallow the exception so Streamlit doesn't blank the page

    Behaviour on success: no-op.
    """
    log = logging.getLogger("app.view")
    try:
        log_event("view_entered", view=name)
        yield
    except Exception as exc:  # noqa: BLE001 — this is literally a catch-all
        session = get_session_id()
        log.exception("view %s crashed (session=%s)", name, session)
        log_event(
            "view_error",
            view=name,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        try:
            import streamlit as st  # noqa: WPS433
            st.error(
                f"**Something went wrong rendering the '{name}' view.**\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                f"The rest of the app is still working — switch views in the "
                f"sidebar to continue. If this is reproducible, include "
                f"session id `{session}` in your bug report."
            )
        except ImportError:
            # Running under pytest / CLI — re-raise so tests see the failure.
            raise


__all__ = [
    "configure_logging",
    "get_logger",
    "get_session_id",
    "log_event",
    "view_boundary",
]
