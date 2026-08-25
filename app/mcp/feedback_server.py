"""feedback-mcp — feedback persistence tools (spec.md §12, §31).

Plain tool functions delegate to a module-level :class:`FeedbackStore` set
via :func:`set_store` from the orchestrator. Each function is also registered
as an MCP tool via ``@mcp.tool()`` for future stdio/HTTP exposure; the plain
function remains directly importable for in-process calls.

Input validation (spec §31): ``track_id`` must be a non-empty string.
"""

from __future__ import annotations

from typing import Optional

from app.mcp import _make_mcp
from app.personalization.feedback import FeedbackStore

mcp = _make_mcp("feedback-mcp")

_store: Optional[FeedbackStore] = None


def set_store(store: FeedbackStore) -> None:
    """Inject the FeedbackStore used by the tool functions. Called once by the
    orchestrator at startup."""
    global _store
    _store = store


def _require_store() -> FeedbackStore:
    if _store is None:
        raise RuntimeError("feedback-mcp: store not initialized. Call set_store() first.")
    return _store


def _validate_track_id(track_id) -> str:
    if not isinstance(track_id, str) or not track_id:
        raise ValueError("track_id must be a non-empty string")
    return track_id


def record_play(track_id: str, completion_ratio: float = 1.0) -> dict:
    """Record a play event. Returns ``{ok, track_id, plays, completion_ratio}``."""
    tid = _validate_track_id(track_id)
    row = _require_store().record_play(tid, completion_ratio=completion_ratio)
    return {
        "ok": True,
        "track_id": row.track_id,
        "plays": row.plays,
        "completion_ratio": row.completion_ratio,
    }


def record_skip(track_id: str) -> dict:
    """Record a skip event. Returns ``{ok, track_id, skips}``."""
    tid = _validate_track_id(track_id)
    row = _require_store().record_skip(tid)
    return {"ok": True, "track_id": row.track_id, "skips": row.skips}


def record_like(track_id: str) -> dict:
    """Record a like event. Returns ``{ok, track_id, likes}``."""
    tid = _validate_track_id(track_id)
    row = _require_store().record_like(tid)
    return {"ok": True, "track_id": row.track_id, "likes": row.likes}


def get_preference_score(track_id: str) -> dict:
    """Return the raw preference score and raw counts for a track."""
    tid = _validate_track_id(track_id)
    store = _require_store()
    row = store.get(tid)
    if row is None:
        return {"ok": True, "track_id": tid, "plays": 0, "skips": 0, "likes": 0,
                "completion_ratio": 0.0, "last_played": "", "preference_score": 0.0}
    # Lazy import to avoid a circular dependency at module import time
    # (config is light, but keep the call site explicit).
    from app.config import FeedbackWeights
    score = store.preference_score(tid, FeedbackWeights())
    return {
        "ok": True,
        "track_id": tid,
        "plays": row.plays,
        "skips": row.skips,
        "likes": row.likes,
        "completion_ratio": row.completion_ratio,
        "last_played": row.last_played,
        "preference_score": score,
    }


def get_history(limit: int = 50) -> dict:
    """Return up to ``limit`` feedback rows as a list of dicts."""
    rows = list(_require_store().all_rows().values())
    rows.sort(key=lambda r: r.last_played or "", reverse=True)
    limited = rows[:limit]
    return {
        "ok": True,
        "rows": [
            {
                "track_id": r.track_id,
                "plays": r.plays,
                "skips": r.skips,
                "likes": r.likes,
                "completion_ratio": r.completion_ratio,
                "last_played": r.last_played,
            }
            for r in limited
        ],
    }


# ---- FastMCP tool registration (no-op if SDK unavailable) --------------
# Registration is best-effort: if the mcp SDK's introspection rejects the
# function signatures (e.g. stringified annotations from `__future__`), the
# plain functions above still work for in-process orchestration.
if mcp is not None:
    for _fn in (record_play, record_skip, record_like, get_preference_score, get_history):
        try:
            mcp.tool()(_fn)
        except Exception:
            pass