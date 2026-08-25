"""Interaction modes (Phase 4): intent router + Explain / Antakshari /
Generation handlers (spec.md §14–17).

The ``general`` mode is the existing Phase 3 lyric-first chat path
(:meth:`MusicAssistant.chat_lyric`) and lives on the orchestrator; this
package only adds the three new modes plus the router. The "LLM never
touches DBs" invariant holds: each handler reads the store via the
orchestrator's MCP-backed helpers / the shared retrieval pipeline, never
the store directly from a chat path.
"""

from __future__ import annotations

from app.modes.antakshari import AntakshariSession
from app.modes.explain import explain
from app.modes.generation import generate_playlist
from app.modes.list_indexed import list_indexed
from app.modes.router import Intent, detect_intent, detect_intent_keyword, detect_intent_llm

__all__ = [
    "Intent",
    "detect_intent",
    "detect_intent_keyword",
    "detect_intent_llm",
    "explain",
    "generate_playlist",
    "list_indexed",
    "AntakshariSession",
]