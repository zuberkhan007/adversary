"""In-process MCP tool registry (spec.md §12, §31).

Each ``*_server.py`` module exposes plain Python tool functions that the
orchestrator (:class:`app.main.MusicAssistant`) calls directly. The same
functions are also registered with a ``FastMCP`` instance from the ``mcp``
SDK so they can be exposed over stdio/HTTP in a future phase. The plain
functions remain directly importable for in-process calls, so the
orchestrator path is unaffected if the ``mcp`` SDK is unavailable.

Implemented in Phase 2: ``feedback-mcp``, ``rag-mcp``.
Stubbed for Phase 3: ``music-mcp``, ``lyrics-mcp``.
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except Exception:  # pragma: no cover - mcp is optional at runtime
    FastMCP = None  # type: ignore[assignment]


def _make_mcp(name: str):
    """Return a ``FastMCP(name)`` if the SDK is available, else ``None``.

    The fallback keeps the rest of the package importable on environments
    where ``pip install mcp`` failed (per the Phase 2 plan's Windows risk
    note). The plain tool functions still work in-process.
    """
    if FastMCP is None:
        return None
    return FastMCP(name)