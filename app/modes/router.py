"""Intent router (spec.md §14).

Primary path: LLM-based ``detect_intent_llm`` on the configured
``LLMClient``. Fallback: ``detect_intent_keyword`` — a small rule set that
encodes the spec §27 examples so the pipeline still routes when Ollama is
unreachable or returns garbage. Anything the router can't confidently
classify falls back to ``Intent.GENERAL`` (the existing Phase 3 lyric-first
chat path).
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    EXPLAIN = "explain"
    ANTAKSHARI = "antakshari"
    GENERATION = "generation"
    LIST = "list"
    GENERAL = "general"


# Canonical labels the LLM is prompted to return. We map a free-form LLM
# reply onto one of these by substring match (case-insensitive). Note:
# "list" must be matched as a whole word to avoid colliding with "explain"
# (which contains no "list") — but "playlist" contains... no, it doesn't.
# Order matters: "list" is checked before "general"/"chat".
_CANONICAL = {
    "explain": Intent.EXPLAIN,
    "antakshari": Intent.ANTAKSHARI,
    "generation": Intent.GENERATION,
    "generate": Intent.GENERATION,
    "playlist": Intent.GENERATION,
    "list": Intent.LIST,
    "general": Intent.GENERAL,
    "default": Intent.GENERAL,
    "chat": Intent.GENERAL,
}


def detect_intent_llm(llm, user_message: str, history: list[dict] | None = None) -> Intent | None:
    """LLM-based intent detection. Returns ``None`` on any failure (the
    caller falls back to the keyword rule)."""
    try:
        raw = llm.detect_intent(user_message, history)
    except (RuntimeError, AttributeError, TypeError):
        return None
    if not raw:
        return None
    text = str(raw).strip().lower()
    # Map by substring: pick the canonical label that appears in the reply.
    for key, intent in _CANONICAL.items():
        if key in text:
            return intent
    return None


# Keyword fallback rules — encode the spec §27 examples. Order matters:
# the first match wins. ``antakshari`` is checked before ``explain``/generic
# ``play`` because "let's play antakshari" contains both.
_ANTAKSHARI_RE = re.compile(
    r"\b(antakshari|let'?s play|play a game|start a game)\b", re.IGNORECASE
)
_LIST_RE = re.compile(
    r"\b(list|show|display|enumerate)\s+(all|every|the|my|these|those)\s+((my|the|those|these)\s+)?(indexed\s+)?(songs?|tracks?|collection)\b"
    r"|\b(all|every|whole)\s+(the\s+|my\s+)?(songs?|tracks?)\s+(in|of|indexed)\b"
    r"|\bindexed\s+songs?\b"
    r"|\b(my\s+collection|whole\s+collection|entire\s+collection|all\s+indexed\s+songs?)\b"
    r"|\bwhat\b.*\bindexed\b",
    re.IGNORECASE,
)
_EXPLAIN_RE = re.compile(
    r"\b(explain|explaination|elaborate|what does .* mean|meaning of)\b",
    re.IGNORECASE,
)
_GENERATION_RE = re.compile(
    r"\b(make|create|build|generate|curate)\b.*\b(playlist|mixtape|set|mix|remix)\b"
    r"|\bplaylist\b.*\b(around|about|theme|based on)\b"
    r"|\bcreate something around\b"
    r"|\bword cloud\b",
    re.IGNORECASE,
)


def detect_intent_keyword(user_message: str) -> Intent:
    """Rule-based intent detection (spec §27 examples). Used as a fallback
    when the LLM is unreachable or returns unparseable output."""
    msg = user_message or ""
    if _ANTAKSHARI_RE.search(msg):
        return Intent.ANTAKSHARI
    if _LIST_RE.search(msg):
        return Intent.LIST
    if _GENERATION_RE.search(msg):
        return Intent.GENERATION
    if _EXPLAIN_RE.search(msg):
        return Intent.EXPLAIN
    return Intent.GENERAL


def detect_intent(
    llm, user_message: str, history: list[dict] | None = None
) -> Intent:
    """LLM-then-keyword intent detection (spec §14).

    Falls back to the keyword rule when the LLM is unreachable or returns
    garbage. Always returns an :class:`Intent` (never raises).
    """
    intent = detect_intent_llm(llm, user_message, history)
    if intent is not None:
        return intent
    return detect_intent_keyword(user_message)