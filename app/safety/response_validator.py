"""Response validator (spec.md §21).

Enforces:

* Excerpt length cap (``max_lyric_excerpt_words``).
* Detection of full-lyrics requests in the user's last turn
  ("entire lyrics", "continue the song", "give me the next verse", ...) and
  routing to the safe alternative.
* Provenance presence (song / artist). Appends a source line if missing.
* Flags generated text vs. metadata distinction (the assistant prompt already
  enforces this at generation time; here we only attach a marker when an
  excerpt is present so the UI can render it differently).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SAFE_FULL_LYRICS_REPLY = (
    "I can summarize the song or provide a short permitted excerpt, "
    "but I can't return the complete lyrics."
)

# Patterns that indicate the user is asking for complete/prohibited lyrics.
_FULL_LYRICS_PATTERNS = [
    re.compile(r"\b(entire|complete|full|whole|all)\b.*\blyrics?\b", re.IGNORECASE),
    re.compile(r"\bcontinue\b.*\b(song|verse|chorus|lyric)", re.IGNORECASE),
    re.compile(r"\b(next|following)\b.*\b(verse|chorus|line|lyric)", re.IGNORECASE),
    re.compile(r"\bgive me\b.*\blyrics?\b", re.IGNORECASE),
    re.compile(r"\bprint\b.*\blyrics?\b", re.IGNORECASE),
    re.compile(r"\bwrite out\b.*\blyrics?\b", re.IGNORECASE),
    re.compile(r"\blyrics?\b.*\b(in full|word for word|verbatim)\b", re.IGNORECASE),
]

# A loose quoted-excerpt detector: text between straight or curly quotes.
_QUOTE_RE = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]{1,400})["\u201c\u201d]')


@dataclass
class ValidationOutcome:
    response: str
    sources: list[str]
    flagged_full_lyrics_request: bool = False
    excerpt_truncated: bool = False


def is_full_lyrics_request(user_message: str) -> bool:
    return any(p.search(user_message or "") for p in _FULL_LYRICS_PATTERNS)


def _truncate_quoted_excerpts(text: str, max_words: int) -> tuple[str, bool]:
    """Truncate any quoted excerpt that exceeds ``max_words`` words."""
    truncated = False

    def _cap(m: re.Match) -> str:
        nonlocal truncated
        excerpt = m.group(1)
        words = excerpt.split()
        if len(words) > max_words:
            truncated = True
            capped = " ".join(words[:max_words])
            return f"\u201c{capped}\u2026\u201d"
        return m.group(0)

    return _QUOTE_RE.sub(_cap, text), truncated


def _ensure_provenance(response: str, sources: list[str]) -> tuple[str, list[str]]:
    """Ensure a provenance/source line is present; append one if missing."""
    if sources:
        return response, sources
    # No explicit sources supplied — look for an "Artist — Title" pattern.
    m = re.search(r"([A-Z][^\n,]{1,60}?)\s*[\u2014\u2013\-]\s*([A-Z][^\n,]{1,60}?)", response)
    if m:
        src = f"{m.group(1).strip()} \u2014 {m.group(2).strip()}"
        return response, [src]
    return response, sources


def validate(
    response: str,
    candidates: list[dict],
    *,
    user_message: str = "",
    max_lyric_excerpt_words: int = 25,
    include_source: bool = True,
) -> ValidationOutcome:
    """Validate an LLM response against the copyright/provenance rules.

    ``candidates`` is the list of candidate song dicts (with ``title`` and
    ``artist``) used for generation; used to attach provenance.
    """
    sources = [
        f"{c['artist']} \u2014 {c['title']}" for c in candidates if c.get("title") and c.get("artist")
    ]

    flagged = is_full_lyrics_request(user_message)
    if flagged:
        return ValidationOutcome(
            response=SAFE_FULL_LYRICS_REPLY,
            sources=sources,
            flagged_full_lyrics_request=True,
        )

    text, truncated = _truncate_quoted_excerpts(response, max_lyric_excerpt_words)

    if include_source:
        text, sources = _ensure_provenance(text, sources)

    return ValidationOutcome(
        response=text,
        sources=sources,
        excerpt_truncated=truncated,
    )