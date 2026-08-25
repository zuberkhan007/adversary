"""Lyrics provider Protocol + shared excerpt utilities (spec.md §21).

The excerpt cap (``max_lyric_excerpt_words``, default 25) and provenance
attachment are enforced at the provider layer so every implementation
honors them. ``syncedLyrics`` (LRC format with ``[mm:ss.xx]`` timing tags)
is always stripped to plain text before being returned to the UI — we never
return raw synced lyrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LyricsExcerpt:
    """A lyrics lookup result.

    ``available=False`` means the provider had no match. ``excerpt`` is the
    plain-text, length-capped snippet. ``is_synced`` indicates the source
    was LRC ``syncedLyrics`` (we still strip the timing tags before exposing
    text). ``source_url`` is the provider's page/track URL for provenance.
    """

    track_id: str
    available: bool
    excerpt: str
    provider: str
    source_url: str
    is_synced: bool = False


class LyricsProvider(Protocol):
    """Pluggable lyrics provider interface."""

    def get(
        self,
        title: str,
        artist: str,
        album: str = "",
        duration: int | None = None,
        max_words: int = 25,
    ) -> LyricsExcerpt | None:
        ...


# LRC timing tag: ``[00:17.12]`` or ``[01:02:03.456]``.
_LRC_TAG_RE = re.compile(r"\[\d{1,2}:\d{1,2}(?:\.\d+)?\]\s*")


def strip_lrc(lrc: str) -> str:
    """Remove LRC ``[mm:ss.xx]`` timing tags and collapse extra whitespace."""
    if not lrc:
        return ""
    plain = _LRC_TAG_RE.sub("", lrc)
    # Collapse runs of whitespace into a single space; preserve line breaks
    # by splitting on newlines first, then re-joining.
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in plain.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def cap_words(text: str, max_words: int) -> str:
    """Truncate ``text`` to at most ``max_words`` words, appending ``…``.

    Word splitting is whitespace-based. ``max_words <= 0`` returns ``""``.
    """
    if max_words <= 0 or not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "\u2026"


def cap_lines(text: str, max_words: int) -> str:
    """Truncate ``text`` to at most ``max_words`` words, preserving line breaks.

    Unlike :func:`cap_words` (which collapses all whitespace to single
    spaces), this keeps the original line structure so downstream phrase
    splitting (:func:`_split_phrases`) can break the result into individual
    lyric lines. ``max_words <= 0`` returns ``""``.
    """
    if max_words <= 0 or not text:
        return ""
    lines = text.splitlines()
    out: list[str] = []
    count = 0
    for ln in lines:
        words = ln.split()
        if not words:
            continue
        if count + len(words) > max_words:
            remaining = max_words - count
            if remaining > 0:
                out.append(" ".join(words[:remaining]) + "\u2026")
            break
        out.append(ln)
        count += len(words)
    return "\n".join(ln for ln in out if ln.strip()).strip()


# Sentence / phrase boundary splitter for relevant-phrase selection.
_PHRASE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|;\s+")


def _split_phrases(text: str) -> list[str]:
    """Split ``text`` into phrases by newlines, sentence boundaries, and
    semicolons. Returns non-empty stripped phrases."""
    if not text:
        return []
    parts = _PHRASE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _keyword_overlap_score(phrase: str, query_words: set[str]) -> float:
    """Score a phrase by keyword overlap with the query (case-insensitive)."""
    if not phrase or not query_words:
        return 0.0
    pwords = set(w.lower() for w in re.findall(r"\w+", phrase))
    if not pwords:
        return 0.0
    overlap = len(query_words & pwords)
    # Normalize by phrase length so short phrases with many matches win.
    length_penalty = max(0.0, 1.0 - len(pwords) / 40.0)
    return overlap * (1.0 + length_penalty)


def _word_count(phrase: str) -> int:
    """Word count of a phrase (whitespace-split)."""
    return len(phrase.split()) if phrase else 0


def _meaningfulness_bonus(phrase: str) -> float:
    """A small bonus for phrases that are complete, meaningful lines.

    Phrases shorter than 3 words rarely carry a complete thought on
    their own and are penalized; phrases of 4+ words get a small positive
    bonus (capped) so a complete line wins over a tiny fragment when
    semantic similarity is close.
    """
    n = _word_count(phrase)
    if n < 3:
        return -0.5
    if n < 4:
        return 0.0
    return min(0.1, 0.02 * (n - 3))


def _rank_phrases(
    phrases: list[str], query: str, embedder=None
) -> list[str]:
    """Return ``phrases`` sorted by relevance to ``query`` (most relevant first).

    Combines embedding cosine similarity (when an ``embedder`` is provided)
    with keyword overlap, plus a meaningfulness bonus that biases toward
    complete, meaningful lines over tiny fragments. Blending the two
    signals keeps keyword-rich phrases winning even when the embedder is
    noisy or returns hash-collision artifacts.
    """
    if not phrases:
        return []
    query_words = set(w.lower() for w in re.findall(r"\w+", query or ""))
    kw_scores = [_keyword_overlap_score(p, query_words) for p in phrases]
    mb = [_meaningfulness_bonus(p) for p in phrases]

    # Primary: embedding similarity blended with keyword overlap.
    if embedder is not None:
        try:
            import numpy as np

            texts = [query] + phrases
            vecs = embedder.encode(texts)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            qv = vecs[0].astype("float32")
            pv = vecs[1:].astype("float32")
            qn = np.linalg.norm(qv) or 1.0
            pn = np.linalg.norm(pv, axis=1)
            pn[pn == 0] = 1.0
            sims = (pv @ qv) / (pn * qn)
            scored = sorted(
                range(len(phrases)),
                key=lambda i: float(sims[i]) + kw_scores[i] + mb[i],
                reverse=True,
            )
            return [phrases[i] for i in scored]
        except Exception:
            pass  # fall back to keyword overlap

    # Fallback: keyword overlap (+ meaningfulness bonus).
    order = sorted(
        range(len(phrases)),
        key=lambda i: kw_scores[i] + mb[i],
        reverse=True,
    )
    return [phrases[i] for i in order]


def select_relevant_phrase(
    lyrics: str,
    query: str,
    max_words: int,
    embedder=None,
) -> str:
    """Select the single phrase from ``lyrics`` most relevant to ``query``.

    The selected phrase is capped to ``max_words`` via :func:`cap_words`.
    The full ``lyrics`` text is used ONLY for phrase selection — only the
    selected ``max_words``-capped phrase is returned (copyright invariant,
    spec §21: never return large portions of copyrighted lyrics).
    """
    if not lyrics or max_words <= 0:
        return ""
    phrases = _split_phrases(lyrics)
    if not phrases:
        return cap_words(lyrics, max_words)
    if len(phrases) == 1:
        return cap_words(phrases[0], max_words)
    ranked = _rank_phrases(phrases, query, embedder=embedder)
    return cap_words(ranked[0], max_words) if ranked else cap_words(phrases[0], max_words)


def select_relevant_phrases(
    lyrics: str,
    query: str,
    max_words_per_phrase: int,
    n: int = 3,
    embedder=None,
) -> list[str]:
    """Select the top ``n`` phrases from ``lyrics`` most relevant to ``query``.

    Each phrase is capped to ``max_words_per_phrase`` via :func:`cap_words`.
    Phrases are deduped (case-insensitive) so the LLM gets distinct lines to
    weave. Returns at most ``n`` phrases (fewer if the lyric has fewer
    distinct phrases). Copyright invariant (spec §21): each phrase is
    individually capped; the caller controls how many are exposed.
    """
    if not lyrics or max_words_per_phrase <= 0 or n <= 0:
        return []
    phrases = _split_phrases(lyrics)
    if not phrases:
        return []
    if len(phrases) == 1:
        capped = cap_words(phrases[0], max_words_per_phrase)
        return [capped] if capped else []
    ranked = _rank_phrases(phrases, query, embedder=embedder)
    out: list[str] = []
    seen: set[str] = set()
    for p in ranked:
        capped = cap_words(p, max_words_per_phrase)
        if not capped:
            continue
        key = capped.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(capped)
        if len(out) >= n:
            break
    return out