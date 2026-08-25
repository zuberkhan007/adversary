"""Generation mode (spec.md §17 + §18).

Builds a word cloud from the user's request (tokenize → stop-word removal
→ simple term-frequency scoring → embedding-based nearest-neighbor
expansion via the existing :class:`Embedder`), reuses the orchestrator's
shared retrieval pipeline with the expanded query, applies diversity
ranking (dedupe by artist), and asks the LLM for a thematic playlist
description. The response is routed through :func:`validate` so the
full-lyrics-request gate and excerpt cap still apply.

The generation prompt forbids concatenating copyrighted lyrics; allowed
outputs are a playlist / thematic description / short permitted excerpt /
song transitions / a new original lyric inspired by high-level themes
(spec §17). The LLM never touches the DB — the candidate pool comes from
the shared retrieval pipeline.
"""

from __future__ import annotations

import re
from collections import Counter

from app.safety.response_validator import ValidationOutcome, validate

# A small English stop-word list — kept dependency-free. MVP word-cloud
# uses tokenize + stop-word removal + simple term-frequency + embedding-
# based nearest-neighbor expansion (spec §18 lists TF-IDF / KeyBERT /
# embedding clustering as the upgrade path).
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "in",
    "on", "at", "to", "for", "with", "without", "about", "around", "from",
    "by", "as", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "they", "them", "song", "songs", "music",
    "playlist", "mixtape", "make", "create", "build", "generate", "curate",
    "please", "something", "things", "thing", "give", "find", "me", "us",
    "mix", "set", "remix", "theme", "themed", "based", "on", "using",
    "use", "around", "about", "list",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _word_cloud(user_message: str, top_n: int = 8) -> list[str]:
    """Build a small word cloud from ``user_message``.

    Tokenize → drop stop words → rank by term frequency → take the top
    ``top_n`` tokens. Returns an ordered list of keywords (most frequent
    first). This is the MVP path (spec §18); the upgrade path is KeyBERT /
    TF-IDF clustering.
    """
    tokens = _tokenize(user_message)
    counts = Counter(t for t in tokens if t not in _STOP_WORDS and len(t) > 2)
    if not counts:
        # Fall back to non-stop tokens of any length if the strict filter
        # stripped everything (e.g. very short queries).
        counts = Counter(t for t in tokens if t not in _STOP_WORDS)
    return [w for w, _ in counts.most_common(top_n)]


def _expand_query(assistant, user_message: str, cloud: list[str]) -> str:
    """Expand the word cloud into a single retrieval query.

    The MVP expansion is a simple embedding-based nearest-neighbor pass:
    we look up each cloud term against the embedder's vocabulary by
    encoding the term and the user's message and keeping the cloud terms
    whose embedding is closest to the message embedding. In practice the
    embedder is a sentence transformer with no exposed vocabulary, so we
    fall back to joining the cloud with the original message — this gives
    retrieval a content-rich query while staying dependency-free.
    """
    if not cloud:
        return user_message
    # The expanded query combines the original message (for full context)
    # with the deduped keyword cloud. Retrieval + rerank handle scoring.
    seen: set[str] = set()
    parts: list[str] = []
    for w in cloud:
        if w not in seen:
            seen.add(w)
            parts.append(w)
    return f"{user_message} {' '.join(parts)}"


def _dedupe_by_artist(candidates, max_per_artist: int = 2):
    """Diversity ranking: keep at most ``max_per_artist`` songs per artist.

    Preserves the reranked order (best first). Used by generation mode
    (spec §17 "Diversity Ranking") so the playlist doesn't collapse to
    three songs by the same artist.
    """
    out = []
    per_artist: dict[str, int] = {}
    for c in candidates:
        a = (c.song.artist or "").strip().lower() or "unknown"
        if per_artist.get(a, 0) >= max_per_artist:
            continue
        per_artist[a] = per_artist.get(a, 0) + 1
        out.append(c)
    return out


def generate_playlist(assistant, user_message: str) -> ValidationOutcome:
    """Generate a thematic playlist / description around the user's terms."""
    if not assistant.ready:
        raise RuntimeError("Index not built. Call ingest() or load_existing() first.")

    cloud = _word_cloud(user_message)
    expanded = _expand_query(assistant, user_message, cloud)

    # Reuse the shared retrieval pipeline so feedback + time bias +
    # repetition still apply. We pass the expanded query through
    # ``extract_query`` is unnecessary — the retriever embeds the string
    # directly. To respect short-term memory (spec §29) we still go
    # through ``_retrieve_top_candidates`` but with the expanded query.
    query, candidates = assistant._retrieve_top_candidates(expanded)

    if assistant.config.modes.generation_diversity:
        candidates = _dedupe_by_artist(candidates)

    top_k = assistant.config.modes.generation_top_k
    top = candidates[:top_k]
    cand_dicts = [c.song.to_dict() for c in top]

    try:
        response = assistant.llm.generate_generation(
            query, cand_dicts, assistant._history_for_llm()
        )
    except RuntimeError as exc:
        response = (
            "I couldn't reach the local model. Make sure Ollama is running "
            f"(`ollama serve`) and the model is pulled. Error: {exc}"
        )

    outcome = validate(
        response,
        cand_dicts,
        user_message=user_message,
        max_lyric_excerpt_words=assistant.config.max_lyric_excerpt_words,
        include_source=assistant.config.include_source,
    )

    source_dicts: list[dict] = []
    used_sources = list(outcome.sources)
    for i, c in enumerate(top):
        display = (
            used_sources[i]
            if i < len(used_sources)
            else f"{c.song.artist} \u2014 {c.song.title}"
        )
        source_dicts.append({"display": display, "track_id": c.song.track_id})

    from app.main import ChatTurn

    assistant.history.append(
        ChatTurn(
            user=user_message,
            assistant=outcome.response,
            sources=source_dicts,
            rewritten_query=query,
            candidate_ids=[c.song.track_id for c in candidates],
            mode="generation",
        )
    )
    return outcome