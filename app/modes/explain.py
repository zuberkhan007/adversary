"""Explain mode (spec.md §15).

Explains a concept using songs from the user's collection as contextual
examples. Reuses the orchestrator's shared retrieval pipeline (so
feedback + time bias + repetition still apply) and grounds the LLM in
candidate metadata + short licensed lyric phrases (when a lyrics
provider is configured) — the "LLM never touches DBs" invariant holds:
lyrics are fetched via ``lyrics-mcp``, not by the LLM.

The LLM is called with a low-temperature explain prompt that names the
concept and the candidate songs (each with its lyric phrases). It
produces one paragraph per song weaving the lyric phrases as quoted
lines (original language) with minimal English conversational text,
ending each paragraph with an ``Artist — Title`` citation. The response
is routed through :func:`validate` so the full-lyrics-request gate and
excerpt cap still apply.
"""

from __future__ import annotations

from app.lyrics.base import cap_words, select_relevant_phrases
from app.mcp import lyrics_server
from app.safety.response_validator import ValidationOutcome, validate


def explain(assistant, user_message: str) -> ValidationOutcome:
    """Explain a concept using songs from the collection as examples."""
    if not assistant.ready:
        raise RuntimeError("Index not built. Call ingest() or load_existing() first.")

    query, candidates = assistant._retrieve_top_candidates(user_message)
    top = candidates[: assistant.config.rerank_k]
    cand_dicts = [c.song.to_dict() for c in top]

    # Gather short licensed lyric phrases per candidate song (multi-song,
    # like chat_lyric). Each phrase is individually capped at
    # ``max_lyric_excerpt_words`` (copyright invariant, spec §21: we return
    # multiple short excerpts, not one long one). Best-effort: any failure
    # (no provider, 404, network) leaves that song with no phrases and the
    # LLM writes a prose explanation for it instead.
    cap = assistant.config.max_lyric_excerpt_words
    pool_words = max(200, cap * 4)
    phrases_per_song = 3
    max_songs = 3
    song_phrases: list[list[str]] = []
    collected_songs: list = []  # songs that actually contributed phrases
    for i, c in enumerate(top):
        phrases: list[str] = []
        if assistant.lyrics_provider is not None and len(collected_songs) < max_songs:
            try:
                res = lyrics_server.get_excerpt(
                    c.song.track_id, max_words=pool_words
                )
            except Exception:
                res = {}
            if res.get("available"):
                pool_excerpt = res.get("excerpt", "")
                phrases = select_relevant_phrases(
                    pool_excerpt, user_message, cap,
                    n=phrases_per_song, embedder=assistant.embedder,
                )
                if not phrases:
                    fallback = cap_words(pool_excerpt, cap)
                    phrases = [fallback] if fallback else []
        song_phrases.append(phrases)
        if phrases:
            collected_songs.append(c.song)

    try:
        response = assistant.llm.generate_explain(
            query,
            cand_dicts,
            assistant._history_for_llm(),
            song_phrases=song_phrases,
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

    # Build structured sources (display + track_id) for the UI feedback
    # buttons — same pattern as :meth:`MusicAssistant.chat`. Iterate over
    # the top candidates so the index aligns with ``outcome.sources``
    # (which is built from the full candidate list).
    source_dicts: list[dict] = []
    used_sources = list(outcome.sources)
    for i, c in enumerate(top):
        display = (
            used_sources[i]
            if i < len(used_sources)
            else f"{c.song.artist} \u2014 {c.song.title}"
        )
        source_dicts.append({"display": display, "track_id": c.song.track_id})

    assistant.history.append(
        _new_turn(
            user=user_message,
            assistant=outcome.response,
            sources=source_dicts,
            rewritten_query=query,
            candidate_ids=[c.song.track_id for c in candidates],
            mode="explain",
        )
    )
    return outcome


def _new_turn(user, assistant, sources, rewritten_query, candidate_ids, mode):
    # Local import to avoid a circular import at module load time.
    from app.main import ChatTurn

    return ChatTurn(
        user=user,
        assistant=assistant,
        sources=sources,
        rewritten_query=rewritten_query,
        candidate_ids=candidate_ids,
        mode=mode,
    )