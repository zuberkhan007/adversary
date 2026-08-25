"""List mode: enumerate every indexed song in the collection.

Triggered by intent ``Intent.LIST`` (e.g. "list all the songs indexed",
"show all my songs", "what's in my collection"). The orchestrator pulls the
full song list from the store and passes it to the LLM, which formats it
into a clean numbered response. The "LLM never touches DBs" invariant
holds: the handler reads the store via ``assistant.store.all_songs()``
(the same access pattern as the existing MCP-backed paths and the
Antakshari mode) and passes candidate metadata into ``generate_list``;
the LLM never pulls songs itself.

The response IS the formatted list with a short count line. No per-song
``ChatTurn.sources`` are attached \u2014 the list itself is the provenance,
and rendering a Play/Skip/Like/preview/lyrics row per song under the chat
message would be impractical for a large collection. The right-sidebar
"Indexed songs" panel already shows the full collection for interactive
play/preview/lyrics, so the chat response stays clean.
"""

from __future__ import annotations

from app.safety.response_validator import ValidationOutcome, validate

# Soft cap on how many songs are passed to the LLM in one list response.
# Protects the LLM context window for very large collections; the handler
# appends a truncation note when the cap is hit. The right-sidebar panel
# still shows the full unbounded list.
_LIST_CAP = 100


def list_indexed(assistant, user_message: str) -> ValidationOutcome:
    """List every indexed song, formatted by the LLM."""
    if not assistant.ready:
        raise RuntimeError("Index not built. Call ingest() or load_existing() first.")

    songs = assistant.store.all_songs()
    if not songs:
        response = "Your collection is empty \u2014 ingest a playlist to add songs."
        outcome = ValidationOutcome(response=response, sources=[])
        assistant.history.append(
            _new_turn(
                user=user_message,
                assistant=outcome.response,
                sources=[],
                rewritten_query="",
                candidate_ids=[],
                mode="list",
            )
        )
        return outcome

    truncated = len(songs) > _LIST_CAP
    subset = songs[:_LIST_CAP]
    cand_dicts = [s.to_dict() for s in subset]

    try:
        response = assistant.llm.generate_list(
            user_message,
            cand_dicts,
            assistant._history_for_llm(),
        )
    except RuntimeError as exc:
        response = (
            "I couldn't reach the local model. Make sure Ollama is running "
            f"(`ollama serve`) and the model is pulled. Error: {exc}"
        )

    if truncated:
        response = (
            f"{response}\n\n"
            f"(Showing the first {_LIST_CAP} of {len(songs)} songs. The "
            "right-side panel lists them all.)"
        )

    outcome = validate(
        response,
        cand_dicts,
        user_message=user_message,
        max_lyric_excerpt_words=assistant.config.max_lyric_excerpt_words,
        include_source=assistant.config.include_source,
    )

    assistant.history.append(
        _new_turn(
            user=user_message,
            assistant=outcome.response,
            # Empty sources: the list IS the provenance, and per-song
            # buttons under the chat message would be impractical for a
            # large collection. The right-sidebar panel handles
            # play/preview/lyrics interaction.
            sources=[],
            rewritten_query="",
            candidate_ids=[s.track_id for s in subset],
            mode="list",
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