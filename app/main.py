"""MusicAssistant — Phase 1 + Phase 2 + Phase 3 orchestration.

Pipeline (spec.md §35, simplified)::

    user_message
      -> extract_query (Gemma, with short-term memory)
      -> retrieve (semantic top-k)
      -> rerank (semantic + keyword)
      -> finalize_scores (+ feedback + time - repetition)
      -> generate (Gemma, grounded in candidates)
      -> validate (excerpt cap, full-lyrics detection, provenance)

Phase 2 additions:
  - ``FeedbackStore`` is constructed in ``__init__`` and injected into
    ``app.mcp.feedback_server`` so MCP tool functions have their dependency.
  - ``app.mcp.rag_server`` is wired with the retriever + ranking weights.
  - Per-song ``record_play`` / ``record_skip`` / ``record_like`` delegate to
    the ``feedback-mcp`` tool functions (the orchestrator never touches the
    DB directly except through MCP tools — spec §12 invariant holds at the
    orchestration layer).
  - ``set_time_bias`` updates ``personalization.default_time_bias``.
  - ``chat()`` threads feedback + time bias + recently-surfaced track ids
    through ``finalize_scores``.

Phase 3 additions:
  - A ``LyricsProvider`` is built from ``config.lyrics.provider`` and
    injected into ``app.mcp.lyrics_server`` along with the store.
  - ``get_lyrics_excerpt(track_id)`` fetches an on-demand excerpt via
    ``lyrics-mcp`` (the LLM never calls this directly — spec §12 invariant
    holds; lyrics are fetched only when the UI asks for them, per spec §21).
  - ``chat()`` is unchanged — songs flow through the same retrieval
    path. The LLM still only sees candidate metadata; no LLM tool-calling
    loop was added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import Config
from app.ingestion.enrichment import Enricher
from app.ingestion.markdown import Song, parse_playlist
from app.ingestion.youtube import YouTubeClient, YouTubeError
from app.llm.gemma import Gemma
from app.lyrics.base import LyricsProvider
from app.lyrics.lrclib import LRCLIBProvider
from app.lyrics.musixmatch import MusixmatchProvider
from app.mcp import feedback_server, lyrics_server, rag_server
from app.modes import (
    AntakshariSession,
    Intent,
    detect_intent,
    explain as explain_mode,
    generate_playlist as generation_mode,
    list_indexed as list_mode,
)
from app.modes import antakshari as antakshari_mode
from app.personalization.feedback import FeedbackStore
from app.personalization.ranking import finalize_scores
from app.rag.embeddings import Embedder
from app.rag.reranker import rerank
from app.rag.retriever import Candidate, Retriever
from app.rag.vector_store import VectorStore
from app.safety.response_validator import (
    SAFE_FULL_LYRICS_REPLY,
    ValidationOutcome,
    is_full_lyrics_request,
    validate,
)
from app.lyrics.base import cap_words, select_relevant_phrases


class LLMClient(Protocol):
    def extract_query(self, user_message: str, history: list[dict] | None = None) -> str: ...
    def generate(self, query: str, candidates: list[dict], history: list[dict] | None = None) -> str: ...
    def generate_lyric_intro(
        self,
        query: str,
        song: dict,
        lyric_excerpt: str,
        history: list[dict] | None = None,
        extra_phrases: list[str] | None = None,
    ) -> str: ...
    def select_song(
        self, query: str, candidates: list[dict], history: list[dict] | None = None
    ) -> str | None: ...
    def generate_tags(self, title: str, artist: str) -> list[str]: ...
    def detect_intent(self, user_message: str, history: list[dict] | None = None) -> str: ...
    def generate_explain(
        self,
        query: str,
        candidates: list[dict],
        history: list[dict] | None = None,
        song_phrases: list[list[str]] | None = None,
    ) -> str: ...
    def generate_generation(
        self, query: str, candidates: list[dict], history: list[dict] | None = None
    ) -> str: ...
    def generate_list(
        self, query: str, candidates: list[dict], history: list[dict] | None = None
    ) -> str: ...


@dataclass
class ChatTurn:
    user: str
    assistant: str
    # Each entry: {"display": "Artist — Title", "track_id": "..."}
    sources: list[dict] = field(default_factory=list)
    rewritten_query: str = ""
    candidate_ids: list[str] = field(default_factory=list)
    # Phase 4: the interaction mode that produced this turn
    # ("general" | "explain" | "antakshari" | "generation"). Additive —
    # existing turns serialize fine since it has a default.
    mode: str = "general"
    # Per-track lyric data used during generation (lyric-first chat mode).
    # Maps ``track_id`` → ``{phrases, provider, source_url, is_synced,
    # romanized}``. Empty for non-lyric modes. The UI's Lyrics button shows
    # these stored phrases instead of fetching a fresh excerpt, so the user
    # sees the exact lyrics that informed the response.
    lyric_data: dict[str, dict] = field(default_factory=dict)


# Common English single-letter words that are NOT letter requests — the
# article "a" and the pronoun "I". Excluded so natural prose like
# "give me a song" doesn't trigger the lyric-letter path on "a".
_LETTER_EXCLUDE = {"a", "I"}


def _detect_letter(message: str) -> str | None:
    """Parse a single letter from a user message intended for the
    Lyric Prose Recommendation mode. Returns the lowercase letter, or
    ``None`` if no clear letter signal is present (in which case the mode
    falls back to the Phase 1 prose recommendation path).

    Recognised signals (checked in order):
    - ``"letter X"`` / ``"X letter"`` (case-insensitive, word-boundaried)
    - ``"starting with X"`` / ``"starts with X"`` / ``"start with X"``
    - ``"beginning with X"`` / ``"begins with X"`` / ``"begin with X"``
    - A message whose only alphabetic character is a single letter
      (optionally surrounded by punctuation/whitespace, e.g. ``"A"`` or
      ``"'b'"``).
    - Any standalone single-letter word in the message, excluding the
      article ``a`` and the pronoun ``I`` — so natural phrases like
      ``"give me songs for K"`` or ``"I want lyrics for Q"`` still hit.
    """
    if not message:
        return None
    msg = message.strip()
    m = re.search(r"\bletter\s+([A-Za-z])\b", msg, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b([A-Za-z])\s+letter\b", msg, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(
        r"\b(?:start|starts|starting)\s+with\s+([A-Za-z])\b",
        msg, flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    m = re.search(
        r"\b(?:begin|begins|beginning)\s+with\s+([A-Za-z])\b",
        msg, flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    # Whole message is essentially a single letter (with optional
    # surrounding punctuation/quotes/whitespace), e.g. "A" or "'b'".
    m = re.fullmatch(r"[^A-Za-z]*([A-Za-z])[^A-Za-z]*", msg)
    if m:
        return m.group(1).lower()
    # Any standalone single-letter word, excluding common English words
    # ("a", "I"). Picks the first non-excluded match.
    for tok in re.findall(r"\b([A-Za-z])\b", msg):
        if tok in _LETTER_EXCLUDE:
            continue
        return tok.lower()
    return None


def _pick_lines_starting_with(
    pool_text: str, letter: str, n_lines: int, max_words: int
) -> str | None:
    """Find the first line in ``pool_text`` whose first *alphabetic*
    character equals ``letter`` (case-insensitive), then return that line
    plus the next ``n_lines - 1`` non-empty lines, joined with ``" / "`` and
    capped at ``max_words`` words. Returns ``None`` when no line matches.

    The matching line can come from anywhere in the song (the opening *or*
    a line "between"/in the middle) — the LRCLIB excerpt is the first
    ``pool_words`` words of the lyrics, so middle lines within that window
    are reachable. Matching on the first alphabetic char skips leading
    punctuation/quotes/LRC timing tags. The returned excerpt is a short
    multi-line snippet (spec §21: short permitted excerpt, capped at
    ``max_words``).
    """
    if not pool_text:
        return None
    target = letter.lower()
    raw_lines = pool_text.splitlines()
    start_idx = None
    for i, raw in enumerate(raw_lines):
        line = raw.strip()
        if not line:
            continue
        first = next((c for c in line if c.isalpha()), None)
        if first is None or first.lower() != target:
            continue
        start_idx = i
        break
    if start_idx is None:
        return None
    picked: list[str] = []
    for raw in raw_lines[start_idx:]:
        line = raw.strip()
        if not line:
            continue
        picked.append(line)
        if len(picked) >= n_lines:
            break
    if not picked:
        return None
    joined = " / ".join(picked)
    return cap_words(joined, max_words)


class MusicAssistant:
    def __init__(
        self,
        config: Config,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        memory_turns: int = 6,
    ) -> None:
        self.config = config
        self.llm: LLMClient = llm if llm is not None else Gemma(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            temperature=config.temperature,
            api_key=config.ollama_api_key,
        )
        self.embedder = embedder if embedder is not None else Embedder(config.embedding_model)
        self.store = store if store is not None else VectorStore(
            faiss_index_path=config.faiss_index_path,
            sqlite_path=config.sqlite_path,
        )
        self.retriever = Retriever(self.store, self.embedder)
        self.memory_turns = memory_turns
        self.history: list[ChatTurn] = []
        self._ingested = False
        # Phase 4: in-memory Antakshari session (not persisted). ``None``
        # when no game is active; a "Stop game" button resets it.
        self.antakshari: AntakshariSession | None = None

        # Phase 2: feedback store + MCP wiring.
        self.feedback_store = FeedbackStore(config.sqlite_path)
        feedback_server.set_store(self.feedback_store)
        rag_server.set_retriever(self.retriever, config.ranking_weights)

        # Phase 3: YouTube client (no credentials required). Construction
        # is wrapped so the orchestrator still loads when yt-dlp isn't
        # installed; the YouTube path is disabled in that case.
        self.youtube_client: YouTubeClient | None = None
        try:
            self.youtube_client = YouTubeClient()
        except YouTubeError:
            self.youtube_client = None

        # Phase 3: lyrics provider + store wiring.
        self.lyrics_provider: LyricsProvider | None = self._build_lyrics_provider()
        lyrics_server.set_provider(self.lyrics_provider)
        lyrics_server.set_store(self.store)

        # Phase 3 RAG upgrade: song enricher (lyric-enriched index +
        # metadata-tag RAG). Built unconditionally — when ``mode == "none"``
        # or no providers are configured, it no-ops and songs are indexed on
        # their compact ``text`` line (Phase 1 behavior).
        self.enricher = Enricher(
            mode=config.enrichment.mode,
            lyrics_provider=self.lyrics_provider,
            llm=self.llm,
            lyrics_index_words=config.enrichment.lyrics_index_words,
            llm_tags=config.enrichment.llm_tags,
        )

    # ---- runtime model swap ------------------------------------------
    def set_model(self, model: str) -> None:
        """Swap the Ollama model used by the LLM client at runtime."""
        self.config.ollama_model = model
        if isinstance(self.llm, Gemma):
            self.llm.model = model

    def set_api_key(self, api_key: str | None) -> None:
        """Update the Ollama API key used for chat + management calls."""
        self.config.ollama_api_key = api_key or None
        if isinstance(self.llm, Gemma):
            self.llm.api_key = self.config.ollama_api_key

    # ---- Phase 3: lyrics provider construction -----------------------
    def _build_lyrics_provider(self) -> LyricsProvider | None:
        """Construct the configured lyrics provider (spec §21).

        Returns ``None`` when ``config.lyrics.provider == "none"`` or the
        choice is unrecognized (lyrics simply disabled). Unknown values do
        not raise — they fall back to "none" so a typo can't break the chat
        path.
        """
        name = (self.config.lyrics.provider or "").strip().lower()
        if name in ("", "none"):
            return None
        if name == "lrclib":
            return LRCLIBProvider(
                base_url=self.config.lyrics.lrclib_base_url,
                user_agent=self.config.lyrics.lrclib_user_agent,
                timeout=self.config.lyrics.timeout,
            )
        if name == "musixmatch":
            return MusixmatchProvider(
                user_token=self.config.lyrics.musixmatch_user_token,
                base_url=self.config.lyrics.musixmatch_base_url,
                timeout=self.config.lyrics.timeout,
            )
        return None

    # ---- ingestion ----------------------------------------------------
    def _enrich_and_build(
        self,
        songs: list[Song],
        progress: Any = None,
        fetch_done: float = 0.1,
        enrich: bool = True,
    ) -> None:
        """Shared tail of all four ingest paths: enrich, then build.

        Splits the progress bar across phases:
        - ``[0, fetch_done]``: playlist fetch (caller advances this).
        - ``[fetch_done, 0.5]``: enrichment (lyrics + tags).
        - ``[0.5, 1.0]``: FAISS embedding + SQLite write.

        When the enricher is disabled (``mode == "none"``) or no providers
        are configured, or when the caller passes ``enrich=False`` (the UI's
        per-ingest toggle), this collapses to a no-op and the build runs
        straight through, preserving Phase 1 behavior.
        """
        # Enrichment phase: lyrics + tags. The enricher's internal
        # progress callback maps its own ``[0, 1]`` range onto
        # ``[fetch_done, 0.5]`` of the overall bar.
        if enrich and self.enricher.enabled:
            def _enrich_progress(f: float) -> None:
                if progress is not None:
                    try:
                        progress(fetch_done + (0.5 - fetch_done) * max(0.0, min(1.0, f)))
                    except Exception:
                        pass
            self.enricher.enrich(songs, progress=_enrich_progress)
        elif progress is not None:
            try:
                progress(0.5)
            except Exception:
                pass

        # Build phase: embed ``index_text`` (enriched) and write SQLite.
        # ``VectorStore.build`` advances the bar over its own ``[0, 1]``
        # range; we remap that onto ``[0.5, 1.0]`` of the overall bar.
        def _build_progress(f: float) -> None:
            if progress is not None:
                try:
                    progress(0.5 + 0.5 * max(0.0, min(1.0, f)))
                except Exception:
                    pass

        self.store.build(songs, self.embedder, progress=_build_progress)

    def ingest(self, markdown_text: str, progress: Any = None, enrich: bool = True) -> int:
        """Ingest a Markdown playlist and build the FAISS index.

        ``progress`` is an optional Streamlit-style callable ``progress(f)``
        where ``f`` is a float in ``[0, 1]``. When provided, embedding
        generation advances the bar. When ``None`` (e.g. in tests or CLI use),
        no progress is reported. ``enrich`` toggles the enrichment phase
        (lyrics + tags); when ``False``, songs are embedded as-is using
        ``Song.text`` (Phase 1 behavior).
        """
        songs = parse_playlist(markdown_text)
        if not songs:
            raise ValueError("No songs parsed from input.")
        if progress is not None:
            try:
                progress(0.1)
            except Exception:
                pass
        self._enrich_and_build(songs, progress=progress, fetch_done=0.1, enrich=enrich)
        self._ingested = True
        self.history = []
        return len(songs)

    def load_existing(self) -> int:
        self.store.load()
        self._ingested = True
        self.history = []
        return self.store.size

    def ingest_youtube_playlist(self, url_or_id: str, progress: Any = None, enrich: bool = True) -> int:
        """Ingest a YouTube playlist and build the FAISS index.

        Parallel to :meth:`ingest`. No credentials required (``yt-dlp``
        scrapes the public playlist page). Raises ``YouTubeError`` on
        fetch/parse failure (including when ``yt-dlp`` isn't installed) and
        ``ValueError`` on invalid input.
        ``enrich`` toggles the enrichment phase; see :meth:`ingest`.
        """
        if self.youtube_client is None:
            raise RuntimeError(
                "YouTube ingestion is unavailable. Install yt-dlp with "
                "`pip install -r requirements.txt` and restart."
            )
        songs = self.youtube_client.import_playlist(url_or_id)
        if not songs:
            raise YouTubeError("YouTube playlist yielded no usable songs.")
        if progress is not None:
            try:
                progress(0.1)
            except Exception:
                pass
        self._enrich_and_build(songs, progress=progress, fetch_done=0.1, enrich=enrich)
        self._ingested = True
        self.history = []
        return len(songs)

    # ---- Phase 3: lyrics + preview (on-demand) ----------------------
    def get_lyrics_excerpt(self, track_id: str, max_words: int | None = None) -> dict:
        """Fetch a short lyrics excerpt for a song by ``track_id``.

        On-demand (not per-chat). Enforces the excerpt cap
        (``config.max_lyric_excerpt_words`` by default) and attaches
        provenance (``provider`` + ``source_url``). Returns
        ``{ok, available, excerpt, provider, source_url, is_synced}``;
        ``available=False`` when the song isn't in the store or the provider
        has no match (spec §21).
        """
        cap = max_words if max_words is not None else self.config.max_lyric_excerpt_words
        return lyrics_server.get_excerpt(track_id, max_words=cap)

    @property
    def ready(self) -> bool:
        return self._ingested and self.store.size > 0

    def clear_index(self) -> None:
        """Clear the built index: remove the on-disk FAISS index + SQLite
        DB and reset the in-memory store + chat history. After this,
        :attr:`ready` is ``False`` until the user re-ingests a playlist.
        """
        self.store.clear()
        self._ingested = False
        self.history = []
        # Reset the Antakshari session if one was active.
        self.antakshari = None

    def clear_chat_history(self) -> None:
        """Clear the chat history only (and any active Antakshari session).

        The indexed song collection is left intact. Granular companion to
        :meth:`clear_index` for the UI's "Clear Chat History" button.
        """
        self.history = []
        self.antakshari = None

    def clear_collection(self) -> None:
        """Clear the indexed song collection (FAISS + SQLite) and reset any
        active Antakshari session. Chat history is NOT cleared (use
        :meth:`clear_chat_history` for that). After this, :attr:`ready` is
        ``False`` until the user re-ingests a playlist.
        """
        self.store.clear()
        self._ingested = False
        # Antakshari depends on the store; reset it when the collection goes.
        self.antakshari = None

    # ---- feedback + time preference (Phase 2) ------------------------
    def record_play(self, track_id: str, completion_ratio: float = 1.0) -> dict:
        return feedback_server.record_play(track_id, completion_ratio=completion_ratio)

    def record_skip(self, track_id: str) -> dict:
        return feedback_server.record_skip(track_id)

    def record_like(self, track_id: str) -> dict:
        return feedback_server.record_like(track_id)

    def get_feedback(self, track_id: str) -> dict:
        return feedback_server.get_preference_score(track_id)

    def set_time_bias(self, bias: float) -> float:
        """Clamp ``bias`` to ``[-1, 1]`` and store as the default time bias."""
        clamped = max(-1.0, min(1.0, float(bias)))
        self.config.personalization.default_time_bias = clamped
        return clamped

    def _recently_surfaced_track_ids(self) -> set[str]:
        """Collect track_ids from the top-``rerank_k`` candidates of the last
        ``memory_turns`` turns, so they can be penalized as repetition."""
        ids: set[str] = set()
        for turn in self.history[-self.memory_turns :]:
            # ``candidate_ids`` is the full retrieved set for the turn; the
            # top ``rerank_k`` were surfaced to the LLM. We penalize those.
            for tid in turn.candidate_ids[: self.config.rerank_k]:
                ids.add(tid)
        return ids

    # ---- chat ---------------------------------------------------------
    def _history_for_llm(self) -> list[dict]:
        msgs: list[dict] = []
        for t in self.history[-self.memory_turns :]:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": t.assistant})
        return msgs

    def _retrieve_top_candidates(self, user_message: str) -> tuple[str, list[Candidate]]:
        """Shared retrieval pipeline for :meth:`chat` and :meth:`chat_lyric`.

        Returns the (possibly rewritten) query and the full reranked +
        personalized candidate list. The caller picks the top slice.
        """
        # Surface a clear error when a cloud model is configured without an
        # API key, instead of silently degrading. The extract_query fallback
        # below swallows generic RuntimeError for transient LLM outages; this
        # pre-check bypasses the swallow for the auth/config case.
        _ensure_llm = getattr(self.llm, "ensure_configured", None)
        if callable(_ensure_llm):
            _ensure_llm()

        llm_history = self._history_for_llm()

        # 1. Query understanding (resolves follow-ups against short-term memory).
        try:
            query = self.llm.extract_query(user_message, llm_history)
        except RuntimeError:
            # If the LLM is unreachable, fall back to the raw message so the
            # rest of the pipeline still works for retrieval-only testing.
            query = user_message

        # 2. Retrieve (semantic top-k).
        candidates = self.retriever.retrieve(query, top_k=self.config.top_k)

        # 3. Rerank (semantic + keyword).
        candidates = rerank(query, candidates, self.config.ranking_weights)

        # 4. Personalization: + feedback + time - repetition.
        feedback_map = {
            r.track_id: r for r in self.feedback_store.all_rows().values()
        }
        candidates = finalize_scores(
            candidates,
            feedback=feedback_map,
            time_bias=self.config.personalization.default_time_bias,
            seen_track_ids=self._recently_surfaced_track_ids(),
            fb_weights=self.config.personalization.feedback_weights,
            ranking_weights=self.config.ranking_weights,
        )
        return query, candidates

    def chat(self, user_message: str) -> ValidationOutcome:
        """Recommendation mode: the LLM generates a prose response grounded
        in candidate metadata. Use :meth:`chat_lyric` for lyric-first
        responses where the lyric excerpt IS the response.
        """
        if not self.ready:
            raise RuntimeError("Index not built. Call ingest() or load_existing() first.")

        query, candidates = self._retrieve_top_candidates(user_message)

        # 5. Generate, grounded in candidate metadata only.
        top = candidates[: self.config.rerank_k]
        cand_dicts = [c.song.to_dict() for c in top]
        try:
            response = self.llm.generate(query, cand_dicts, self._history_for_llm())
        except RuntimeError as exc:
            response = (
                "I couldn't reach the local model. Make sure Ollama is running "
                f"(`ollama serve`) and the model is pulled. Error: {exc}"
            )

        # 6. Validate.
        outcome = validate(
            response,
            cand_dicts,
            user_message=user_message,
            max_lyric_excerpt_words=self.config.max_lyric_excerpt_words,
            include_source=self.config.include_source,
        )

        # Build structured sources (display + track_id) for the UI feedback
        # buttons. The validator already attached provenance from the
        # candidate dicts; here we pair each source with its track_id.
        source_dicts: list[dict] = []
        used_sources = list(outcome.sources)
        for i, c in enumerate(top):
            display = used_sources[i] if i < len(used_sources) else (
                f"{c.song.artist} \u2014 {c.song.title}"
            )
            source_dicts.append({"display": display, "track_id": c.song.track_id})

        # Record this turn for short-term memory + repetition tracking.
        self.history.append(
            ChatTurn(
                user=user_message,
                assistant=outcome.response,
                sources=source_dicts,
                rewritten_query=query,
                candidate_ids=[c.song.track_id for c in candidates],
            )
        )
        return outcome

    # ---- Lyric Prose Recommendation (Phase 4 addition) ---------------
    def chat_lyric_prose(self, user_message: str) -> ValidationOutcome:
        """Lyric Prose Recommendation mode: the user supplies a letter and
        the assistant returns a collection of short permitted lyric
        **excerpts that begin with that letter**, gathered from across the
        indexed songs. Each excerpt is a short multi-line snippet (5
        lines starting from the first line whose first alphabetic
        character equals the input letter) and is paired with its song
        title and artist as detail.

        The matching line can come from anywhere in a song (the opening
        *or* a line from the middle), so a song can contribute even when
        its title doesn't start with the letter. One excerpt per song,
        deduped, each capped at ``max_lyric_excerpt_words`` words (with a
        floor that ensures 5 lines fit; copyright invariant, spec §21:
        multiple short excerpts, never one long one). The LLM never
        generates lyric text; lyrics are fetched via ``lyrics-mcp``
        (spec §12 invariant holds).

        If no letter can be parsed from the message, falls back to
        :meth:`chat` so the mode is forgiving for ordinary prose queries
        (e.g. "Explain heartbreak" still produces a prose recommendation).
        """
        if not self.ready:
            raise RuntimeError("Index not built. Call ingest() or load_existing() first.")

        letter = _detect_letter(user_message)
        if letter is None:
            # No letter signal — fall back to the prose recommendation path.
            return self.chat(user_message)

        cap = self.config.max_lyric_excerpt_words
        # Floor on the per-excerpt display cap so the 5-line snippet
        # actually fits. Still a short permitted excerpt (spec §21);
        # respects a higher configured cap verbatim.
        display_cap = max(cap, 60)
        # Fetch a larger lyric pool per song so a matching line can sit
        # anywhere in the first ``pool_words`` words of the song, and the
        # following 4 lines are reachable too.
        pool_words = max(400, cap * 8)
        lines_per_song = 5          # lines of lyrics per excerpt
        target_examples = 6         # how many songs to gather from
        max_songs_to_check = 30    # cap the number of network calls
        provider_name = ""
        any_romanized = False
        seen_lines: set[str] = set()  # dedupe identical excerpts across songs
        collected: list[tuple] = []  # list of (song, picked_excerpt)
        checked_track_ids: list[str] = []
        for song in self.store.all_songs():
            if len(collected) >= target_examples:
                break
            if len(checked_track_ids) >= max_songs_to_check:
                break
            checked_track_ids.append(song.track_id)
            res = lyrics_server.get_excerpt(song.track_id, max_words=pool_words)
            pool = (res or {}).get("excerpt", "")
            if (res or {}).get("provider"):
                provider_name = res["provider"]
            if (res or {}).get("romanized"):
                any_romanized = True
            if not (res or {}).get("available") or not pool:
                continue
            picked = _pick_lines_starting_with(
                pool, letter, lines_per_song, display_cap
            )
            if not picked:
                continue
            key = picked.strip().lower()
            if key in seen_lines:
                continue
            seen_lines.add(key)
            collected.append((song, picked))

        if not collected:
            response = (
                f"No lyrics starting with the letter '{letter.upper()}' "
                f"were found in your collection. Try a different letter, "
                f"or ingest more songs first."
            )
            outcome = ValidationOutcome(response=response, sources=[])
            self.history.append(
                ChatTurn(
                    user=user_message,
                    assistant=outcome.response,
                    sources=[],
                    rewritten_query=user_message,
                    candidate_ids=[],
                    mode="lyric_prose",
                )
            )
            return outcome

        lines: list[str] = [
            f"Here are some lyrics that begin with the letter "
            f"**{letter.upper()}**:",
            "",
        ]
        for song, exc in collected:
            lines.append(
                f"- \u201c{exc}\u201d \u2014 **{song.title}** \u00b7 {song.artist}"
            )
        if provider_name:
            lines.append("")
            lines.append(f"(Lyrics source: {provider_name})")
        if any_romanized:
            lines.append("(Lyrics transliterated to Latin)")
        response = "\n".join(lines)

        outcome = ValidationOutcome(
            response=response,
            sources=[],  # title/artist inlined in the response body
            excerpt_truncated=False,
        )
        self.history.append(
            ChatTurn(
                user=user_message,
                assistant=outcome.response,
                sources=[],  # title/artist inlined in the response body
                rewritten_query=user_message,
                candidate_ids=checked_track_ids,
                mode="lyric_prose",
            )
        )
        return outcome

    # ---- lyric-first chat (Phase 3 addition) ------------------------
    def chat_lyric(self, user_message: str) -> ValidationOutcome:
        """Lyric mode: the response IS a song lyric excerpt sourced from the
        configured lyrics provider (LRCLIB by default).

        Flow (spec.md §21)::

            user_message
              -> extract_query (Gemma, with short-term memory)
              -> retrieve + rerank + finalize (same pipeline as chat())
              -> pick the top candidate with an available lyric
              -> fetch the capped excerpt from the lyrics provider
              -> attach provenance (song / artist / album / lyrics provider)
              -> validate (full-lyrics-request gate still applies)

        The playlist (Markdown / YouTube) defines the candidate
        pool — its metadata (title, artist, album) is what the retrieval
        layer matches against, and what the lyrics provider uses to look up
        the excerpt. The LLM never generates lyric text; it only helps with
        query understanding. The "LLM never touches DBs" invariant (spec
        §12) holds: lyrics are fetched via ``lyrics-mcp``, not by the LLM.

        Tries the top ``rerank_k`` candidates in order and returns the
        first one with an available lyric, so a top match that has no
        lyrics in LRCLIB doesn't block the response. If none have lyrics,
        returns a graceful "no lyrics available" message naming the top
        song. Full-lyrics requests ("give me the entire lyrics") are still
        routed to ``SAFE_FULL_LYRICS_REPLY`` (spec §21).
        """
        if not self.ready:
            raise RuntimeError("Index not built. Call ingest() or load_existing() first.")

        query, candidates = self._retrieve_top_candidates(user_message)
        top = candidates[: self.config.rerank_k]
        candidate_ids = [c.song.track_id for c in candidates]

        # No candidates at all.
        if not top:
            outcome = ValidationOutcome(
                response="No matching songs found in your collection.",
                sources=[],
            )
            self.history.append(
                ChatTurn(
                    user=user_message,
                    assistant=outcome.response,
                    sources=[],
                    rewritten_query=query,
                    candidate_ids=candidate_ids,
                )
            )
            return outcome

        top_song = top[0].song

        # Full-lyrics-request gate (spec §21) — still applies in lyric mode.
        if is_full_lyrics_request(user_message):
            display = f"{top_song.artist} \u2014 {top_song.title}"
            outcome = ValidationOutcome(
                response=SAFE_FULL_LYRICS_REPLY,
                sources=[display],
                flagged_full_lyrics_request=True,
            )
            self.history.append(
                ChatTurn(
                    user=user_message,
                    assistant=outcome.response,
                    sources=[{"display": display, "track_id": top_song.track_id}],
                    rewritten_query=query,
                    candidate_ids=candidate_ids,
                )
            )
            return outcome

        # No lyrics provider configured → graceful message naming the top song.
        if self.lyrics_provider is None:
            display = f"{top_song.artist} \u2014 {top_song.title}"
            response = (
                f"Lyrics are disabled (LYRICS_PROVIDER=none). I found "
                f"{top_song.title} by {top_song.artist} in your collection, "
                f"but can't fetch lyrics. Configure a lyrics provider in "
                f"config.yaml or set LYRICS_PROVIDER=lrclib."
            )
            outcome = ValidationOutcome(response=response, sources=[display])
            self.history.append(
                ChatTurn(
                    user=user_message,
                    assistant=outcome.response,
                    sources=[{"display": display, "track_id": top_song.track_id}],
                    rewritten_query=query,
                    candidate_ids=candidate_ids,
                )
            )
            return outcome

        # LLM-based song selection: the model uses its world knowledge of
        # song moods/themes/eras (not just title keyword matching) to pick
        # the best fit from the top-k candidates. Falls back to retrieval
        # top-1 if the LLM is unreachable or returns an unparseable answer.
        cand_dicts = [c.song.to_dict() for c in top]
        selected_idx = 0
        try:
            selected_tid = self.llm.select_song(
                query, cand_dicts, self._history_for_llm()
            )
            if selected_tid:
                for i, c in enumerate(top):
                    if c.song.track_id == selected_tid:
                        selected_idx = i
                        break
        except RuntimeError:
            pass

        # Try candidates in priority order: the LLM's pick first, then the
        # rest in retrieval order. Gather lyrics from up to ``max_songs``
        # candidates that have an available lyric, so the LLM can weave
        # phrases from multiple songs — this improves context/relevance
        # when no single song's lyrics fully capture the user's request.
        # Each phrase is individually capped at ``cap`` words (copyright
        # invariant, spec §21: never return large portions of copyrighted
        # lyrics; we return multiple short excerpts, not one long one).
        priority_order = [selected_idx] + [
            i for i in range(len(top)) if i != selected_idx
        ]
        cap = self.config.max_lyric_excerpt_words
        # Fetch a larger lyric pool per song so we can pick the phrase most
        # relevant to the user's actual message (semantic match via
        # embeddings, keyword-overlap fallback).
        pool_words = max(200, cap * 4)
        max_songs = 3
        phrases_per_song = 3
        # Collected entries: list of (song, res, phrases_for_this_song).
        collected: list[tuple] = []
        last_res: dict = {}
        for idx in priority_order:
            if len(collected) >= max_songs:
                break
            res = lyrics_server.get_excerpt(
                top[idx].song.track_id, max_words=pool_words
            )
            last_res = res
            if not res.get("available"):
                continue
            pool_excerpt = res.get("excerpt", "")
            song_phrases = select_relevant_phrases(
                pool_excerpt, user_message, cap, n=phrases_per_song,
                embedder=self.embedder,
            )
            if not song_phrases:
                fallback = cap_words(pool_excerpt, cap)
                song_phrases = [fallback] if fallback else []
            if not song_phrases:
                continue
            collected.append((top[idx].song, res, song_phrases))

        if not collected:
            # No lyrics for any candidate — graceful message naming the top song.
            display = f"{top_song.artist} \u2014 {top_song.title}"
            provider_name = (last_res or {}).get("provider", "")
            tail = f" from {provider_name}" if provider_name else ""
            response = (
                f"I found {top_song.title} by {top_song.artist} in your "
                f"collection, but no lyrics are available for it{tail}. "
                f"Try a different song or switch lyrics provider "
                f"(LYRICS_PROVIDER=musixmatch with a user token has better "
                f"Hindi/Urdu coverage)."
            )
            outcome = ValidationOutcome(response=response, sources=[display])
            self.history.append(
                ChatTurn(
                    user=user_message,
                    assistant=outcome.response,
                    sources=[{"display": display, "track_id": top_song.track_id}],
                    rewritten_query=query,
                    candidate_ids=candidate_ids,
                )
            )
            return outcome

        # Flatten the per-song phrase lists into one ordered list (primary
        # song's phrases first, then the next song's, etc.). The primary
        # song is the first collected (the LLM's pick or retrieval top-1
        # with an available lyric).
        primary_song, primary_res, primary_phrases = collected[0]
        all_phrases: list[str] = []
        for _song, _res, sp in collected:
            all_phrases.extend(sp)
        # Dedupe while preserving order.
        seen_p: set[str] = set()
        unique_phrases: list[str] = []
        for p in all_phrases:
            k = (p or "").lower().strip()
            if not k or k in seen_p:
                continue
            seen_p.add(k)
            unique_phrases.append(p)
        excerpt = unique_phrases[0] if unique_phrases else ""
        provider = primary_res.get("provider", "")
        is_synced = any(bool(r.get("is_synced")) for _, r, _ in collected)

        # Ask the LLM for the conversational body. Pass the original user
        # message (not the rewritten query) and the combined phrases from
        # multiple songs so the LLM can weave the most relevant lyrical
        # phrases with minimal conversational text. If the LLM is
        # unreachable, fall back to a templated body that strings the
        # phrases together as quoted lines.
        try:
            body = self.llm.generate_lyric_intro(
                user_message,
                primary_song.to_dict(),
                excerpt,
                self._history_for_llm(),
                extra_phrases=unique_phrases[1:],
            )
            body = (body or "").strip()
        except (RuntimeError, TypeError):
            body = ""
        if not body:
            # Templated fallback when the LLM is down or returns nothing.
            # Weave the first 2-3 phrases into a sentence-like line with
            # minimal connective text. Never mentions the song names — the
            # provenance block below shows them separately.
            picks = unique_phrases[:3] if len(unique_phrases) >= 2 else unique_phrases
            if len(picks) >= 2:
                quoted = " and ".join(f"\u201c{p}\u201d" for p in picks)
                body = f"It feels like {quoted}."
            elif picks:
                body = f"Here's a line that fits \u2014 \u201c{picks[0]}\u201d"
            else:
                body = ""

        # Provenance: list every contributing song (multi-song mode).
        lines: list[str] = [body, ""]
        if len(collected) == 1:
            lines.append(
                f"Song: {primary_song.artist} \u2014 {primary_song.title}"
            )
            if primary_song.album:
                lines.append(f"Album: {primary_song.album}")
            if primary_song.release_date:
                lines.append(f"Released: {primary_song.release_date}")
        else:
            lines.append("Songs:")
            for song, _r, _sp in collected:
                lines.append(f"- {song.artist} \u2014 {song.title}")
        lines.append(f"(Lyrics source: {provider})")
        if is_synced:
            lines.append("(Lyrics type: synced \u2014 timing tags stripped)")
        if any(bool(r.get("romanized")) for _, r, _ in collected):
            lines.append("(Lyrics transliterated to Latin)")
        response = "\n".join(lines)

        source_displays = [
            {"display": f"{s.artist} \u2014 {s.title}", "track_id": s.track_id}
            for s, _r, _sp in collected
        ]
        # Per-track lyric data: the exact phrases selected and used in the
        # LLM prompt, so the UI's Lyrics button can show them rather than
        # fetching a fresh (different) excerpt.
        lyric_data: dict[str, dict] = {}
        for song, res, sp in collected:
            lyric_data[song.track_id] = {
                "title": song.title,
                "artist": song.artist,
                "album": song.album or "",
                "release_date": song.release_date or "",
                "source": song.source or "",
                "source_url": song.source_url or "",
                "duration": song.duration or 0,
                "tags": list(song.tags or []),
                "phrases": list(sp),
                "provider": res.get("provider", ""),
                "lyrics_source_url": res.get("source_url", ""),
                "is_synced": bool(res.get("is_synced", False)),
                "romanized": bool(res.get("romanized", False)),
            }
        outcome = ValidationOutcome(
            response=response,
            sources=[d["display"] for d in source_displays],
            excerpt_truncated=False,  # provider already capped per phrase
        )
        self.history.append(
            ChatTurn(
                user=user_message,
                assistant=outcome.response,
                sources=source_displays,
                rewritten_query=query,
                candidate_ids=candidate_ids,
                lyric_data=lyric_data,
            )
        )
        return outcome

    # ---- Phase 4: interaction modes ---------------------------------
    def start_antakshari(self) -> dict:
        """Start a new Antakshari game (spec §16). Delegates to
        :mod:`app.modes.antakshari`."""
        return antakshari_mode.start(self)

    def stop_antakshari(self) -> dict:
        """End the current Antakshari game (the "Stop game" button)."""
        return antakshari_mode.stop(self)

    def submit_antakshari(self, user_answer: str) -> dict:
        """Validate the user's answer and advance the Antakshari round.
        Returns a structured dict the UI renders (``{ok, message, session,
        ...}``). When no game is active, auto-starts one."""
        return antakshari_mode.submit(self, user_answer)

    def chat_dispatch(
        self, user_message: str, override: str | None = None
    ) -> ValidationOutcome:
        """Auto-route the user's message by detected intent (spec §14).

        ``override`` (set by the UI's mode selector) forces a specific
        path regardless of intent:
        - ``"lyric"``   → :meth:`chat_lyric` (lyrical conversational mode)
        - ``"explain"`` → :func:`app.modes.explain.explain` (explanation mode)
        - ``"prose"``   → :meth:`chat_lyric_prose` (Lyric Prose
          Recommendation mode — letter → multiple short lyric excerpts as
          Antakshari examples; falls back to :meth:`chat` if no letter is
          detected in the message)
        - ``None``      → auto-route by intent (the default path)

        Auto-routing:
        - ``EXPLAIN``     → :func:`app.modes.explain.explain`
        - ``ANTAKSHARI``  → :meth:`submit_antakshari` (starts/advances the
          game; the UI renders the structured dict, so this method wraps
          the result in a :class:`ValidationOutcome`).
        - ``GENERATION``  → :func:`app.modes.generation.generate_playlist`
        - ``LIST``        → :func:`app.modes.list_indexed.list_indexed`
        - ``GENERAL``     → :meth:`chat_lyric` (lyrical conversational mode)

        Records the mode on the appended :class:`ChatTurn` (except for
        Antakshari, which manages its own state and doesn't append a
        chat turn — the game UI is rendered separately).
        """
        if override == "lyric":
            outcome = self.chat_lyric(user_message)
            if self.history and self.history[-1].user == user_message:
                self.history[-1].mode = "general"
            return outcome
        if override == "prose":
            outcome = self.chat_lyric_prose(user_message)
            # chat_lyric_prose records mode="lyric_prose" on the turn (or
            # falls back to chat(), which records mode="general"). Leave the
            # recorded mode as-is.
            return outcome
        if override == "explain":
            return explain_mode(self, user_message)

        intent = detect_intent(self.llm, user_message, self._history_for_llm())

        if intent == Intent.EXPLAIN:
            outcome = explain_mode(self, user_message)
            return outcome
        if intent == Intent.GENERATION:
            outcome = generation_mode(self, user_message)
            return outcome
        if intent == Intent.LIST:
            outcome = list_mode(self, user_message)
            return outcome
        if intent == Intent.ANTAKSHARI:
            # Antakshari manages its own state; wrap the structured dict
            # into a ValidationOutcome so the UI's chat path can render
            # the message. The dedicated Antakshari panel shows the game
            # state (required char, score, used-songs count, Stop button).
            res = self.submit_antakshari(user_message)
            outcome = ValidationOutcome(
                response=res.get("message", ""),
                sources=[],
            )
            return outcome
        # GENERAL → lyric-first (the current Phase 3 default).
        outcome = self.chat_lyric(user_message)
        if self.history and self.history[-1].user == user_message:
            self.history[-1].mode = "general"
        return outcome