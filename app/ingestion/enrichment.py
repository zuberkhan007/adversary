"""Song enrichment for the Phase 3 RAG upgrade.

Two complementary enrichment strategies (per the RAG-upgrade plan):

1. **Lyric-enriched index (Option 2):** fetch plain lyrics per song from
   the configured :class:`LyricsProvider` (LRCLIB by default), cap to
   ``lyrics_index_words`` (~100 words), and store on :attr:`Song.index_lyrics`.
   The indexed text (what FAISS embeds and BM25 scores against) becomes
   ``"Title — Artist — tags — lyric excerpt"``. This makes the index
   content-rich: a query like "i'm feeling lost tonight" can match songs
   whose lyrics contain "lost"/"tonight" even when the title/artist line
   doesn't. The indexed lyric excerpt is NEVER returned to the LLM/UI
   (copyright: spec.md §21) — it's a derivative used only for indexing.

2. **Metadata-tag RAG (Option 3):** derive mood/theme tags per song.
   When ``llm_tags`` is enabled, the LLM generates 3-5 mood/theme tags
   from title+artist using its world knowledge of the song.

Both enrichments are best-effort: any failure (provider down, LLM down)
leaves the song's enrichment fields empty and the song is still indexed on
its compact ``text`` line. The orchestrator wires the right enricher based
on :class:`EnrichmentConfig.mode`.

The :class:`Enricher` is constructed in :meth:`MusicAssistant.__init__`
and called between fetch and ``store.build(...)`` in the ingestion paths.
Enrichment mutates the ``Song`` objects in place and returns a small stats
dict for UI logging.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.ingestion.markdown import Song


class LyricEnricher:
    """Fetch plain lyrics per song (capped) and store on ``Song.index_lyrics``.

    Uses the configured :class:`LyricsProvider` (LRCLIB by default). Best-effort:
    any failure (no provider, 404, network error) leaves ``index_lyrics`` empty.
    """

    def __init__(self, provider: Any | None, index_words: int = 100) -> None:
        self.provider = provider
        # Floor at 1 (not higher) so tests can exercise the cap with tiny
        # values; a production ``lyrics_index_words < 20`` is still honored
        # because the admin who set it knows what they're doing.
        self.index_words = max(1, int(index_words or 100))

    def enrich_one(self, song: Song) -> bool:
        if self.provider is None:
            return False
        if song.index_lyrics:
            return True  # already enriched
        try:
            excerpt = self.provider.get(
                title=song.title,
                artist=song.artist,
                album=song.album,
                duration=song.duration or None,
            )
        except Exception:
            return False
        if excerpt is None or not excerpt.available:
            return False
        # Use plain (non-synced) lyrics for indexing — we don't want LRC
        # timing tags polluting the BM25 token stream. ``plainLyrics`` is
        # already stored on ``excerpt.excerpt`` by the LRCLIB provider when
        # ``plain=True``; otherwise the excerpt is already stripped of tags.
        text = (excerpt.excerpt or "").strip()
        if not text:
            return False
        # Cap to ``index_words`` for indexing (separate from the chat-time
        # output cap of ``max_lyric_excerpt_words``). Indexing can afford a
        # longer window because it's never shown to the user.
        words = text.split()
        if len(words) > self.index_words:
            text = " ".join(words[: self.index_words])
        song.index_lyrics = text
        return True


class LLMTagEnricher:
    """Generate mood tags from title+artist via the LLM.

    The LLM uses its world knowledge of the song's vibe/themes.
    """

    def __init__(self, llm: Any | None) -> None:
        self.llm = llm

    def enrich_one(self, song: Song) -> bool:
        if self.llm is None:
            return False
        if song.tags:
            return True  # already tagged
        try:
            tags = self.llm.generate_tags(song.title, song.artist)
        except Exception:
            return False
        if not tags:
            return False
        # Normalize: lowercase, strip, dedupe.
        seen: set[str] = set()
        clean: list[str] = []
        for t in tags:
            t = (t or "").strip().lower()
            if t and t not in seen:
                seen.add(t)
                clean.append(t)
        if not clean:
            return False
        song.tags = clean[:6]
        return True


class Enricher:
    """Compose lyric + tag enrichment based on ``EnrichmentConfig.mode``.

    The orchestrator constructs one of these in ``__init__`` and calls
    :meth:`enrich` between fetch and ``store.build(...)``. ``mode`` selects
    which enrichments run; ``"auto"`` (default) runs both, with lyrics only
    when a provider is configured and tags via the LLM otherwise.
    """

    def __init__(
        self,
        mode: str = "auto",
        lyrics_provider: Any | None = None,
        llm: Any | None = None,
        lyrics_index_words: int = 100,
        llm_tags: bool = True,
    ) -> None:
        self.mode = (mode or "auto").strip().lower()
        self.lyrics_provider = lyrics_provider
        self.llm = llm
        self.lyrics_index_words = lyrics_index_words
        self.llm_tags = llm_tags

        # Build the per-strategy enrichers up front (cheap; they hold no
        # state beyond their constructor args). ``None`` providers make the
        # enrichers no-ops.
        self._lyric = LyricEnricher(lyrics_provider, index_words=lyrics_index_words)
        self._llm_tags = LLMTagEnricher(llm) if llm_tags else None

    @property
    def enabled(self) -> bool:
        """Whether this enricher will do any work at all."""
        return self.mode != "none"

    def _wants_lyrics(self) -> bool:
        return self.mode in ("lyrics", "both", "auto")

    def _wants_tags(self) -> bool:
        return self.mode in ("tags", "both", "auto")

    def enrich(self, songs: list[Song], progress: Any = None) -> dict:
        """Enrich songs in place. Returns a stats dict for UI logging.

        ``progress`` is an optional Streamlit-style ``progress(f)`` callable;
        the bar advances across both phases (lyrics, then tags).
        """
        stats = {
            "songs": len(songs),
            "lyrics_enriched": 0,
            "tags_enriched": 0,
            "skipped": 0,
        }
        if not songs or self.mode == "none":
            return stats

        n = len(songs)
        # Phase A: lyrics (per-song, best-effort).
        if self._wants_lyrics() and self.lyrics_provider is not None:
            for i, song in enumerate(songs):
                if self._lyric.enrich_one(song):
                    stats["lyrics_enriched"] += 1
                if progress is not None:
                    try:
                        # First 60% of the bar is lyrics (lyric fetching is
                        # the slow part — one HTTP call per song).
                        progress(0.1 + 0.6 * ((i + 1) / n))
                    except Exception:
                        pass
        elif progress is not None:
            try:
                progress(0.7)
            except Exception:
                pass

        # Phase B: tags.
        if self._wants_tags():
            # LLM tags for songs. This is per-song LLM calls — only run
            # when ``llm_tags`` is enabled.
            if self._llm_tags is not None:
                untagged = [s for s in songs if not s.tags]
                for i, song in enumerate(untagged):
                    if self._llm_tags.enrich_one(song):
                        stats["tags_enriched"] += 1
                    if progress is not None:
                        try:
                            # Last 30% of the bar is LLM tags.
                            progress(0.7 + 0.3 * ((i + 1) / max(1, len(untagged))))
                        except Exception:
                            pass
        elif progress is not None:
            try:
                progress(1.0)
            except Exception:
                pass

        stats["skipped"] = n - stats["lyrics_enriched"] - stats["tags_enriched"]
        return stats