"""Musixmatch lyrics provider — Phase 3 stub.

The seam + config is the Phase 3 deliverable. The live call is deferred until
the user supplies a ``MUSIXMATCH_USER_TOKEN`` (Musixmatch requires a
per-user token harvested from browser cookies after logging in to
musixmatch.com — there is no anonymous app-level access). Until the token is
set, every lookup returns ``LyricsExcerpt(available=False, provider="Musixmatch")``.

The upgrade path (better Hindi/Urdu coverage than LRCLIB) is documented in
CLAUDE.md. When the user sets the token, the live ``track.lyrics.get`` call
can be filled in here without touching the orchestrator or other providers.
"""

from __future__ import annotations

from app.lyrics.base import LyricsExcerpt


class MusixmatchProvider:
    """Musixmatch lyrics provider (key-gated stub)."""

    def __init__(
        self,
        user_token: str = "",
        base_url: str = "https://api.musixmatch.com/ws/1.1",
        timeout: float = 10.0,
    ) -> None:
        self.user_token = (user_token or "").strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def get(
        self,
        title: str,
        artist: str,
        album: str = "",
        duration: int | None = None,
        max_words: int = 25,
    ) -> LyricsExcerpt | None:
        if not title or not artist:
            return None
        if not self.user_token:
            return LyricsExcerpt(
                track_id="",
                available=False,
                excerpt="",
                provider="Musixmatch",
                source_url="",
            )

        # Full implementation is deferred — the seam + config is the Phase 3
        # deliverable. The live ``track.search`` → ``track.lyrics.get`` flow
        # can be filled in here once the user has a token. Until then we
        # explicitly return "not available" so the UI can suggest switching
        # to LRCLIB or supplying a token.
        return LyricsExcerpt(
            track_id="",
            available=False,
            excerpt="",
            provider="Musixmatch",
            source_url="https://www.musixmatch.com/",
        )