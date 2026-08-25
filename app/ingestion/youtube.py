"""YouTube playlist ingestion via ``yt-dlp`` (no credentials required).

Phase 3 ingestion path. YouTube public playlists can be read by scraping
the public playlist page — no API key, no OAuth. ``yt-dlp`` is the engine;
it is actively maintained and handles YouTube's page-structure changes.

This module is isolated in ``app/ingestion/`` so it can be removed/swapped
without touching the orchestrator.

Each entry becomes a ``Song`` with ``source="youtube"``,
``track_id="youtube:<video_id>"``, ``source_url`` set to the watch URL, and
``preview_url=""`` (YouTube doesn't expose a 30s audio clip — the UI renders
no ``<audio>`` player for YouTube songs; the watch URL is still attached as
``source_url`` for provenance).

Title → artist/title parsing: YouTube music videos are commonly titled
``"Artist - Title"``. We split on the same separators as the Markdown parser
(em dash, en dash, hyphen-with-spaces). When no separator is found, the full
title is kept and the artist falls back to ``"Unknown Artist"`` so the song
is still indexable by title.

No network calls happen at import time. ``YouTubeClient`` is constructed
lazily by the orchestrator; if ``yt-dlp`` is not installed, construction
raises ``YouTubeError`` which the orchestrator catches to disable the
YouTube path gracefully.
"""

from __future__ import annotations

import re

from app.ingestion.markdown import Song


class YouTubeError(RuntimeError):
    """Raised on parse/fetch failures from YouTube."""


# YouTube playlist IDs: alphanumeric + dash + underscore, typically 10-50
# chars. Common prefixes: PL, RD, LL, FL, OL. We accept any plausible shape
# and let yt-dlp reject genuinely invalid IDs.
_PLAYLIST_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.|music\.)?youtube\.com/playlist\?list=([A-Za-z0-9_-]+)"
)
_WATCH_WITH_LIST_RE = re.compile(
    r"(?:https?://)?(?:www\.|music\.)?youtube\.com/watch\?(?:[^#]*&)?list=([A-Za-z0-9_-]+)"
)
_YOUTU_BE_RE = re.compile(r"(?:https?://)?youtu\.be/[^?\s]+[?&]list=([A-Za-z0-9_-]+)")
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,50}$")

# Reuse the Markdown parser's separator set (em dash, en dash, hyphen w/ spaces).
_TITLE_SEP = re.compile(r"\s+[\u2014\u2013\-]\s+")


def parse_playlist_id(url_or_id: str) -> str:
    """Parse a YouTube playlist URL / bare ID into the playlist ID.

    Accepts:
      - ``https://www.youtube.com/playlist?list=PLxxxx``
      - ``https://www.youtube.com/watch?v=xxx&list=PLxxxx``
      - ``https://youtu.be/xxx?list=PLxxxx``
      - ``https://music.youtube.com/playlist?list=PLxxxx``
      - a bare playlist ID (alphanumeric + ``-_``, 10-50 chars).

    Raises ``ValueError`` for anything else (spec §31 input validation).
    """
    if not isinstance(url_or_id, str) or not url_or_id.strip():
        raise ValueError("YouTube playlist input must be a non-empty string")
    s = url_or_id.strip()

    for rx in (_PLAYLIST_URL_RE, _WATCH_WITH_LIST_RE, _YOUTU_BE_RE):
        m = rx.search(s)
        if m:
            return m.group(1)

    # Strip a leading "list=" just in case the user pasted a bare param value.
    if s.startswith("list="):
        s = s[len("list=") :]

    if _BARE_ID_RE.match(s):
        return s

    raise ValueError(
        "Unrecognized YouTube playlist input. Expected a playlist URL "
        "(https://www.youtube.com/playlist?list=...), a watch URL with a "
        "list= param, or a bare playlist ID."
    )


def _split_artist_title(title: str) -> tuple[str, str]:
    """Split a YouTube video title into (artist, title).

    Music videos are commonly titled ``"Artist - Title"``. When no
    separator is found, the full title becomes the song title and the
    artist falls back to ``"Unknown Artist"`` (still indexable by title).
    """
    parts = _TITLE_SEP.split(title, maxsplit=1)
    if len(parts) == 2:
        a, t = parts[0].strip(), parts[1].strip()
        if a and t:
            return a, t
    return "Unknown Artist", title.strip()


class YouTubeClient:
    """YouTube playlist ingestion client (no credentials required)."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_entries: int = 500,
    ) -> None:
        self.timeout = float(timeout)
        self.max_entries = int(max_entries)
        # Lazy import so the module loads even when yt-dlp isn't installed;
        # construction surfaces a clear error instead.
        try:
            import yt_dlp  # noqa: F401
        except ImportError as exc:
            raise YouTubeError(
                "yt-dlp is not installed. Run `pip install -r requirements.txt`."
            ) from exc

    def import_playlist(self, url_or_id: str) -> list[Song]:
        """Fetch a YouTube playlist and map its entries to ``Song`` records.

        Uses ``extract_flat="in_playlist"`` so we get the entry list without
        downloading each video's full metadata (one HTTP call for the
        playlist page, not one per video). Duration / uploader are not
        always present in flat mode; we degrade gracefully.
        """
        import yt_dlp

        playlist_id = parse_playlist_id(url_or_id)
        url = f"https://www.youtube.com/playlist?list={playlist_id}"
        opts = {
            "extract_flat": "in_playlist",
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "flat_playlist": True,
            "playlistend": self.max_entries,
            "noplaylist": False,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise YouTubeError(f"YouTube playlist fetch failed: {exc}") from exc

        entries = (info or {}).get("entries") or []
        songs = self._entries_to_songs(entries)
        if not songs:
            raise YouTubeError(
                "YouTube playlist is empty or has no usable entries."
            )
        return songs

    def _entries_to_songs(self, entries: list) -> list[Song]:
        """Map yt-dlp ``entries`` to ``Song`` records.

        Used by :meth:`import_playlist`. Dedupes by ``track_id`` (a video
        can appear once per call), skips entries without an id or title,
        and splits ``"Artist - Title"`` via :func:`_split_artist_title`.
        ``preview_url`` is always ``""`` — YouTube has no 30s audio clip
        equivalent (spec §21 preview invariant).
        """
        songs: list[Song] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            vid = entry.get("id")
            if not vid:
                continue
            track_id = f"youtube:{vid}"
            if track_id in seen:
                continue
            seen.add(track_id)

            title = entry.get("title") or ""
            if not title:
                continue

            artist, song_title = _split_artist_title(title)
            duration = int(entry.get("duration") or 0)
            source_url = f"https://www.youtube.com/watch?v={vid}"

            songs.append(
                Song(
                    title=song_title,
                    artist=artist,
                    source="youtube",
                    source_url=source_url,
                    preview_url="",  # No 30s audio preview clip for YouTube.
                    track_id=track_id,
                    duration=duration,
                )
            )
        return songs