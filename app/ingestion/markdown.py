"""Markdown playlist parser.

Accepts lines of the form::

    - Title — Artist
    - Title – Artist
    - Title - Artist

Headings (``# ...``), blank lines, and lines without a separator are ignored.
The parser is tolerant of em dash (``\\u2014``), en dash (``\\u2013``) and
hyphen (``-``) separators.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Separators: em dash, en dash, hyphen surrounded by spaces, or hyphen with
# surrounding whitespace. We require whitespace around the hyphen so titles
# like "Half-Life" are not split accidentally.
_SEPARATOR = re.compile(r"\s+[\u2014\u2013\-]\s+|[\u2014\u2013]")

# Optional trailing release-date annotation: "(2019)" or "(2019-05-12)".
_YEAR_RE = re.compile(r"\s*\((\d{4}(?:-\d{2}-\d{2})?)\)\s*$")


@dataclass
class Song:
    """A normalized song record (spec.md §22 subset).

    Enrichment fields (Phase 3 RAG upgrade):
    - ``index_lyrics``: plain lyrics (capped to ~``lyrics_index_words``) used
      to enrich the indexed text. NEVER exposed in ``to_dict()`` (copyright:
      the LLM never sees indexed lyrics — only the title/artist metadata
      passed into ``generate()``).
    - ``tags``: mood/theme/genre tags (e.g. "sad", "upbeat", "synthwave")
      derived from LLM world knowledge. Indexed and exposed in
      ``to_dict()`` (tags are non-copyrightable descriptors).
    - ``audio_features``: optional dict of audio-feature values
      (valence/energy/danceability/tempo/...). Indexed indirectly via
      ``tags`` and exposed in ``to_dict()`` for debugging/UI.
    """

    title: str
    artist: str
    source: str = "markdown"
    source_url: str = ""
    album: str = ""
    release_date: str = ""
    genre: list[str] = field(default_factory=list)
    language: str = ""
    duration: int = 0
    track_id: str = ""
    preview_url: str = ""
    index_lyrics: str = ""
    tags: list[str] = field(default_factory=list)
    audio_features: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.track_id:
            self.track_id = _deterministic_id(self.title, self.artist)

    @property
    def text(self) -> str:
        """Compact display representation: ``"Title — Artist"``.

        Used for citations and anywhere a short human-readable label is
        needed. Not the indexed text — see :attr:`index_text`.
        """
        return f"{self.title} \u2014 {self.artist}".strip()

    @property
    def index_text(self) -> str:
        """Text actually embedded into FAISS and scored by BM25.

        Combines the compact title/artist line with enrichment tags and a
        capped lyric excerpt. This is what makes the index content-rich:
        a query like "sad synthwave" can match songs whose lyrics/tags
        contain those terms even when the title/artist line does not.

        ``index_lyrics`` is only used here (for indexing) and never
        returned to the LLM in candidate metadata (copyright invariant).
        """
        parts: list[str] = [self.text]
        if self.tags:
            parts.append(" ".join(self.tags))
        if self.index_lyrics:
            parts.append(self.index_lyrics)
        return " — ".join(parts)

    def to_dict(self) -> dict:
        # Note: ``index_lyrics`` is intentionally omitted. Indexed lyrics
        # are a derivative of copyrighted material and must not be exposed
        # to the LLM or UI (spec.md §21). ``tags`` and ``audio_features``
        # are non-copyrightable descriptors and are safe to expose.
        return {
            "track_id": self.track_id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "release_date": self.release_date,
            "genre": self.genre,
            "language": self.language,
            "duration": self.duration,
            "source": self.source,
            "source_url": self.source_url,
            "preview_url": self.preview_url,
            "tags": list(self.tags),
            "audio_features": dict(self.audio_features),
        }


def _deterministic_id(title: str, artist: str) -> str:
    raw = f"{title.strip().lower()}|{artist.strip().lower()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def parse_playlist(text: str) -> list[Song]:
    """Parse a Markdown playlist string into a list of :class:`Song`.

    Headings and blank lines are ignored. Lines without a recognizable
    title/artist separator are skipped.
    """
    songs: list[Song] = []
    seen: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Headings and section titles are not songs.
            continue
        if not line.startswith("-"):
            # Only bullet lines are treated as songs; be lenient about "* ".
            if line.startswith("*"):
                line = line[1:].lstrip()
            else:
                continue

        body = line.lstrip("-").lstrip("*").strip()
        if not body:
            continue

        parts = _SEPARATOR.split(body, maxsplit=1)
        if len(parts) != 2:
            continue

        title = parts[0].strip()
        artist = parts[1].strip()
        if not title or not artist:
            continue

        # Optional trailing "(YYYY)" or "(YYYY-MM-DD)" on the artist field.
        release_date = ""
        m = _YEAR_RE.search(artist)
        if m:
            release_date = m.group(1)
            artist = _YEAR_RE.sub("", artist).strip()
            if not artist:
                continue

        song = Song(title=title, artist=artist, release_date=release_date)
        if song.track_id in seen:
            continue
        seen.add(song.track_id)
        songs.append(song)

    return songs