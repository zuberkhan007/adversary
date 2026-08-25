"""Lyrics package: pluggable provider seam (spec.md §12, §21).

Default provider is LRCLIB (free, no key, language-agnostic). A
``MusixmatchProvider`` stub is included for better Hindi/Urdu coverage once
the user adds a user token. The orchestrator picks the provider via
``config.lyrics.provider``.
"""

from app.lyrics.base import LyricsExcerpt, LyricsProvider, cap_words, strip_lrc

__all__ = ["LyricsExcerpt", "LyricsProvider", "cap_words", "strip_lrc"]