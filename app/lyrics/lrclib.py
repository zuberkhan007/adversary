"""LRCLIB lyrics provider (https://lrclib.net/docs).

Free, no API key, language-agnostic, royalty-licensed. Two endpoints:

* ``GET /api/get?track_name=...&artist_name=...&album_name=...&duration=...``
  → exact-ish match (404 when album/duration don't line up).
* ``GET /api/search?q=...`` → array (max 20) for fuzzy matches. Used as a
  fallback when ``/api/get`` returns 404.

Required: ``User-Agent`` header identifying the client. Rate limit: 429 with
``Retry-After``; we serialize requests (no concurrency) and sleep 200ms
between calls to stay polite.

Excerpt extraction: strip LRC timing tags from ``syncedLyrics`` (or use
``plainLyrics``), take the first ``max_words`` words, append ``…``. Always
attach ``Lyrics source: LRCLIB``.
"""

from __future__ import annotations

import time
from urllib.parse import quote_plus

import requests

from app.lyrics.base import LyricsExcerpt, cap_lines, strip_lrc


class LRCLIBProvider:
    """LRCLIB lyrics provider (no API key required)."""

    def __init__(
        self,
        base_url: str = "https://lrclib.net/api",
        user_agent: str = "OHH/1.0",
        timeout: float = 10.0,
        max_calls_per_sec: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = float(timeout)
        # Polite pacing: at most 5 calls/sec (>=200ms between requests).
        self._min_interval = 1.0 / max(1.0, max_calls_per_sec)
        self._last_call_at = 0.0

    # ---- internal ----------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent}

    def _pacing_sleep(self) -> None:
        now = time.time()
        delta = now - self._last_call_at
        if delta < self._min_interval:
            time.sleep(self._min_interval - delta)
        self._last_call_at = time.time()

    def _get(self, path: str, params: dict[str, str]) -> dict | list | None:
        url = f"{self.base_url}/{path.lstrip('/')}"
        self._pacing_sleep()
        try:
            resp = requests.get(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
        except (requests.RequestException, OSError) as exc:
            # Network failure → treat as "not available" rather than crashing
            # the chat path. The orchestrator surfaces this as a soft failure.
            # ``OSError`` covers low-level socket/SSL errors (e.g.
            # ``[Errno 22] Invalid argument`` on Windows) that ``requests``
            # doesn't always wrap in ``RequestException``.
            return None

        if resp.status_code == 404:
            return None

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "1") or 1)
            retry_after = max(0.1, min(retry_after, 10.0))
            time.sleep(retry_after)
            try:
                resp = requests.get(
                    url, params=params, headers=self._headers(), timeout=self.timeout
                )
            except (requests.RequestException, OSError):
                return None
            if resp.status_code != 200:
                return None

        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    # ---- public ------------------------------------------------------
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

        params: dict[str, str] = {
            "track_name": title,
            "artist_name": artist,
        }
        if album:
            params["album_name"] = album
        if duration is not None and duration > 0:
            params["duration"] = str(int(duration))

        payload = self._get("/get", params)
        if payload is None:
            # Fallback: fuzzy search.
            q = f"{title} {artist}".strip()
            results = self._get("/search", {"q": q})
            if isinstance(results, list) and results:
                payload = results[0]
            else:
                payload = None

        if not isinstance(payload, dict) or not payload:
            return LyricsExcerpt(
                track_id="",
                available=False,
                excerpt="",
                provider="LRCLIB",
                source_url="",
            )

        synced = payload.get("syncedLyrics") or ""
        plain = payload.get("plainLyrics") or ""
        is_synced = bool(synced)
        raw = strip_lrc(synced) if is_synced else (plain or "")
        if not raw:
            # Nothing usable — fall back to plainLyrics if we had synced only.
            raw = plain or ""

        excerpt = cap_lines(raw, max_words) if max_words > 0 else (raw.strip())

        track_url = (
            payload.get("githubUrl")
            or payload.get("link")
            or "https://lrclib.net/"
        )

        return LyricsExcerpt(
            track_id="",
            available=bool(excerpt),
            excerpt=excerpt,
            provider="LRCLIB",
            source_url=str(track_url),
            is_synced=is_synced,
        )