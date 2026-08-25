"""Antakshari mode (spec.md §16).

An interactive music game: the AI sings a song, the user must reply with a
song whose title starts with the last letter of the AI's song, then the AI
picks a song whose title starts with the last letter of the user's song,
and so on. State is held in-memory on :class:`MusicAssistant` (not
persisted); a "Stop game" button resets it.

No LLM is needed for the core loop — selection is deterministic (first
matching song in store order). The "LLM never touches DBs" invariant
holds: the mode reads the store via ``assistant.store.get`` /
``assistant.store.all_songs()`` (the same access pattern as the existing
MCP-backed ``get_lyrics_excerpt`` path).

MVP supports the first/last-letter rule (spec §16 lists other optional
rules — language/decade/genre/artist — as out of scope).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.markdown import Song


def _last_alpha(title: str) -> str:
    """Return the last alphabetic character of ``title`` (uppercased), or
    the empty string when ``title`` has no letters."""
    for ch in reversed(title or ""):
        if ch.isalpha():
            return ch.upper()
    return ""


def _first_alpha(title: str) -> str:
    """Return the first alphabetic character of ``title`` (uppercased)."""
    for ch in title or "":
        if ch.isalpha():
            return ch.upper()
    return ""


@dataclass
class AntakshariSession:
    """In-memory Antakshari game state (spec §16).

    ``current_song`` is the AI's last picked song; ``required_character`` is
    the letter the user's next answer must start with (the last letter of
    ``current_song.title``). ``used_songs`` / ``used_artists`` prevent
    repeats. ``score`` increments by 1 per valid user answer.
    """

    current_song: Song | None = None
    required_character: str = ""
    used_songs: set[str] = field(default_factory=set)
    used_artists: set[str] = field(default_factory=set)
    score: int = 0
    active: bool = False

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "current_song": (
                {
                    "title": self.current_song.title,
                    "artist": self.current_song.artist,
                    "track_id": self.current_song.track_id,
                }
                if self.current_song
                else None
            ),
            "required_character": self.required_character,
            "used_songs_count": len(self.used_songs),
            "used_artists_count": len(self.used_artists),
            "score": self.score,
        }


def _pick_ai_song(assistant, start_char: str, used_songs: set[str], used_artists: set[str]) -> Song | None:
    """Pick the first song in store order whose title starts with
    ``start_char`` and that hasn't been used yet (by track_id or artist).

    Falls back to ignoring the artist restriction if no unused song by a
    new artist matches (so the game can continue when the collection is
    small). Returns ``None`` when no song matches the letter at all.
    """
    songs = assistant.store.all_songs()
    # First pass: respect both the song and artist used-sets.
    for s in songs:
        if s.track_id in used_songs:
            continue
        if (s.artist or "").strip().lower() in used_artists:
            continue
        if _first_alpha(s.title) == start_char:
            return s
    # Second pass: only enforce the song used-set (allow repeat artists).
    for s in songs:
        if s.track_id in used_songs:
            continue
        if _first_alpha(s.title) == start_char:
            return s
    return None


def start(assistant) -> dict:
    """Start a new Antakshari session. Picks the AI's opening song from
    the store (first song with a usable last letter), sets
    ``required_character`` to that song's last letter, and returns the
    initial state dict for the UI to render."""
    session = AntakshariSession(active=True)
    # Pick any opening song — the first one in store order with a usable
    # last letter.
    opening: Song | None = None
    for s in assistant.store.all_songs():
        last = _last_alpha(s.title)
        if last:
            opening = s
            break
    if opening is None:
        return {
            "ok": False,
            "error": "No songs in the collection can start the game.",
            "session": session.to_dict(),
        }
    session.current_song = opening
    session.required_character = _last_alpha(opening.title)
    session.used_songs = {opening.track_id}
    session.used_artists = {(opening.artist or "").strip().lower()}
    session.score = 0
    session.active = True
    assistant.antakshari = session
    return {
        "ok": True,
        "message": (
            f"Let's play Antakshari! I'll start with "
            f"\u201c{opening.title}\u201d by {opening.artist}. "
            f"Your turn: give me a song that starts with the letter "
            f"\u201c{session.required_character}\u201d."
        ),
        "session": session.to_dict(),
    }


def submit(assistant, user_answer: str) -> dict:
    """Validate the user's answer and advance the round.

    Validation (spec §16):
      1. The named song exists in the store (matched by title, case-
         insensitive, ignoring leading articles like "The"/"A").
      2. The title starts with ``required_character``.
      3. The song hasn't already been used (by track_id) and the artist
         hasn't already been used (warn only — we don't hard-block repeat
         artists so small collections stay playable; the first pass of
         AI selection prefers new artists).

    On a valid answer: increment the score, mark the song + artist as
    used, pick the AI's next song (starting with the last letter of the
    user's title), update ``current_song`` / ``required_character``, and
    return the new state. On an invalid answer: return the reason + the
    unchanged state so the UI can prompt the user to try again.
    """
    session = assistant.antakshari
    if session is None or not session.active:
        # Auto-start if the user typed an answer without an active session.
        return start(assistant)

    answer = (user_answer or "").strip()
    if not answer:
        return {
            "ok": False,
            "error": "Please type a song title.",
            "session": session.to_dict(),
        }

    # Match the user's answer to a song in the store by title.
    normalized = answer.lower()
    # Strip a leading "the "/"a "/"an " for matching so "The Police" /
    # "Police" round-trip; we match on the title only (Antakshari is a
    # title game).
    normalized_bare = re.sub(r"^(the|a|an)\s+", "", normalized)
    matched: Song | None = None
    for s in assistant.store.all_songs():
        t = s.title.lower()
        if t == normalized or re.sub(r"^(the|a|an)\s+", "", t) == normalized_bare:
            matched = s
            break

    if matched is None:
        return {
            "ok": False,
            "error": (
                f"\u201c{answer}\u201d isn't in your collection. Pick a song "
                f"from your playlist."
            ),
            "session": session.to_dict(),
        }

    if matched.track_id in session.used_songs:
        return {
            "ok": False,
            "error": f"\u201c{matched.title}\u201d has already been used. Pick another.",
            "session": session.to_dict(),
        }

    first = _first_alpha(matched.title)
    if session.required_character and first != session.required_character:
        return {
            "ok": False,
            "error": (
                f"\u201c{matched.title}\u201d starts with \u201c{first}\u201d, "
                f"but the required letter is \u201c{session.required_character}\u201d."
            ),
            "session": session.to_dict(),
        }

    # Valid answer — advance the round.
    session.score += 1
    session.used_songs.add(matched.track_id)
    session.used_artists.add((matched.artist or "").strip().lower())

    next_char = _last_alpha(matched.title)
    ai_song = _pick_ai_song(assistant, next_char, session.used_songs, session.used_artists)
    if ai_song is None:
        # No AI reply available — the user wins this round.
        session.current_song = None
        session.required_character = ""
        session.active = False
        return {
            "ok": True,
            "won": True,
            "message": (
                f"Nice! \u201c{matched.title}\u201d is valid. I can't find a "
                f"song in your collection that starts with \u201c{next_char}\u201d "
                f"\u2014 you win this round! Final score: {session.score}."
            ),
            "session": session.to_dict(),
        }

    session.current_song = ai_song
    session.required_character = _last_alpha(ai_song.title)
    session.used_songs.add(ai_song.track_id)
    session.used_artists.add((ai_song.artist or "").strip().lower())
    return {
        "ok": True,
        "won": False,
        "message": (
            f"Good! \u201c{matched.title}\u201d is valid. My turn: "
            f"\u201c{ai_song.title}\u201d by {ai_song.artist}. "
            f"Your turn: give me a song that starts with \u201c{session.required_character}\u201d."
        ),
        "session": session.to_dict(),
    }


def stop(assistant) -> dict:
    """End the current Antakshari session (the "Stop game" button)."""
    session = assistant.antakshari
    if session is None:
        return {"ok": True, "message": "No active Antakshari session.", "session": None}
    final_score = session.score
    assistant.antakshari = None
    return {
        "ok": True,
        "message": f"Antakshari stopped. Final score: {final_score}.",
        "session": None,
    }