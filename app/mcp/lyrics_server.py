"""lyrics-mcp — licensed lyrics tools (spec.md §12, §21).

Phase 3 implementation. Plain tool functions delegate to a
:class:`LyricsProvider` set via :func:`set_provider`, and to the
:class:`VectorStore` for ``track_id`` → song lookups. Each function is also
registered as an MCP tool via ``@mcp.tool()`` (best-effort).

Copyright enforcement (spec §21):
  - Excerpt cap (``max_lyric_excerpt_words``, default 25) is always applied.
  - Provenance (``provider`` + ``source_url``) is always attached.
  - ``syncedLyrics`` is stripped to plain text before being exposed; we
    never return raw LRC to the UI.
  - The response validator's full-lyrics-request detector (Phase 1) still
    gates "give me the entire lyrics" requests at the chat layer.

Lyrics are fetched **on-demand**, not per-chat: the UI calls
``get_excerpt(track_id)`` only when the user clicks the "Lyrics" button on a
cited song. This keeps chat fast and matches the "lyrics only when asked"
requirement (spec §21).
"""

from __future__ import annotations

from typing import Optional

from app.lyrics.base import LyricsExcerpt, LyricsProvider
from app.mcp import _make_mcp
from app.rag.vector_store import VectorStore

mcp = _make_mcp("lyrics-mcp")

_provider: Optional[LyricsProvider] = None
_store: Optional[VectorStore] = None


def _has_non_latin_script(text: str) -> bool:
    """True if ``text`` contains any alphabetic character outside the
    Latin script ranges (i.e. Devanagari, Cyrillic, Arabic, Greek, CJK,
    etc.). Accented Latin (Latin-1 supplement + Latin Extended-A/B,
    up to U+024F) is treated as Latin and left untouched. Punctuation
    (including the U+2026 ellipsis) and digits are ignored.
    """
    if not text:
        return False
    for ch in text:
        if ch.isalpha() and ord(ch) > 0x024F:
            return True
    return False


def _has_devanagari(text: str) -> bool:
    """True if ``text`` contains any Devanagari character (U+0900–U+097F)."""
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def _has_arabic_script(text: str) -> bool:
    """True if ``text`` contains any Arabic-script character
    (U+0600–U+06FF). Covers Arabic + Urdu letters."""
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


# Urdu/Arabic → Latin letter table (ASCII-only). Digraphs (kh, sh, ch,
# zh, gh) handle the composite letters. Short vowels (usually omitted
# in Urdu/Arabic script) are not recovered — the reader infers them.
# Retroflex/emphatic letters collapse to their plain ASCII equivalents
# so the output stays English-letters-only (per user request).
_URDU_MAP = {
    # Alef variants
    "\u0627": "a",  # ا alef
    "\u0623": "a",  # أ alef-hamza-above
    "\u0625": "i",  # إ alef-hamza-below
    "\u0622": "aa",  # آ alef-madda
    # Consonants
    "\u0628": "b",   # ب
    "\u067E": "p",   # پ (Urdu)
    "\u062A": "t",   # ت
    "\u0679": "t",   # ٹ (Urdu retroflex)
    "\u062B": "s",   # ث
    "\u062C": "j",   # ج
    "\u0686": "ch",  # چ (Urdu)
    "\u062D": "h",   # ح
    "\u062E": "kh",  # خ
    "\u062F": "d",   # د
    "\u0688": "d",   # ڈ (Urdu retroflex)
    "\u0630": "z",   # ذ
    "\u0631": "r",   # ر
    "\u0691": "r",   # ڑ (Urdu retroflex)
    "\u0632": "z",   # ز
    "\u0698": "zh",  # ژ (Urdu)
    "\u0633": "s",   # س
    "\u0634": "sh",  # ش
    "\u0635": "s",   # ص (emphatic → plain s)
    "\u0636": "z",   # ض (emphatic → z)
    "\u0637": "t",   # ط (emphatic → t)
    "\u0638": "z",   # ظ (emphatic → z)
    "\u0639": "",    # ع ayn — usually silent in romanization
    "\u063A": "gh",  # غ
    "\u0641": "f",   # ف
    "\u0642": "q",   # ق
    "\u06A9": "k",   # ک (Urdu kaf)
    "\u06AB": "k",   # ڭ (variant kaf)
    "\u06AF": "g",   # گ (Urdu gaf)
    "\u0644": "l",   # ل
    "\u0645": "m",   # م
    "\u0646": "n",   # ن
    "\u06BA": "n",   # ں (Urdu nun ghunnah — nasal)
    "\u0648": "w",   # و (also vowel o/u — handled contextually)
    "\u06C1": "h",   # ہ gol hay
    "\u06BE": "h",   # ھ do-cashmi (aspiration — combines with prev consonant)
    "\u0621": "",    # ء hamza
    "\u06CC": "y",   # ی (also vowel i/e — handled contextually)
    "\u06D2": "e",   # ے baṛī yeh (Urdu)
    "\u0626": "y",   # ئ hamza-y
    "\u0624": "w",   # ؤ hamza-w
    "\u06C0": "a",   # ۀ hamza-on-alef
    "\u06C2": "ah",  # ۢ alef-hamza-below variant
    "\u06C3": "ah",  # ۣ
    "\u0629": "h",   # ة ta marbuta
    # Harakat (short-vowel diacritics) — usually absent in Urdu/Arabic
    # text, but if present, recover the short vowel.
    "\u064E": "a",   # َ fatha
    "\u064F": "u",   # ُ damma
    "\u0650": "i",   # ِ kasra
    "\u0651": "",    # ّ shadda (gemination — not represented)
    "\u0652": "",    # ْ sukun (no vowel)
    "\u0670": "a",   # ٰ superscript alef
    "\u064B": "an",  # ً fathatan
    "\u064C": "un",  # ٌ dammatan
    "\u064D": "in",  # ٍ kasratan
    "\u0640": "",    # ـ tatweel
}


def _transliterate_urdu(text: str) -> str:
    """Transliterate Urdu/Arabic (Arabic script) text to Latin letters
    using a static letter table with two contextual vowel rules:

    - ``ی`` (yeh) → ``"i"`` after a consonant, ``"y"`` after a vowel /
      at a word start (so ``میری`` → ``miri`` but ``یار`` → ``yar``).
    - ``و`` (waw) → ``"o"`` after a consonant, ``"w"`` after a vowel /
      at a word start (so ``بولا`` → ``bola`` but ``وہ`` → ``wh``).

    Short vowels that are not written in the script are not recovered
    (``دل`` → ``dl``), so the reader infers them — same as standard
    Urdu romanization practice. Aspiration (``ھ`` after a consonant)
    composes naturally: ``بھ`` → ``bh``, ``کھ`` → ``kh``, ``تھ`` →
    ``th``. ASCII-only output (no diacritics).
    """
    if not text:
        return text
    out_parts: list[str] = []
    last_alpha = ""  # last output ALPHABETIC char, reset at word boundaries
    for ch in text:
        if ch.isspace():
            out_parts.append(ch)
            last_alpha = ""
            continue
        if ch not in _URDU_MAP:
            # Non-Arabic char (Latin, punctuation, etc.) — pass through.
            out_parts.append(ch)
            if ch.isalpha():
                last_alpha = ch
            continue
        if ch == "\u06CC":  # ی
            # Vowel mode ("i") after a consonant; consonant mode ("y")
            # at a word start or after another vowel (two vowels don't
            # merge). So "میری" → "miri" but "یار" → "yar".
            mapped = "i" if (last_alpha and last_alpha not in "aeiou") else "y"
        elif ch == "\u0648":  # و
            # Vowel mode ("o") after a consonant; consonant mode ("w")
            # at a word start or after another vowel. So "بولا" →
            # "bola" but "وہ" → "wh".
            mapped = "o" if (last_alpha and last_alpha not in "aeiou") else "w"
        else:
            mapped = _URDU_MAP[ch]
        out_parts.append(mapped)
        if mapped:
            last_alpha = mapped[-1]
    return "".join(out_parts)


def _romanize(text: str) -> str:
    """Transliterate non-Latin scripts to English/Latin letters.

    Three-stage, best-effort:
    1. Devanagari → OPTITRANS (ASCII-only, preserves vowels — e.g.
       ``नमस्ते`` → ``namaste``) via ``indic-transliteration`` when
       available. Falls through to stage 2 if the lib is missing.
    2. Urdu/Arabic (Arabic script) → ASCII via a built-in letter table
       with contextual vowel rules (e.g. ``میری زندگی ہے تو`` →
       ``miri zndgi he to``). No external dependency.
    3. Any remaining non-Latin script (Cyrillic, Greek, CJK, …)
       → ASCII via ``unidecode`` when available.

    Returns the original text when:
    - it is empty, or
    - it contains only Latin-script characters, or
    - no applicable transliterator is installed (graceful no-op).
    """
    if not text or not _has_non_latin_script(text):
        return text
    out = text
    # Stage 1: Devanagari → Latin (OPTITRANS, ASCII-only).
    if _has_devanagari(out):
        try:
            from indic_transliteration import sanscript
            out = sanscript.transliterate(out, sanscript.DEVANAGARI, sanscript.OPTITRANS)
        except ImportError:
            pass
    # Stage 2: Urdu/Arabic → Latin (manual table, contextual vowels).
    if _has_arabic_script(out):
        out = _transliterate_urdu(out)
    # Stage 3: any remaining non-Latin script → ASCII via unidecode.
    if _has_non_latin_script(out):
        try:
            from unidecode import unidecode
            out = unidecode(out)
        except ImportError:
            pass
    return out


def set_provider(provider: LyricsProvider | None) -> None:
    """Inject the :class:`LyricsProvider`. Pass ``None`` to disable lyrics."""
    global _provider
    _provider = provider


def set_store(store: VectorStore) -> None:
    """Inject the :class:`VectorStore` used for ``track_id`` → song lookups."""
    global _store
    _store = store


def _require_provider() -> LyricsProvider:
    if _provider is None:
        raise RuntimeError(
            "lyrics-mcp: provider not initialized. Call set_provider() first."
        )
    return _provider


def _require_store() -> VectorStore:
    if _store is None:
        raise RuntimeError(
            "lyrics-mcp: store not initialized. Call set_store() first."
        )
    return _store


def _validate_str(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_max_words(max_words, default: int = 25) -> int:
    try:
        n = int(max_words) if max_words is not None else default
    except (TypeError, ValueError) as exc:
        raise ValueError("max_words must be a positive integer") from exc
    if n <= 0:
        raise ValueError("max_words must be a positive integer")
    return n


def _excerpt_to_dict(exc: LyricsExcerpt, track_id: str) -> dict:
    romanized_text = _romanize(exc.excerpt)
    return {
        "ok": True,
        "track_id": track_id,
        "available": bool(exc.available),
        "excerpt": romanized_text,
        "provider": exc.provider,
        "source_url": exc.source_url,
        "is_synced": bool(exc.is_synced),
        # True when the excerpt was transliterated from a non-Latin script
        # (e.g. Devanagari/Cyrillic/Arabic) to English/Latin letters.
        "romanized": romanized_text != exc.excerpt,
    }


def _unavailable_dict(track_id: str, provider: str = "") -> dict:
    return {
        "ok": True,
        "track_id": track_id,
        "available": False,
        "excerpt": "",
        "provider": provider,
        "source_url": "",
        "is_synced": False,
        "romanized": False,
    }


def search_licensed_lyrics(
    title: str,
    artist: str,
    album: str = "",
    duration: int | None = None,
    max_words: int = 25,
) -> dict:
    """Look up a lyrics excerpt by (title, artist, album?, duration?). Returns
    ``{ok, available, excerpt, provider, source_url, is_synced}``."""
    t = _validate_str(title, "title")
    a = _validate_str(artist, "artist")
    mw = _validate_max_words(max_words)
    provider = _require_provider()
    exc = provider.get(t, a, album=album or "", duration=duration, max_words=mw)
    if exc is None:
        return _unavailable_dict("")
    return _excerpt_to_dict(exc, track_id=exc.track_id or "")


def get_excerpt(track_id: str, max_words: int = 25) -> dict:
    """Look up a lyrics excerpt for a song in the store by ``track_id``.

    Resolves ``(title, artist, album, duration)`` from the store, then calls
    the provider. Returns ``{ok, available, excerpt, provider, source_url,
    is_synced}``. ``available=False`` when the song isn't in the store or the
    provider has no match.
    """
    tid = _validate_str(track_id, "track_id")
    mw = _validate_max_words(max_words)
    store = _require_store()
    song = store.get(tid)
    if song is None:
        # Unknown song — can't form a provider query.
        return _unavailable_dict(tid)
    provider = _require_provider()
    exc = provider.get(
        song.title,
        song.artist,
        album=song.album or "",
        duration=song.duration or None,
        max_words=mw,
    )
    if exc is None:
        return _unavailable_dict(tid)
    # Attach the requested track_id for the UI's feedback wiring.
    return _excerpt_to_dict(exc, track_id=tid)


def get_source(track_id: str) -> dict:
    """Return provenance only — ``{ok, provider, source_url}``. Does not fetch
    the excerpt. Useful for rendering the "Lyrics source: …" label cheaply."""
    tid = _validate_str(track_id, "track_id")
    store = _require_store()
    song = store.get(tid)
    if song is None:
        return {"ok": True, "track_id": tid, "provider": "", "source_url": ""}
    # We don't fetch from the provider here (no network); just report which
    # provider would be used, plus the song's source URL.
    provider_name = ""
    if _provider is not None:
        provider_name = getattr(_provider, "__class__", type(_provider)).__name__
    return {
        "ok": True,
        "track_id": tid,
        "provider": provider_name,
        "source_url": song.source_url,
    }


# ---- FastMCP tool registration (best-effort, no-op if SDK unavailable) ----
if mcp is not None:
    for _fn in (search_licensed_lyrics, get_excerpt, get_source):
        try:
            mcp.tool()(_fn)
        except Exception:
            pass