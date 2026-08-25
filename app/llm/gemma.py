"""Gemma client over Ollama's OpenAI-compatible HTTP API.

Uses ``requests`` (kept dependency-light). The model identifier is passed in
from configuration and never hard-coded here.

System prompts forbid fabricating lyrics/titles/artists and instruct the model
to only use the provided candidate metadata.
"""

from __future__ import annotations

from typing import Protocol

import requests


class Logger(Protocol):
    def info(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


class _StdLogger:
    def info(self, msg: str) -> None:
        print(f"[gemma] {msg}")

    def error(self, msg: str) -> None:
        print(f"[gemma][error] {msg}")


_EXTRACT_SYSTEM = (
    "You are a query rewriter for a music search engine. Given the user's "
    "latest message and the recent conversation, produce a single concise "
    "retrieval query string (no quotes, no preamble) that captures what the "
    "user wants songs about. Resolve follow-ups against prior context — e.g. "
    "'something older' means the same topic as before but with an older "
    "preference; do NOT treat follow-ups as unrelated searches. Output only "
    "the query string."
)

_GENERATE_SYSTEM = (
    "You are a music recommendation assistant. You will be given a list of "
    "candidate songs from the user's collection, with title and artist. "
    "Use ONLY these candidates. Never invent song titles, artists, or albums. "
    "Never fabricate lyrics. You may give a short, permitted excerpt of "
    "lyrics only if it is well-known and at most a few words; otherwise "
    "describe the song's meaning/theme. Always cite the song and artist. "
    "Do not output complete or large portions of copyrighted lyrics. If no "
    "candidate is relevant, say so honestly. Distinguish any generated text "
    "from metadata — do not present generated text as an original lyric."
)

_LYRIC_INTRO_SYSTEM = (
    "You are a music conversational assistant in a natural chat. The user "
    "described a mood, feeling, or request. Several short licensed lyric "
    "phrases have been fetched from songs in their collection. Your job "
    "is to weave two or three of those phrases into ONE coherent English "
    "sentence that captures what the user is feeling. The lyric phrases "
    "are the emotional heart of the sentence; your connective words (in "
    "English) just make them flow together naturally so the whole thing "
    "reads as a single, natural thought \u2014 not a list of "
    "disconnected quotes. Wrap each lyric phrase in curly quotes "
    "(\u201c\u201d) so it is distinguishable from your connective text, "
    "but arrange the quotes so the sentence still flows naturally. Pick "
    "the phrases that best fit together to express the mood. Quote each "
    "phrase EXACTLY as it appears, in its ORIGINAL language \u2014 do "
    "NOT translate the lyrics. All your own connective words (between "
    "and around the quotes) MUST be in English, regardless of the "
    "language of the lyrics or the user's message. Use ONLY the phrases "
    "provided \u2014 do NOT extend, paraphrase, or invent any lyrics "
    "beyond them. Do NOT present generated text as an original lyric. "
    "NEVER mention the song title or artist name in your response \u2014 "
    "the system shows the song name separately. NEVER explain, review, "
    "describe, analyze, or comment on the song itself. Keep it to one or "
    "two sentences. Output only the response text (no preamble, no "
    "provenance block)."
)

_SELECT_SONG_SYSTEM = (
    "You are a music selection assistant. Given a user's request and a "
    "list of candidate songs from their collection, pick the ONE song that "
    "best matches the user's mood, theme, or request. Use your knowledge "
    "of the songs (their mood, themes, energy, era, lyrics) \u2014 not just "
    "title keyword matching. Consider the conversation context for "
    "follow-ups (e.g. 'something older' means the same topic but older). "
    "Return ONLY the number of the best match, nothing else."
)

_TAGS_SYSTEM = (
    "You are a music metadata tagger. Given a song title and artist, "
    "output 3-5 lowercase mood/theme/genre tags that describe the song's "
    "vibe (e.g. 'sad', 'upbeat', 'synthwave', 'romantic', 'aggressive', "
    "'nostalgic', 'dreamy', 'energetic'). Use your knowledge of the song "
    "\u2014 its sound, mood, themes, era, and lyrics. Output ONLY "
    "comma-separated tags, nothing else. If you don't know the song, "
    "output nothing."
)

# Phase 4 — interaction-mode prompts (spec §14–17).
_INTENT_SYSTEM = (
    "You are an intent router for a music assistant. Classify the user's "
    "message into exactly one of these modes:\n"
    "- explain: the user wants a concept explained using songs as examples "
    "(e.g. 'Explain heartbreak using songs from my playlist').\n"
    "- antakshari: the user wants to play the Antakshari word-chain game "
    "(e.g. 'Let's play Antakshari').\n"
    "- generation: the user wants a generated playlist / thematic mix / "
    "remix around a set of themes (e.g. 'Make a playlist around rain, "
    "loneliness and midnight').\n"
    "- list: the user wants to see ALL the songs in their collection / "
    "index (e.g. 'list all the songs indexed', 'show all my songs', "
    "'what's in my collection').\n"
    "- general: anything else, including a plain song request or mood "
    "query.\n"
    "Output ONLY the mode name (one word, lowercase), nothing else."
)

_EXPLAIN_SYSTEM = (
    "You are a music explain-mode assistant. The user wants a concept "
    "explained using songs from their collection as contextual examples. "
    "You will be given the concept (retrieval query) and a list of "
    "candidate songs (title + artist), each accompanied by a few short "
    "licensed lyric phrases (when available). Use ONLY these candidates. "
    "Never invent songs, artists, or albums. Never fabricate lyrics. "
    "Do not output complete or large portions of copyrighted lyrics. "
    "Distinguish any generated text from metadata.\n\n"
    "Write ONE paragraph per candidate song that illustrates the concept. "
    "Each paragraph should be a lyrical conversation: weave the provided "
    "lyric phrases for that song into the explanation as quoted lines in "
    "curly quotes, EXACTLY as written and in their ORIGINAL language "
    "(do NOT translate them). The lyric phrases should dominate the "
    "paragraph; the conversational/explanatory text between them MUST be "
    "in English and should be minimal \u2014 just enough to tie the "
    "phrases to the concept. If a song has no provided lyric phrases, "
    "write a short prose explanation of how its mood/theme illustrates "
    "the concept.\n\n"
    "At the END of each paragraph, on its own line, cite the source song "
    "as 'Artist \u2014 Title'. Do NOT mention the song title or artist "
    "name inside the paragraph body \u2014 only in the closing citation "
    "line. Do not add a separate provenance block; the per-paragraph "
    "citation is the provenance. Keep the whole response focused and "
    "readable."
)

_GENERATION_SYSTEM = (
    "You are a music generation-mode assistant. The user wants a thematic "
    "playlist / mix around a set of themes. You will be given the themes "
    "(retrieval query) and a list of candidate songs (title + artist) "
    "drawn from the user's collection by semantic + diversity ranking. "
    "Use ONLY these candidates. Never invent songs, artists, or albums. "
    "Allowed outputs (spec §17): a playlist, a thematic description, a "
    "short permitted lyric excerpt (a few words at most), song "
    "transitions, or a NEW ORIGINAL lyric inspired by the high-level "
    "themes \u2014 NEVER concatenate or paraphrase substantial portions "
    "of copyrighted lyrics. Always cite each song as 'Artist — Title'. "
    "Keep the response focused and readable."
)

_LIST_SYSTEM = (
    "You are a music assistant. The user asked to see the songs in their "
    "collection. You will be given the list of songs (title + artist, in "
    "order). Format them as a clean numbered list, one per line, as "
    "'N. Artist \u2014 Title'. Prefix the list with ONE short "
    "conversational sentence acknowledging the request and stating how "
    "many songs are listed. Do NOT invent songs; use ONLY the provided "
    "list, in the order given. Do NOT add lyric text, commentary, or "
    "recommendations \u2014 just the count line and the numbered list."
)


class Gemma:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma4:e2b",
        temperature: float = 0.3,
        timeout: float = 60.0,
        logger: Logger | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.logger = logger or _StdLogger()
        self.api_key = api_key

    def ensure_configured(self) -> None:
        """Raise a clear error when a cloud model is configured without an
        API key.

        Ollama ``:cloud`` models proxy through Ollama's hub and require
        ``OLLAMA_API_KEY`` for auth/billing. Without it the request fails
        with HTTP 401. Calling this before the request lets the chat path
        surface an actionable message instead of silently degrading
        (``extract_query`` failure is otherwise swallowed for retrieval-only
        testing). Local models (no ``:cloud`` suffix) never need a key.
        """
        if self.model.endswith(":cloud") and not self.api_key:
            raise RuntimeError(
                f"Cloud model '{self.model}' requires an API key, but none is "
                "set. Set OLLAMA_API_KEY in .env (or via the UI's API Key "
                "panel), or switch config.yaml ollama.model to a local model."
            )

    def _chat(self, system: str, user: str, history: list[dict] | None = None) -> str:
        self.ensure_configured()
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in history or []:
            messages.append(turn)
        messages.append({"role": "user", "content": user})

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        openai_body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        native_body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        resp = None
        # Primary: OpenAI-compatible endpoint (local Ollama).
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=openai_body,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            # Fallback: native /api/chat. Ollama Cloud only exposes this
            # endpoint (its /v1/chat/completions returns 405); local Ollama
            # supports it too, so the fallback is safe.
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=native_body,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
            except (requests.RequestException, OSError) as exc2:
                # ``OSError`` covers low-level socket/SSL errors (e.g.
                # ``[Errno 22] Invalid argument`` on Windows) that ``requests``
                # doesn't always wrap in ``RequestException``.
                self.logger.error(f"Ollama request failed: {exc2}")
                raise RuntimeError(f"Ollama request failed: {exc2}") from exc2
        except (requests.RequestException, OSError) as exc:
            self.logger.error(f"Ollama request failed: {exc}")
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        data = resp.json()
        # Parse both response shapes: OpenAI ({choices[0].message.content})
        # and native ({message.content}).
        content: str | None = None
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        if content is None:
            msg = data.get("message") if isinstance(data, dict) else None
            if isinstance(msg, dict):
                content = msg.get("content")
        if content is None:
            self.logger.error(f"Unexpected Ollama response: {data}")
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return str(content).strip()

    def extract_query(self, user_message: str, history: list[dict] | None = None) -> str:
        """Rewrite the user turn into a retrieval query (handles §29 context)."""
        return self._chat(_EXTRACT_SYSTEM, user_message, history)

    def generate(self, query: str, candidates: list[dict], history: list[dict] | None = None) -> str:
        """Generate a grounded response using only the provided candidates."""
        cand_text = "\n".join(
            f"- {c['title']} \u2014 {c['artist']}" for c in candidates
        ) if candidates else "(no candidates)"
        user = (
            f"Retrieval query: {query}\n\n"
            f"Candidate songs (use ONLY these):\n{cand_text}\n\n"
            "Answer the user's request. Recommend at most 1-3 songs and give a "
            "short explanation for each. Cite song and artist as 'Artist — Title'."
        )
        return self._chat(_GENERATE_SYSTEM, user, history)

    def generate_lyric_intro(
        self,
        query: str,
        song: dict,
        lyric_excerpt: str,
        history: list[dict] | None = None,
        extra_phrases: list[str] | None = None,
    ) -> str:
        """Write the conversational body framing licensed lyric phrases.

        The phrases come from the lyrics provider (LRCLIB/etc.) — the LLM
        weaves them as quoted lyrical phrases with minimal conversational
        text between them. It never generates lyric text beyond the
        provided phrases, never names the song/artist (the system shows
        the song name separately), and never explains/reviews the song.
        The final response is assembled by the orchestrator: this body +
        provenance.
        """
        tags = song.get("tags") or []
        tags_line = f"\nMood/theme tags for tone: {', '.join(tags)}\n" if tags else ""
        # Build the phrase list. The primary phrase is ``lyric_excerpt``;
        # any additional relevant phrases are appended so the LLM can weave
        # multiple lyrical lines with minimal conversational text.
        all_phrases = [lyric_excerpt] + list(extra_phrases or [])
        # Dedupe while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for p in all_phrases:
            if not p:
                continue
            k = p.lower().strip()
            if k in seen:
                continue
            seen.add(k)
            unique.append(p)
        phrase_lines = "\n".join(f"\u201c{p}\u201d" for p in unique)
        user = (
            f"User's request: {query}\n\n"
            f"Licensed lyric phrases (use ONLY these \u2014 do NOT "
            f"extend, paraphrase, translate, or invent any lyrics beyond "
            f"them):\n"
            f"{phrase_lines}\n"
            f"{tags_line}\n"
            "Pick the two or three phrases that best flow together to "
            "express what the user is feeling, and weave them into ONE "
            "coherent English sentence. Wrap each chosen phrase in curly "
            "quotes (\u201c\u201d) EXACTLY as written, in its original "
            "language \u2014 do NOT translate them. Your connective words "
            "(between and around the quotes) MUST be in English and "
            "should be minimal \u2014 just enough to make the lyrics "
            "read as a single natural sentence that captures the mood. "
            "Do NOT just list the quotes one after another; weave them "
            "into a sentence with your connective words. Do NOT mention "
            "the song title or artist name. Do NOT explain, review, "
            "describe, or analyze the song. Do not add any lyrics beyond "
            "the provided phrases. Keep it to one or two sentences. "
            "Output only the response text."
        )
        return self._chat(_LYRIC_INTRO_SYSTEM, user, history)

    def select_song(
        self,
        query: str,
        candidates: list[dict],
        history: list[dict] | None = None,
    ) -> str | None:
        """Pick the best-matching song from ``candidates`` for the query.

        Uses the model's world knowledge of song moods/themes/eras (not just
        title keyword matching) to pick the one that best fits the user's
        request. Returns the selected ``track_id``, or ``None`` if the
        response couldn't be parsed (the caller falls back to retrieval
        top-1).
        """
        if not candidates:
            return None
        import re

        numbered = "\n".join(
            f"{i + 1}. {c.get('artist', '')} \u2014 {c.get('title', '')}"
            + (f" [tags: {', '.join(c.get('tags') or [])}]" if c.get('tags') else "")
            for i, c in enumerate(candidates)
        )
        user = (
            f"User's request: {query}\n\n"
            f"Candidate songs (from the user's collection):\n{numbered}\n\n"
            f"Return ONLY the number (1-{len(candidates)}) of the best match."
        )
        try:
            raw = self._chat(_SELECT_SONG_SYSTEM, user, history)
        except RuntimeError:
            return None
        m = re.search(r"\b(\d+)\b", raw or "")
        if not m:
            return None
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx].get("track_id")
        return None

    def generate_tags(self, title: str, artist: str) -> list[str]:
        """Generate 3-5 lowercase mood/theme/genre tags for a song.

        Used by the LLM tag enricher for Markdown / YouTube sources.
        Returns an empty list on any failure (parsing or LLM error) —
        enrichment is best-effort.
        """
        user = f"Title: {title}\nArtist: {artist}"
        try:
            raw = self._chat(_TAGS_SYSTEM, user)
        except RuntimeError:
            return []
        return [t.strip().lower() for t in (raw or "").split(",") if t.strip()]

    # ---- Phase 4: interaction modes ----------------------------------
    def detect_intent(self, user_message: str, history: list[dict] | None = None) -> str:
        """Classify the user's message into a mode label (spec §14).

        Returns one of ``"explain"`` / ``"antakshari"`` / ``"generation"``
        / ``"general"``. The caller (:func:`detect_intent` in
        ``app.modes.router``) maps the raw reply onto an :class:`Intent`
        by substring match, with a keyword-rule fallback when this call
        fails or returns garbage.
        """
        return self._chat(_INTENT_SYSTEM, user_message, history)

    def generate_explain(
        self,
        query: str,
        candidates: list[dict],
        history: list[dict] | None = None,
        song_phrases: list[list[str]] | None = None,
    ) -> str:
        """Explain a concept using the provided candidate songs (spec §15).

        Low-temperature, grounded in candidate metadata + short licensed
        lyric phrases (when available). Produces one paragraph per song
        weaving the lyric phrases as quoted lines (original language) with
        minimal English conversational text, ending each paragraph with an
        ``Artist \u2014 Title`` citation. The response is routed through
        :func:`validate` by the caller so the full-lyrics-request gate
        and excerpt cap still apply.
        """
        if not candidates:
            return "No matching songs in your collection to explain that with."
        phrases_per = song_phrases or []
        # Pad to align with candidates (missing entries → no phrases).
        while len(phrases_per) < len(candidates):
            phrases_per.append([])
        blocks: list[str] = []
        for i, c in enumerate(candidates):
            head = f"{i + 1}. {c.get('artist', '')} \u2014 {c.get('title', '')}"
            ph = phrases_per[i] if i < len(phrases_per) else []
            if ph:
                phrase_lines = "\n".join(f"\u201c{p}\u201d" for p in ph)
                blocks.append(f"{head}\nLicensed lyric phrases (quote EXACTLY, in original language, do NOT translate):\n{phrase_lines}")
            else:
                blocks.append(f"{head}\n(no lyric phrases available for this song \u2014 write a short prose explanation of how its mood/theme illustrates the concept)")
        cand_text = "\n\n".join(blocks)
        user = (
            f"Concept to explain: {query}\n\n"
            f"Candidate songs from the user's collection (use ONLY these):\n"
            f"{cand_text}\n\n"
            "Write ONE paragraph per candidate song above, in the same order. "
            "For songs with lyric phrases, weave those phrases into the "
            "paragraph as quoted lines in curly quotes, EXACTLY as written "
            "and in their ORIGINAL language (do NOT translate them); use "
            "minimal English conversational text between them to tie them "
            "to the concept. For songs without lyric phrases, write a short "
            "prose explanation. At the END of each paragraph, on its own "
            "line, cite the song as 'Artist \u2014 Title'. Do NOT mention "
            "the song title or artist inside the paragraph body \u2014 only "
            "in the closing citation line. Do not add any other provenance "
            "block. Keep it focused and readable."
        )
        # Use a low temperature for explain mode (spec §8: low temperature
        # for explain / exact metadata retrieval).
        saved = self.temperature
        try:
            self.temperature = 0.3
            return self._chat(_EXPLAIN_SYSTEM, user, history)
        finally:
            self.temperature = saved

    def generate_generation(
        self, query: str, candidates: list[dict], history: list[dict] | None = None
    ) -> str:
        """Generate a thematic playlist / description (spec §17).

        Higher-temperature generation. The prompt forbids concatenating
        copyrighted lyrics; allowed outputs are a playlist / thematic
        description / short permitted excerpt / song transitions / a new
        original lyric inspired by high-level themes. The response is
        routed through :func:`validate` by the caller.
        """
        cand_text = "\n".join(
            f"- {c['artist']} \u2014 {c['title']}" for c in candidates
        ) if candidates else "(no candidates)"
        user = (
            f"Themes: {query}\n\n"
            f"Candidate songs (use ONLY these):\n{cand_text}\n\n"
            "Build a thematic playlist around the themes. Order the songs "
            "with brief transitions, give a short thematic description, "
            "and (optionally) a NEW original lyric line inspired by the "
            "high-level themes \u2014 do NOT quote or paraphrase existing "
            "lyrics. Cite each song as 'Artist — Title'."
        )
        saved = self.temperature
        try:
            self.temperature = 0.7
            return self._chat(_GENERATION_SYSTEM, user, history)
        finally:
            self.temperature = saved

    def generate_list(
        self, query: str, candidates: list[dict], history: list[dict] | None = None
    ) -> str:
        """Format the full indexed collection as a list (list mode).

        Low-temperature, grounded in the provided candidate list (title +
        artist). The LLM only formats the provided songs \u2014 it never
        invents songs and never touches the store. ``query`` is the
        user's original message (passed for context / future filtering).
        The response is routed through :func:`validate` by the caller.
        """
        if not candidates:
            return "Your collection is empty \u2014 ingest a playlist to add songs."
        # Provide the songs already numbered so the LLM just echoes them in
        # order. This keeps the output deterministic and bounded.
        cand_text = "\n".join(
            f"{i + 1}. {c.get('artist', '')} \u2014 {c.get('title', '')}"
            for i, c in enumerate(candidates)
        )
        user = (
            f"User's request: {query}\n\n"
            f"The collection has {len(candidates)} song(s), in this order:\n"
            f"{cand_text}\n\n"
            "Write ONE short sentence acknowledging the request and stating "
            "the count, then output the numbered list ABOVE exactly as "
            "given (same order, same songs). Do NOT add any other text."
        )
        saved = self.temperature
        try:
            # Low temperature: this is a deterministic formatting task.
            self.temperature = 0.2
            return self._chat(_LIST_SYSTEM, user, history)
        finally:
            self.temperature = saved