"""Flask UI for the AI Music Conversational Assistant.

Replaces the Streamlit UI with a Flask single-page app served at ``/``.
The chat / ingest / pull / model-switch endpoints are JSON + Server-Sent
Events (SSE). A single global :class:`MusicAssistant` instance backs the
app — the offline-first, single-browser-session model the Streamlit app
used. A ``threading.Lock`` serializes chat / ingest / pull so the FAISS
index isn't touched concurrently.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Force UTF-8 on stdout/stderr so yt-dlp / Sentence Transformers / tqdm
# progress bars (which write Unicode block characters) don't crash the
# worker thread with ``OSError: [Errno 22] Invalid argument`` on Windows
# where the default console/pipe codec is cp1252 and can't encode them.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from flask import Flask, Response, jsonify, render_template, request

from app.config import load_config
from app.llm.ollama import (
    is_server_up,
    list_models,
    parse_pull_progress,
    pull_model,
)
from app.main import MusicAssistant


# --------------------------------------------------------------------------
# Helpers (ported from the Streamlit UI)
# --------------------------------------------------------------------------

def _provider_label(source: str) -> str:
    """Human-readable provider label for a cited song."""
    s = (source or "").strip().lower()
    if s == "youtube":
        return "YouTube"
    if s == "markdown":
        return "Markdown"
    return s.title() or "Markdown"


def _phase_label(f: float) -> str:
    """Progress-bar label for the three-phase ingest pipeline.

    The orchestrator's ``_enrich_and_build`` maps the full ``[0, 1]`` range
    across fetch (0–0.1), enrichment (0.1–0.5), embedding (0.5–1.0).
    """
    f = max(0.0, min(1.0, float(f)))
    if f < 0.1:
        return "Fetching playlist \u2026"
    if f < 0.5:
        return "Enriching songs (lyrics + tags) \u2026"
    return "Generating embeddings \u2026"


def _resolve_installed(installed: list[str], requested: str) -> str | None:
    """Match a requested name to an exact installed name.

    Handles Ollama storing ``tinyllama`` as ``tinyllama:latest``.
    Returns the exact installed name or ``None`` when not installed.
    """
    r = (requested or "").strip()
    if not r:
        return None
    if r in installed:
        return r
    for m in installed:
        if m.split(":", 1)[0] == r:
            return m
    return None


def _is_online_endpoint(base_url: str, api_key: str | None) -> bool:
    """Treat an endpoint as online (no pull needed) when the host is not
    localhost or an API key is set."""
    local_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        host = ""
    remote_host = host not in local_hosts and host != ""
    return remote_host or bool(api_key)


def _fmt_duration(seconds: int | float | None) -> str:
    """Format a duration in seconds as ``m:ss`` (empty for 0/None)."""
    s = int(seconds or 0)
    if s <= 0:
        return ""
    m, sec = divmod(s, 60)
    return f"{m}:{sec:02d}"


# --------------------------------------------------------------------------
# Global state (single-user, offline-first)
# --------------------------------------------------------------------------

cfg = load_config(_ROOT / "config.yaml")
assistant = MusicAssistant(cfg)
try:
    assistant.load_existing()
except Exception:
    pass

lock = threading.Lock()
app = Flask(
    __name__,
    template_folder=str(_ROOT / "ui" / "templates"),
    static_folder=str(_ROOT / "ui" / "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


# --------------------------------------------------------------------------
# State serialization
# --------------------------------------------------------------------------

def _serialize_turn(turn) -> dict:
    sources = []
    for s in (turn.sources or []):
        if isinstance(s, dict):
            sources.append({
                "display": s.get("display", ""),
                "track_id": s.get("track_id", ""),
            })
        else:
            sources.append({"display": str(s), "track_id": ""})
    return {
        "user": turn.user,
        "assistant": turn.assistant,
        "sources": sources,
        "rewritten_query": turn.rewritten_query,
        "mode": turn.mode,
        "lyric_data": dict(turn.lyric_data or {}),
    }


def _serialize_song(song) -> dict:
    return {
        "title": song.title,
        "artist": song.artist,
        "provider": _provider_label(song.source),
        "duration": _fmt_duration(song.duration),
        "track_id": song.track_id,
        "preview_url": song.preview_url or "",
        "tags": list(song.tags or []),
        "source_url": song.source_url or "",
    }


def _antakshari_state() -> dict | None:
    sess = assistant.antakshari
    if sess is None or not sess.active:
        return None
    return sess.to_dict()


def _chat_mode_state() -> str:
    """Active chat-mode label, mirrored from the client via /api/state.

    The server has no opinion on chat mode — it's a UI toggle that maps to
    an ``override`` passed into ``chat_dispatch``. We return ``""`` here;
    the client owns the value.
    """
    return ""


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index() -> Any:
    return render_template("index.html")


@app.route("/api/state")
def api_state() -> Any:
    cfg = assistant.config
    online = _is_online_endpoint(cfg.ollama_base_url, cfg.ollama_api_key)
    up = is_server_up(cfg.ollama_base_url, api_key=cfg.ollama_api_key)
    return jsonify({
        "chat_history": [_serialize_turn(t) for t in assistant.history],
        "songs_count": assistant.store.size,
        "ready": assistant.ready,
        "model": cfg.ollama_model,
        "base_url": cfg.ollama_base_url,
        "api_key_set": bool(cfg.ollama_api_key),
        "server_up": bool(up),
        "online": bool(online),
        "antakshari": _antakshari_state(),
        "time_bias": float(cfg.personalization.default_time_bias),
    })


@app.route("/api/songs")
def api_songs() -> Any:
    songs = assistant.store.all_songs()
    return jsonify([_serialize_song(s) for s in songs])


@app.route("/api/chat", methods=["POST"])
def api_chat() -> Any:
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    override = data.get("override")
    if not message:
        return jsonify({"error": "Empty message."}), 400
    if not assistant.ready:
        return jsonify({
            "response": "Build the index first (left panel).",
            "sources": [],
            "rewritten_query": "",
            "mode": "general",
            "antakshari": _antakshari_state(),
        })

    with lock:
        # Active Antakshari game takes precedence regardless of override.
        if assistant.antakshari is not None and assistant.antakshari.active:
            res = assistant.submit_antakshari(message)
            turn = {
                "user": message,
                "assistant": res.get("message", ""),
                "sources": [],
                "rewritten_query": "",
                "mode": "antakshari",
                "lyric_data": {},
            }
            return jsonify({
                "response": res.get("message", ""),
                "sources": [],
                "rewritten_query": "",
                "mode": "antakshari",
                "antakshari": _antakshari_state(),
                "antakshari_result": res,
                "turn": turn,
            })

        try:
            outcome = assistant.chat_dispatch(message, override=override)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        latest = assistant.history[-1] if assistant.history else None
        sources = []
        if latest and latest.sources:
            for s in latest.sources:
                if isinstance(s, dict):
                    sources.append({
                        "display": s.get("display", ""),
                        "track_id": s.get("track_id", ""),
                    })
                else:
                    sources.append({"display": str(s), "track_id": ""})

        turn = {
            "user": message,
            "assistant": outcome.response,
            "sources": sources,
            "rewritten_query": (latest.rewritten_query if latest else ""),
            "mode": (latest.mode if latest else "general"),
            "lyric_data": dict(latest.lyric_data) if latest else {},
        }
        return jsonify({
            "response": outcome.response,
            "sources": sources,
            "rewritten_query": turn["rewritten_query"],
            "mode": turn["mode"],
            "antakshari": _antakshari_state(),
            "turn": turn,
        })


@app.route("/api/ingest/youtube", methods=["POST"])
def api_ingest_youtube() -> Any:
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    enrich = bool(data.get("enrich", True))
    if not url:
        return jsonify({"error": "Enter a YouTube playlist URL or ID first."}), 400
    if assistant.youtube_client is None:
        return jsonify({
            "error": "YouTube ingestion is unavailable. Install yt-dlp with "
                     "`pip install -r requirements.txt` and restart the app.",
        }), 400

    q: "queue.Queue[tuple]" = queue.Queue()

    def worker() -> None:
        try:
            with lock:
                def cb(f: float) -> None:
                    q.put(("progress", float(f), _phase_label(f)))
                n = assistant.ingest_youtube_playlist(url, progress=cb, enrich=enrich)
            q.put(("done", int(n), None))
        except Exception as exc:
            import traceback
            tb_text = traceback.format_exc()
            # Persist the full traceback so it can be inspected even if the
            # browser alert only shows a short message.
            try:
                from pathlib import Path as _P
                _log = _P(_ROOT, "data", "ingest_error.log")
                _log.parent.mkdir(parents=True, exist_ok=True)
                _log.write_text(
                    f"URL: {url}\nenrich: {enrich}\n{tb_text}", encoding="utf-8"
                )
            except Exception:
                pass
            tb = tb_text.strip().splitlines()
            last = tb[-1] if tb else ""
            msg = f"{type(exc).__name__}: {exc}"
            if last and last not in msg:
                msg = f"{msg}\n{last}"
            q.put(("error", msg, None))

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            try:
                kind, payload, label = q.get(timeout=600)
            except queue.Empty:
                yield f"event: error\ndata: {json.dumps({'error': 'Timed out.'})}\n\n"
                break
            if kind == "progress":
                yield (
                    "event: progress\ndata: "
                    + json.dumps({"frac": payload, "label": label})
                    + "\n\n"
                )
            elif kind == "done":
                yield "event: done\ndata: " + json.dumps({"n": payload}) + "\n\n"
                break
            elif kind == "error":
                yield (
                    "event: error\ndata: "
                    + json.dumps({"error": payload})
                    + "\n\n"
                )
                break

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/pull", methods=["POST"])
def api_pull() -> Any:
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"error": "Enter a model name."}), 400

    q: "queue.Queue[tuple]" = queue.Queue()

    def worker() -> None:
        try:
            base = assistant.config.ollama_base_url
            key = assistant.config.ollama_api_key
            last_label = {"v": "starting"}
            last_frac = {"v": 0.0}
            for evt in pull_model(base, model, api_key=key):
                label, frac = parse_pull_progress(evt)
                if label:
                    last_label["v"] = label
                if frac is not None:
                    last_frac["v"] = frac
                q.put(("progress", last_frac["v"], last_label["v"]))
            q.put(("done", 1.0, None))
        except Exception as exc:
            q.put(("error", str(exc), None))

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            try:
                kind, payload, label = q.get(timeout=600)
            except queue.Empty:
                yield f"event: error\ndata: {json.dumps({'error': 'Timed out.'})}\n\n"
                break
            if kind == "progress":
                yield (
                    "event: progress\ndata: "
                    + json.dumps({"frac": payload, "label": label})
                    + "\n\n"
                )
            elif kind == "done":
                yield "event: done\ndata: " + json.dumps({"ok": True}) + "\n\n"
                break
            elif kind == "error":
                yield (
                    "event: error\ndata: "
                    + json.dumps({"error": payload})
                    + "\n\n"
                )
                break

    return Response(stream(), mimetype="text/event-stream")


@app.route("/api/clear", methods=["POST"])
def api_clear() -> Any:
    with lock:
        assistant.clear_index()
        # Reset the in-memory Ollama API key so the UI's API-key field clears
        # too (``/api/state`` reports ``api_key_set = bool(cfg.ollama_api_key)``).
        assistant.set_api_key(None)
        try:
            removed = assistant.embedder.clear_cache()
        except Exception:
            removed = 0
    return jsonify({"ok": True, "removed_embeddings": int(removed or 0)})


@app.route("/api/clear/chat", methods=["POST"])
def api_clear_chat() -> Any:
    """Clear chat history only (and any active Antakshari session)."""
    with lock:
        assistant.clear_chat_history()
    return jsonify({"ok": True})


@app.route("/api/clear/collection", methods=["POST"])
def api_clear_collection() -> Any:
    """Clear the indexed song collection (FAISS + SQLite + embedding cache).
    Chat history is left intact. Antakshari is reset (depends on the store).
    """
    with lock:
        assistant.clear_collection()
        try:
            removed = assistant.embedder.clear_cache()
        except Exception:
            removed = 0
    return jsonify({"ok": True, "removed_embeddings": int(removed or 0)})


@app.route("/api/clear/api_key", methods=["POST"])
def api_clear_api_key() -> Any:
    """Clear the in-memory Ollama API key."""
    with lock:
        assistant.set_api_key(None)
    return jsonify({"ok": True})


@app.route("/api/time_bias", methods=["POST"])
def api_time_bias() -> Any:
    data = request.get_json(silent=True) or {}
    try:
        bias = float(data.get("bias", 0.0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid bias."}), 400
    clamped = assistant.set_time_bias(bias)
    return jsonify({"bias": clamped})


@app.route("/api/model", methods=["POST"])
def api_model() -> Any:
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or "").strip()
    if not model:
        return jsonify({"error": "Enter a model name."}), 400

    cfg = assistant.config
    online = _is_online_endpoint(cfg.ollama_base_url, cfg.ollama_api_key)

    if not is_server_up(cfg.ollama_base_url, api_key=cfg.ollama_api_key):
        if online:
            with lock:
                assistant.set_model(model)
            return jsonify({"ok": True, "model": model})
        return jsonify({"error": "Ollama server is offline. Start it with `ollama serve`."}), 400

    installed = list_models(cfg.ollama_base_url, api_key=cfg.ollama_api_key)
    resolved = _resolve_installed(installed, model)

    if resolved is not None:
        with lock:
            assistant.set_model(resolved)
        return jsonify({"ok": True, "model": resolved})

    if online:
        with lock:
            assistant.set_model(model)
        return jsonify({"ok": True, "model": model})

    # Local Ollama + not installed → caller should open the pull SSE.
    return jsonify({"error": "not_installed", "model": model})


@app.route("/api/api_key", methods=["POST"])
def api_api_key() -> Any:
    data = request.get_json(silent=True) or {}
    key = (data.get("api_key") or "").strip() or None
    with lock:
        assistant.set_api_key(key)
    return jsonify({"ok": True})


@app.route("/api/models")
def api_models() -> Any:
    cfg = assistant.config
    online = _is_online_endpoint(cfg.ollama_base_url, cfg.ollama_api_key)
    up = is_server_up(cfg.ollama_base_url, api_key=cfg.ollama_api_key)
    installed = list_models(cfg.ollama_base_url, api_key=cfg.ollama_api_key) if up else []
    return jsonify({
        "installed": installed,
        "configured": cfg.ollama_model,
        "server_up": bool(up),
        "online": bool(online),
    })


@app.route("/api/lyrics/<path:track_id>")
def api_lyrics(track_id: str) -> Any:
    with lock:
        try:
            res = assistant.get_lyrics_excerpt(track_id)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(res)


@app.route("/api/antakshari/stop", methods=["POST"])
def api_antakshari_stop() -> Any:
    with lock:
        res = assistant.stop_antakshari()
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=False, threaded=True)