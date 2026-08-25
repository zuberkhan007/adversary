"""Vector store: FAISS index + SQLite metadata.

SQLite schema::

    songs(track_id PRIMARY KEY, title, artist, source, source_url, text,
          position, release_date, preview_url, index_lyrics, tags, audio_features)
    feedback(track_id PRIMARY KEY, plays, skips, likes, completion_ratio,
             last_played)

``text`` is the compact display representation (``"{title} — {artist}"``)
kept for citations/lookups. The actual indexed text (what FAISS embeds
and what BM25 scores against) is :attr:`Song.index_text` — a combination
of the compact line + enrichment tags + a capped lyric excerpt. The
enrichment columns (``index_lyrics``, ``tags``, ``audio_features``) are
additive ALTER TABLE migrations: legacy Phase 2 DBs upgrade in place.

``index_lyrics`` is a capped plain-lyrics excerpt used ONLY for indexing
(copyright: never returned to the LLM/UI). ``tags`` is a JSON list of
mood/theme descriptors. ``audio_features`` is a JSON dict of optional
audio-feature values (valence/energy/danceability/tempo/...).

``position`` records FAISS insertion order so the in-memory song list can
be rebuilt to match the index on load. ``release_date`` is an optional
``YYYY`` or ``YYYY-MM-DD`` string. ``preview_url`` is a preview clip URL
(empty for YouTube-sourced songs, which have no 30s preview). The ``feedback`` table is owned at runtime by
``app.personalization.feedback`` but its schema is created here so the
DB file is self-describing.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from app.ingestion.markdown import Song


SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    track_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    text TEXT NOT NULL,
    position INTEGER NOT NULL,
    release_date TEXT NOT NULL DEFAULT ''
);
"""

FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    track_id TEXT PRIMARY KEY,
    plays INTEGER NOT NULL DEFAULT 0,
    skips INTEGER NOT NULL DEFAULT 0,
    likes INTEGER NOT NULL DEFAULT 0,
    completion_ratio REAL NOT NULL DEFAULT 0.0,
    last_played TEXT NOT NULL DEFAULT ''
);
"""


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if ``column`` exists on ``table`` (parameterized probe)."""
    cur = conn.execute(f"PRAGMA table_info({table})")  # table is a hard-coded name here
    for row in cur.fetchall():
        # row: (cid, name, type, notnull, dflt_value, pk)
        if len(row) >= 2 and row[1] == column:
            return True
    return False


class VectorStore:
    def __init__(self, faiss_index_path: str = "data/faiss.index", sqlite_path: str = "data/songs.db") -> None:
        self.faiss_index_path = str(faiss_index_path)
        self.sqlite_path = str(sqlite_path)
        self._index = None
        self._songs: dict[str, Song] = {}
        self._order: list[str] = []  # track_ids in FAISS insertion order

    @property
    def index(self):
        if self._index is None:
            raise RuntimeError("VectorStore not loaded. Call build() or load() first.")
        return self._index

    @property
    def size(self) -> int:
        return len(self._songs)

    @staticmethod
    def _encode_with_progress(embedder, texts: list[str], progress) -> np.ndarray:
        """Encode texts in small batches, advancing a Streamlit-style progress bar.

        ``progress`` is optional. When ``None`` the embedder is called once
        with the full list (preserving the cached fast path).
        """
        if progress is None:
            return embedder.encode(texts)

        n = len(texts)
        batch = 8 if n > 8 else n
        out: list[np.ndarray] = []
        for i in range(0, n, batch):
            chunk = texts[i : i + batch]
            out.append(embedder.encode(chunk))
            done = min(n, i + len(chunk))
            try:
                progress(done / n)
            except Exception:
                pass
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype="float32")

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create tables if missing and add ``release_date``/``preview_url``/
        enrichment columns to legacy DBs.

        Idempotent: safe to call on a fresh or already-migrated DB.
        """
        conn.execute(SCHEMA)
        conn.execute(FEEDBACK_SCHEMA)
        if not _column_exists(conn, "songs", "release_date"):
            conn.execute(
                "ALTER TABLE songs ADD COLUMN release_date TEXT NOT NULL DEFAULT ''"
            )
        if not _column_exists(conn, "songs", "preview_url"):
            conn.execute(
                "ALTER TABLE songs ADD COLUMN preview_url TEXT NOT NULL DEFAULT ''"
            )
        # Phase 3 RAG-upgrade enrichment columns (additive, idempotent).
        if not _column_exists(conn, "songs", "index_lyrics"):
            conn.execute(
                "ALTER TABLE songs ADD COLUMN index_lyrics TEXT NOT NULL DEFAULT ''"
            )
        if not _column_exists(conn, "songs", "tags"):
            conn.execute(
                "ALTER TABLE songs ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'"
            )
        if not _column_exists(conn, "songs", "audio_features"):
            conn.execute(
                "ALTER TABLE songs ADD COLUMN audio_features TEXT NOT NULL DEFAULT '{}'"
            )
        conn.commit()

    # ---- build -------------------------------------------------------
    def build(self, songs: list[Song], embedder, progress=None) -> None:
        import faiss

        # Embed the content-rich ``index_text`` (title/artist + tags + capped
        # lyric excerpt), not the compact display ``text``. This is the whole
        # point of the Phase 3 RAG upgrade: a sparse "Title — Artist" line is
        # nearly content-free for semantic search; the enriched index lets a
        # query like "sad synthwave" match on lyrics/tags.
        texts = [s.index_text for s in songs]
        if not texts:
            raise ValueError("Cannot build index from empty song list.")
        vectors = self._encode_with_progress(embedder, texts, progress)
        dim = int(vectors.shape[1])

        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(vectors, dtype="float32"))

        Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, self.faiss_index_path)

        conn = sqlite3.connect(self.sqlite_path)
        try:
            self._ensure_schema(conn)
            conn.execute("DELETE FROM songs")
            conn.executemany(
                "INSERT OR REPLACE INTO songs(track_id,title,artist,source,source_url,text,position,release_date,preview_url,index_lyrics,tags,audio_features) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        s.track_id,
                        s.title,
                        s.artist,
                        s.source,
                        s.source_url,
                        s.text,
                        i,
                        s.release_date or "",
                        s.preview_url or "",
                        s.index_lyrics or "",
                        json.dumps(list(s.tags)) if s.tags else "[]",
                        json.dumps(s.audio_features) if s.audio_features else "{}",
                    )
                    for i, s in enumerate(songs)
                ],
            )
            conn.commit()
        finally:
            conn.close()

        self._index = index
        self._songs = {s.track_id: s for s in songs}
        self._order = [s.track_id for s in songs]

    # ---- load --------------------------------------------------------
    def load(self) -> None:
        import faiss

        if not Path(self.faiss_index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_index_path}")

        index = faiss.read_index(self.faiss_index_path)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            self._ensure_schema(conn)
            rows = conn.execute(
                "SELECT track_id,title,artist,source,source_url,text,position,release_date,preview_url,index_lyrics,tags,audio_features "
                "FROM songs ORDER BY position ASC"
            ).fetchall()
        finally:
            conn.close()

        songs: dict[str, Song] = {}
        order: list[str] = []
        for row in rows:
            (track_id, title, artist, source, source_url, _text, _position,
             release_date, preview_url, index_lyrics, tags_json, af_json) = row
            try:
                tags = json.loads(tags_json) if tags_json else []
            except (json.JSONDecodeError, TypeError):
                tags = []
            try:
                audio_features = json.loads(af_json) if af_json else {}
            except (json.JSONDecodeError, TypeError):
                audio_features = {}
            songs[track_id] = Song(
                title=title,
                artist=artist,
                source=source,
                source_url=source_url,
                track_id=track_id,
                release_date=release_date or "",
                preview_url=preview_url or "",
                index_lyrics=index_lyrics or "",
                tags=tags if isinstance(tags, list) else [],
                audio_features=audio_features if isinstance(audio_features, dict) else {},
            )
            order.append(track_id)

        self._index = index
        self._songs = songs
        self._order = order

    # ---- clear -------------------------------------------------------
    def clear(self) -> None:
        """Remove the on-disk FAISS index + SQLite DB and reset in-memory state.

        After this, the store is empty: :attr:`size` is 0, :attr:`index`
        raises (call :meth:`build` or :meth:`load` to repopulate). The
        on-disk files are deleted so the next build starts from scratch.
        The SQLite DB is then re-created with an empty schema (so the
        shared ``feedback`` table still exists for :class:`FeedbackStore`).
        """
        import os

        self._index = None
        self._songs = {}
        self._order = []
        for path in (self.faiss_index_path, self.sqlite_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        # Re-create an empty SQLite DB with the schema so the shared
        # ``feedback`` table (owned at runtime by FeedbackStore) still
        # exists after the songs table is wiped.
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            self._ensure_schema(conn)
            # songs table is empty after _ensure_schema (CREATE IF NOT EXISTS
            # only creates; it doesn't carry rows). feedback table is empty
            # too — orphaned feedback for deleted songs is purged.
            conn.execute("DELETE FROM feedback")
            conn.commit()
        finally:
            conn.close()

    # ---- query -------------------------------------------------------
    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[Song, float]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = np.ascontiguousarray(query_vec, dtype="float32")
        scores, ids = self.index.search(query_vec, top_k)
        out: list[tuple[Song, float]] = []
        for sid, score in zip(ids[0].tolist(), scores[0].tolist()):
            if sid < 0 or sid >= len(self._order):
                continue
            track_id = self._order[sid]
            song = self._songs.get(track_id)
            if song is None:
                continue
            out.append((song, float(score)))
        return out

    def all_songs(self) -> list[Song]:
        return [self._songs[tid] for tid in self._order if tid in self._songs]

    def get(self, track_id: str) -> Song | None:
        return self._songs.get(track_id)