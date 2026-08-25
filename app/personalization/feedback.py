"""Feedback persistence (spec.md §10, §11, §31).

Wraps the ``feedback`` table (created by :mod:`app.rag.vector_store`) in a
SQLite connection. All queries use parameterized SQL — no string
interpolation (spec §31).

The ``completion_ratio`` is stored as a running average across plays, so a
single skip-after-1s play doesn't permanently mark a song as "low
completion" if the user later listens to it in full.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.config import FeedbackWeights


@dataclass
class FeedbackRow:
    track_id: str
    plays: int = 0
    skips: int = 0
    likes: int = 0
    completion_ratio: float = 0.0
    last_played: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    track_id TEXT PRIMARY KEY,
    plays INTEGER NOT NULL DEFAULT 0,
    skips INTEGER NOT NULL DEFAULT 0,
    likes INTEGER NOT NULL DEFAULT 0,
    completion_ratio REAL NOT NULL DEFAULT 0.0,
    last_played TEXT NOT NULL DEFAULT ''
);
"""


class FeedbackStore:
    """SQLite-backed feedback store. Safe to construct before the songs DB
    exists — the ``feedback`` table is created on demand."""

    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = str(sqlite_path)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.sqlite_path)

    def _ensure_schema(self) -> None:
        conn = self._conn()
        try:
            conn.execute(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _row_to_feedback(self, row: tuple) -> FeedbackRow:
        return FeedbackRow(
            track_id=row[0],
            plays=int(row[1]),
            skips=int(row[2]),
            likes=int(row[3]),
            completion_ratio=float(row[4]),
            last_played=row[5] or "",
        )

    def get(self, track_id: str) -> Optional[FeedbackRow]:
        conn = self._conn()
        try:
            cur = conn.execute(
                "SELECT track_id, plays, skips, likes, completion_ratio, last_played "
                "FROM feedback WHERE track_id = ?",
                (track_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_feedback(row)
        finally:
            conn.close()

    def all_rows(self) -> dict[str, FeedbackRow]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT track_id, plays, skips, likes, completion_ratio, last_played "
                "FROM feedback"
            ).fetchall()
            return {r[0]: self._row_to_feedback(r) for r in rows}
        finally:
            conn.close()

    def record_play(self, track_id: str, completion_ratio: float = 1.0) -> FeedbackRow:
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("track_id must be a non-empty string")
        completion_ratio = max(0.0, min(1.0, float(completion_ratio)))
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT plays, completion_ratio FROM feedback WHERE track_id = ?",
                (track_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO feedback(track_id, plays, skips, likes, "
                    "completion_ratio, last_played) VALUES (?, 1, 0, 0, ?, ?)",
                    (track_id, completion_ratio, now),
                )
            else:
                old_plays = int(existing[0])
                old_completion = float(existing[1])
                n = old_plays + 1
                new_completion = (old_completion * (n - 1) / n) + (completion_ratio / n)
                conn.execute(
                    "UPDATE feedback SET plays = ?, completion_ratio = ?, last_played = ? "
                    "WHERE track_id = ?",
                    (n, new_completion, now, track_id),
                )
            conn.commit()
        finally:
            conn.close()
        row = self.get(track_id)
        assert row is not None  # we just inserted/updated
        return row

    def record_skip(self, track_id: str) -> FeedbackRow:
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("track_id must be a non-empty string")
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT skips FROM feedback WHERE track_id = ?", (track_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO feedback(track_id, plays, skips, likes, "
                    "completion_ratio, last_played) VALUES (?, 0, 1, 0, 0.0, '')",
                    (track_id,),
                )
            else:
                conn.execute(
                    "UPDATE feedback SET skips = ? WHERE track_id = ?",
                    (int(existing[0]) + 1, track_id),
                )
            conn.commit()
        finally:
            conn.close()
        row = self.get(track_id)
        assert row is not None
        return row

    def record_like(self, track_id: str) -> FeedbackRow:
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("track_id must be a non-empty string")
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT likes FROM feedback WHERE track_id = ?", (track_id,)
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO feedback(track_id, plays, skips, likes, "
                    "completion_ratio, last_played) VALUES (?, 0, 0, 1, 0.0, '')",
                    (track_id,),
                )
            else:
                conn.execute(
                    "UPDATE feedback SET likes = ? WHERE track_id = ?",
                    (int(existing[0]) + 1, track_id),
                )
            conn.commit()
        finally:
            conn.close()
        row = self.get(track_id)
        assert row is not None
        return row

    def preference_score(self, track_id: str, weights: FeedbackWeights) -> float:
        """Return a raw feedback score in roughly ``[0, 1]`` for this track.

        Uses ``plays_max`` = the max plays across the store (or 1 if empty) so
        a single play on a fresh store isn't automatically maxed out.
        """
        row = self.get(track_id)
        if row is None:
            return 0.0
        all_rows = self.all_rows()
        plays_max = max((r.plays for r in all_rows.values()), default=0)
        return _feedback_score(row, weights, plays_max)


def _feedback_score(row: FeedbackRow, weights: FeedbackWeights, plays_max: int) -> float:
    norm_plays = row.plays / max(1, plays_max)
    like_score = row.likes / max(1, row.plays)
    skip_score = row.skips / max(1, row.plays)
    return (
        weights.plays * norm_plays
        + weights.likes * like_score
        + weights.completion * row.completion_ratio
        - weights.skips * skip_score
    )