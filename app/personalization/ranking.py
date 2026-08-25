"""Final-score seam: applies the remaining terms of the spec.md §28 formula.

The hybrid reranker (``app.rag.reranker``) already applied the semantic and
keyword terms (``α·S + β·K``) and stored the result in
``Candidate.final_score``. This module adds the personalization terms::

    Score = a*S + b*K + g*F + d*T - l*R

``finalize_scores`` mutates ``final_score`` in place and re-sorts the list by
the new score (descending). The ``α·S + β·K`` portion is preserved.
"""

from __future__ import annotations

from app.config import FeedbackWeights, RankingWeights
from app.personalization.feedback import FeedbackRow, _feedback_score
from app.personalization.recency import _year_of, time_score
from app.rag.retriever import Candidate


def finalize_scores(
    candidates: list[Candidate],
    *,
    feedback: dict[str, FeedbackRow] | None = None,
    time_bias: float = 0.0,
    seen_track_ids: set[str] | None = None,
    fb_weights: FeedbackWeights | None = None,
    ranking_weights: RankingWeights | None = None,
) -> list[Candidate]:
    """Apply ``+ γ·F + δ·T - λ·R`` to ``final_score`` in place, re-sort.

    Returns the same list object (mutated), sorted by ``final_score``
    descending. Empty input returns empty. With no feedback rows, zero time
    bias, no seen ids, or zero weights, the personalization terms contribute
    ``0`` and the input order (which is already sorted by the reranker) is
    preserved.
    """
    if not candidates:
        return candidates

    fb = feedback or {}
    seen = seen_track_ids or set()
    fb_w = fb_weights or FeedbackWeights()
    rw = ranking_weights or RankingWeights()

    # Compute min/max year across candidates that have a known release year.
    years = [_year_of(c.song.release_date) for c in candidates]
    known = [y for y in years if y is not None]
    if known:
        min_year, max_year = min(known), max(known)
    else:
        min_year, max_year = 0, 0

    # plays_max across the provided feedback rows so a single play isn't
    # automatically maxed out on a fresh store.
    plays_max = max((r.plays for r in fb.values()), default=0)

    for i, c in enumerate(candidates):
        F = 0.0
        row = fb.get(c.song.track_id)
        if row is not None:
            F = _feedback_score(row, fb_w, plays_max)

        T = time_score(c.song.release_date, min_year, max_year, time_bias)

        R = 1.0 if c.song.track_id in seen else 0.0

        c.final_score = c.final_score + (
            rw.feedback * F + rw.time * T - rw.repetition_penalty * R
        )

    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates