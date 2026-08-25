"""Pure scoring helpers for personalization (spec.md §9, §11).

These functions take already-fetched data and return floats. They have no
side effects and are trivially testable in isolation.
"""

from __future__ import annotations

import re

from app.config import FeedbackWeights
from app.personalization.feedback import FeedbackRow, _feedback_score


_YEAR_PARSE = re.compile(r"^\s*(\d{4})(?:-\d{2}-\d{2})?\s*$")


def _year_of(release_date: str) -> int | None:
    if not release_date:
        return None
    m = _YEAR_PARSE.match(release_date)
    if not m:
        return None
    return int(m.group(1))


def time_score(
    release_date: str,
    min_year: int,
    max_year: int,
    time_bias: float,
) -> float:
    """Score a song by release year and the user's time preference.

    ``time_bias`` ∈ ``[-1, +1]``: ``+1`` boosts newer songs, ``-1`` boosts
    older songs, ``0`` is neutral. Unknown release dates return ``0.0``.
    With a degenerate year range (``min == max``) every known-date song is
    treated as the midpoint → also ``0.0``.
    """
    year = _year_of(release_date)
    if year is None:
        return 0.0
    if max_year == min_year:
        return 0.0
    norm = (year - min_year) / (max_year - min_year)
    norm = max(0.0, min(1.0, norm))
    bias = max(-1.0, min(1.0, float(time_bias)))
    return bias * (2.0 * norm - 1.0)


def feedback_score(
    row: FeedbackRow,
    weights: FeedbackWeights,
    plays_max: int,
) -> float:
    """Public wrapper around the feedback-score formula (spec §11)."""
    return _feedback_score(row, weights, plays_max)