"""Personalization package.

Phase 2 will plug feedback + time-bias terms into the final score here.
For Phase 1 this module is a thin passthrough so the pipeline can call it
without depending on feedback data that does not exist yet.
"""

from app.personalization.ranking import finalize_scores

__all__ = ["finalize_scores"]