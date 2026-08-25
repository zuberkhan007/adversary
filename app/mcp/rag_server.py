"""rag-mcp — retrieval tools (spec.md §12, §31).

Plain tool functions delegate to a :class:`Retriever` and the
:func:`app.rag.reranker.rerank` function set via :func:`set_retriever`. Each
returns serializable dicts (``{track_id, title, artist, score}``).
"""

from __future__ import annotations

from typing import Optional

from app.config import RankingWeights
from app.mcp import _make_mcp
from app.rag.reranker import rerank
from app.rag.retriever import Retriever

mcp = _make_mcp("rag-mcp")

_retriever: Optional[Retriever] = None
_weights: RankingWeights = RankingWeights()


def set_retriever(retriever: Retriever, weights: RankingWeights | None = None) -> None:
    """Inject the Retriever + ranking weights used by the tool functions."""
    global _retriever, _weights
    _retriever = retriever
    if weights is not None:
        _weights = weights


def _require_retriever() -> Retriever:
    if _retriever is None:
        raise RuntimeError("rag-mcp: retriever not initialized. Call set_retriever() first.")
    return _retriever


def _validate_query(query) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return query


def _validate_top_k(top_k) -> int:
    try:
        n = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be a positive integer") from exc
    if n <= 0:
        raise ValueError("top_k must be a positive integer")
    return n


def _candidates_to_dicts(candidates) -> list[dict]:
    return [
        {
            "track_id": c.song.track_id,
            "title": c.song.title,
            "artist": c.song.artist,
            "score": float(c.final_score),
        }
        for c in candidates
    ]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Pure semantic (FAISS) top-k search. Returns serializable candidate dicts."""
    q = _validate_query(query)
    n = _validate_top_k(top_k)
    candidates = _require_retriever().retrieve(q, top_k=n)
    return [
        {
            "track_id": c.song.track_id,
            "title": c.song.title,
            "artist": c.song.artist,
            "score": float(c.semantic_score),
        }
        for c in candidates
    ]


def keyword_search(query: str, top_k: int = 10) -> list[dict]:
    """Semantic retrieval then BM25-lite keyword rerank, returned as dicts.

    The underlying keyword scorer is :class:`app.rag.reranker._BM25Lite`; it
    is applied to the top-k retrieved candidates so it has something to score.
    """
    q = _validate_query(query)
    n = _validate_top_k(top_k)
    candidates = _require_retriever().retrieve(q, top_k=n)
    # Use keyword-only weights so final_score ≈ normalized keyword score.
    kw_weights = RankingWeights(semantic=0.0, keyword=1.0, context=0.0,
                                feedback=0.0, time=0.0, repetition_penalty=0.0)
    ranked = rerank(q, candidates, kw_weights)
    return _candidates_to_dicts(ranked)


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """Semantic + keyword hybrid search using the configured weights."""
    q = _validate_query(query)
    n = _validate_top_k(top_k)
    candidates = _require_retriever().retrieve(q, top_k=n)
    ranked = rerank(q, candidates, _weights)
    return _candidates_to_dicts(ranked)


def rerank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Rerank a list of candidate dicts (each with ``track_id``) against a query.

    Looks up each ``track_id`` in the store, builds :class:`Candidate`
    objects, and applies the hybrid reranker. Unknown track_ids are skipped.
    """
    q = _validate_query(query)
    retriever = _require_retriever()
    cand_objs = []
    for entry in candidates or []:
        tid = entry.get("track_id") if isinstance(entry, dict) else None
        if not tid:
            continue
        song = retriever.store.get(tid)
        if song is None:
            continue
        from app.rag.retriever import Candidate
        cand_objs.append(Candidate(song=song, semantic_score=float(entry.get("score", 0.0))))
    ranked = rerank(q, cand_objs, _weights)
    return _candidates_to_dicts(ranked)


if mcp is not None:
    for _fn in (semantic_search, keyword_search, hybrid_search, rerank_candidates):
        try:
            mcp.tool()(_fn)
        except Exception:
            pass