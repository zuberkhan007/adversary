"""Hybrid reranker: semantic similarity + lightweight keyword (BM25-lite) score.

Phase 1 combines only the ``semantic`` and ``keyword`` terms. Feedback,
time and repetition terms are wired in Phase 2 via
``app.personalization.ranking``.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.config import RankingWeights
from app.rag.retriever import Candidate


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class _BM25Lite:
    """Tiny in-memory BM25 over candidate texts."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        self.df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                self.df[term] += 1
        self.N = len(docs)

    def score(self, query: str, idx: int) -> float:
        if idx >= len(self.doc_tokens) or self.N == 0:
            return 0.0
        toks = self.doc_tokens[idx]
        tf = Counter(toks)
        q_terms = tokenize(query)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0)
            denom = tf[term] + self.k1 * (1 - self.b + self.b * (self.doc_lens[idx] / (self.avg_len or 1.0)))
            score += idf * (tf[term] * (self.k1 + 1)) / denom
        return score


def rerank(
    query: str,
    candidates: list[Candidate],
    weights: RankingWeights,
) -> list[Candidate]:
    """Rerank candidates by ``Score = a*S + b*K``.

    Returns the candidates sorted by final score (descending). The semantic
    scores are clamped to [0, 1] (Sentence Transformers normalizes embeddings
    so cosine similarity is in [-1, 1]; we clip negatives to 0). Keyword
    scores are min-max normalized across the candidate set so they are
    comparable to semantic scores.
    """
    if not candidates:
        return []

    # BM25 over the content-rich ``index_text`` (title/artist + tags + capped
    # lyric excerpt) so keyword matches hit lyrics and tags, not just the
    # 4-word "Title — Artist" line. Phase 3 RAG upgrade.
    bm25 = _BM25Lite([c.song.index_text for c in candidates])
    raw_kw = [bm25.score(query, i) for i in range(len(candidates))]
    kw_max = max(raw_kw) if raw_kw else 0.0
    kw_min = min(raw_kw) if raw_kw else 0.0
    denom = (kw_max - kw_min) or 1.0

    ranked: list[Candidate] = []
    for i, c in enumerate(candidates):
        sem = max(0.0, c.semantic_score)
        kw = (raw_kw[i] - kw_min) / denom if denom else 0.0
        final = weights.semantic * sem + weights.keyword * kw
        ranked.append(
            Candidate(
                song=c.song,
                semantic_score=c.semantic_score,
                keyword_score=kw,
                final_score=final,
            )
        )

    ranked.sort(key=lambda c: c.final_score, reverse=True)
    return ranked