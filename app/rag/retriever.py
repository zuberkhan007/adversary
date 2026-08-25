"""Semantic retriever: top-k candidates from the FAISS index."""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.markdown import Song
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorStore


@dataclass
class Candidate:
    song: Song
    semantic_score: float
    keyword_score: float = 0.0
    final_score: float = 0.0


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 10) -> list[Candidate]:
        qvec = self.embedder.encode(query)
        hits = self.store.search(qvec, top_k)
        return [Candidate(song=song, semantic_score=score) for song, score in hits]