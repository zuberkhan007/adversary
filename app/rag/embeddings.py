"""Sentence Transformers embedding wrapper.

Lazy-loads the model on first use and caches per-document embeddings to disk
keyed by ``track_id`` so the FAISS index can be rebuilt cheaply.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str | Path = "data/embeddings") -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._model = None

    @property
    def dim(self) -> int:
        # Probe the model to get the embedding dimension. Lazy-load first.
        if self._model is None:
            self._load_model()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def _load_model(self) -> None:
        if self._model is not None:
            return
        # Local import keeps the module importable without the heavy dep
        # for tests that stub the embedder.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def encode(self, texts: list[str] | str, use_cache: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        self._load_model()

        results: list[np.ndarray | None] = [None] * len(text_list)
        to_compute: list[int] = []
        for i, t in enumerate(text_list):
            if not use_cache:
                to_compute.append(i)
                continue
            cp = self._cache_path(t)
            if cp.exists():
                try:
                    results[i] = np.asarray(json.loads(cp.read_text("utf-8"))["vector"], dtype="float32")
                except Exception:
                    to_compute.append(i)
            else:
                to_compute.append(i)

        if to_compute:
            texts_to_compute = [text_list[i] for i in to_compute]
            vecs = self._model.encode(  # type: ignore[union-attr]
                texts_to_compute, normalize_embeddings=True, convert_to_numpy=True
            )
            vecs = np.asarray(vecs, dtype="float32")
            for j, idx in enumerate(to_compute):
                v = vecs[j]
                results[idx] = v
                cp = self._cache_path(text_list[idx])
                try:
                    cp.write_text(json.dumps({"vector": v.tolist()}), "utf-8")
                except OSError:
                    pass

        arr = np.stack([r for r in results if r is not None]).astype("float32")
        if single:
            return arr[0]
        return arr

    def clear_cache(self) -> int:
        """Delete all on-disk cached embeddings. Returns the count removed.

        Safe to call when the cache directory doesn't exist (returns 0).
        Does not unload the in-memory model.
        """
        removed = 0
        if not self.cache_dir.exists():
            return removed
        for cp in self.cache_dir.glob("*.json"):
            try:
                cp.unlink()
                removed += 1
            except OSError:
                pass
        return removed