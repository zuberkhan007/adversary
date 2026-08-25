"""Configuration loader.

Reads ``config.yaml`` and overlays environment variables
(``OLLAMA_BASE_URL``, ``OLLAMA_MODEL``, ``EMBEDDING_MODEL``). The model
identifier is intentionally configurable and never hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


@dataclass
class RankingWeights:
    semantic: float = 0.35
    keyword: float = 0.10
    context: float = 0.0  # reserved; 0 in Phase 1
    feedback: float = 0.20
    time: float = 0.10
    repetition_penalty: float = 0.10


@dataclass
class FeedbackWeights:
    plays: float = 0.4
    likes: float = 0.3
    completion: float = 0.2
    skips: float = 0.1


@dataclass
class PersonalizationConfig:
    feedback_weights: FeedbackWeights = field(default_factory=FeedbackWeights)
    default_time_bias: float = 0.0
    diversity_weight: float = 0.20  # reserved; used as λ alt or for novelty


@dataclass
class LyricsConfig:
    provider: str = "lrclib"  # "lrclib" | "musixmatch" | "none"
    lrclib_base_url: str = "https://lrclib.net/api"
    lrclib_user_agent: str = "OHH/1.0"
    musixmatch_user_token: str = ""
    musixmatch_base_url: str = "https://api.musixmatch.com/ws/1.1"
    timeout: float = 10.0


@dataclass
class EnrichmentConfig:
    """Phase 3 RAG-upgrade config: how songs are enriched before indexing.

    ``mode`` selects what enrichment is applied:
    - ``"none"``: no enrichment; ``index_text == text`` (Phase 1 behavior).
    - ``"lyrics"``: fetch plain lyrics (capped) per song and append to index.
    - ``"tags"``: derive mood/theme tags via the LLM and append to index.
    - ``"both"``: lyrics + tags.
    - ``"auto"``: lyrics if a lyrics provider is configured, tags via the
      LLM otherwise. Default.

    ``lyrics_index_words`` caps the indexed lyric excerpt (~100 words). The
    excerpt is for INDEXING ONLY — never returned to the LLM/UI (copyright:
    spec.md §21). The chat-time output cap is ``max_lyric_excerpt_words``
    (default 25) which applies to the separately-fetched excerpt returned
    in chat responses.

    ``llm_tags`` enables LLM-generated tags. When ``False``, songs get no
    LLM-derived tags.
    """

    mode: str = "auto"
    lyrics_index_words: int = 100
    llm_tags: bool = True


@dataclass
class ModesConfig:
    """Phase 4 interaction-mode config (spec §14–17).

    All additive with defaults — existing ``Config(...)`` call sites keep
    working. ``explain_temperature`` / ``generation_temperature`` are the
    LLM temperatures used for the explain / generation prompts (spec §8:
    low for explain, higher for generation). ``antakshari_rule`` selects
    the chain rule (MVP supports ``"last_letter"`` and ``"first_letter"``;
    other spec §16 optional rules — language/decade/genre/artist — are out
    of scope). ``generation_top_k`` is the candidate-pool size for
    generation mode (larger than ``rerank_k`` so the diversity dedupe has
    room to work). ``generation_diversity`` toggles the per-artist dedupe.
    """

    explain_temperature: float = 0.3
    generation_temperature: float = 0.7
    antakshari_rule: str = "last_letter"
    generation_top_k: int = 10
    generation_diversity: bool = True


@dataclass
class Config:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:120b-cloud"
    ollama_api_key: str | None = None
    temperature: float = 0.3
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "data/faiss.index"
    sqlite_path: str = "data/songs.db"
    top_k: int = 10
    rerank_k: int = 5
    max_lyric_excerpt_words: int = 25
    include_source: bool = True
    ranking_weights: RankingWeights = field(default_factory=RankingWeights)
    personalization: PersonalizationConfig = field(default_factory=PersonalizationConfig)
    lyrics: LyricsConfig = field(default_factory=LyricsConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    modes: ModesConfig = field(default_factory=ModesConfig)


def _deep_get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from YAML with env var overlays."""
    cfg = Config()
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        ollama = _deep_get(raw, "ollama") or {}
        embedding = _deep_get(raw, "embedding") or {}
        rag = _deep_get(raw, "rag") or {}
        ranking = _deep_get(raw, "ranking") or {}
        response = _deep_get(raw, "response") or {}
        personalization = _deep_get(raw, "personalization") or {}
        lyrics = _deep_get(raw, "lyrics") or {}
        enrichment = _deep_get(raw, "enrichment") or {}
        modes = _deep_get(raw, "modes") or {}

        if ollama.get("base_url"):
            cfg.ollama_base_url = str(ollama["base_url"])
        if ollama.get("model"):
            cfg.ollama_model = str(ollama["model"])
        if ollama.get("api_key"):
            cfg.ollama_api_key = str(ollama["api_key"])
        if ollama.get("temperature") is not None:
            cfg.temperature = float(ollama["temperature"])

        if embedding.get("model"):
            cfg.embedding_model = str(embedding["model"])

        if rag.get("faiss_index_path"):
            cfg.faiss_index_path = str(rag["faiss_index_path"])
        if rag.get("sqlite_path"):
            cfg.sqlite_path = str(rag["sqlite_path"])
        if rag.get("top_k") is not None:
            cfg.top_k = int(rag["top_k"])
        if rag.get("rerank_k") is not None:
            cfg.rerank_k = int(rag["rerank_k"])

        rw = cfg.ranking_weights
        if ranking.get("semantic") is not None:
            rw.semantic = float(ranking["semantic"])
        if ranking.get("keyword") is not None:
            rw.keyword = float(ranking["keyword"])
        if ranking.get("context") is not None:
            rw.context = float(ranking["context"])
        if ranking.get("feedback") is not None:
            rw.feedback = float(ranking["feedback"])
        if ranking.get("time") is not None:
            rw.time = float(ranking["time"])
        if ranking.get("repetition_penalty") is not None:
            rw.repetition_penalty = float(ranking["repetition_penalty"])

        if response.get("max_lyric_excerpt_words") is not None:
            cfg.max_lyric_excerpt_words = int(response["max_lyric_excerpt_words"])
        if response.get("include_source") is not None:
            cfg.include_source = bool(response["include_source"])

        # Personalization (Phase 2). All additive with defaults.
        pc = cfg.personalization
        if personalization.get("default_time_bias") is not None:
            pc.default_time_bias = float(personalization["default_time_bias"])
        if personalization.get("diversity_weight") is not None:
            pc.diversity_weight = float(personalization["diversity_weight"])
        fbw = personalization.get("feedback_weights") or {}
        if fbw.get("plays") is not None:
            pc.feedback_weights.plays = float(fbw["plays"])
        if fbw.get("likes") is not None:
            pc.feedback_weights.likes = float(fbw["likes"])
        if fbw.get("completion") is not None:
            pc.feedback_weights.completion = float(fbw["completion"])
        if fbw.get("skips") is not None:
            pc.feedback_weights.skips = float(fbw["skips"])

        # Lyrics (Phase 3). All additive with defaults.
        lc = cfg.lyrics
        if lyrics.get("provider"):
            lc.provider = str(lyrics["provider"])
        if lyrics.get("lrclib_base_url"):
            lc.lrclib_base_url = str(lyrics["lrclib_base_url"])
        if lyrics.get("lrclib_user_agent"):
            lc.lrclib_user_agent = str(lyrics["lrclib_user_agent"])
        if lyrics.get("musixmatch_user_token"):
            lc.musixmatch_user_token = str(lyrics["musixmatch_user_token"])
        if lyrics.get("musixmatch_base_url"):
            lc.musixmatch_base_url = str(lyrics["musixmatch_base_url"])
        if lyrics.get("timeout") is not None:
            lc.timeout = float(lyrics["timeout"])

        # Enrichment (Phase 3 RAG upgrade). All additive with defaults.
        ec = cfg.enrichment
        if enrichment.get("mode"):
            ec.mode = str(enrichment["mode"])
        if enrichment.get("lyrics_index_words") is not None:
            ec.lyrics_index_words = int(enrichment["lyrics_index_words"])
        if enrichment.get("llm_tags") is not None:
            ec.llm_tags = bool(enrichment["llm_tags"])

        # Modes (Phase 4). All additive with defaults.
        mc = cfg.modes
        if modes.get("explain_temperature") is not None:
            mc.explain_temperature = float(modes["explain_temperature"])
        if modes.get("generation_temperature") is not None:
            mc.generation_temperature = float(modes["generation_temperature"])
        if modes.get("antakshari_rule"):
            mc.antakshari_rule = str(modes["antakshari_rule"])
        if modes.get("generation_top_k") is not None:
            mc.generation_top_k = int(modes["generation_top_k"])
        if modes.get("generation_diversity") is not None:
            mc.generation_diversity = bool(modes["generation_diversity"])

    # Env var overlays (take precedence over YAML).
    if os.getenv("OLLAMA_BASE_URL"):
        cfg.ollama_base_url = os.environ["OLLAMA_BASE_URL"]
    if os.getenv("OLLAMA_MODEL"):
        cfg.ollama_model = os.environ["OLLAMA_MODEL"]
    if os.getenv("OLLAMA_API_KEY"):
        cfg.ollama_api_key = os.environ["OLLAMA_API_KEY"]
    if os.getenv("EMBEDDING_MODEL"):
        cfg.embedding_model = os.environ["EMBEDDING_MODEL"]

    # Phase 3 env overlays.
    if os.getenv("LYRICS_PROVIDER"):
        cfg.lyrics.provider = os.environ["LYRICS_PROVIDER"]
    if os.getenv("MUSIXMATCH_USER_TOKEN"):
        cfg.lyrics.musixmatch_user_token = os.environ["MUSIXMATCH_USER_TOKEN"]
    if os.getenv("ENRICHMENT_MODE"):
        cfg.enrichment.mode = os.environ["ENRICHMENT_MODE"]

    return cfg