"""End-to-end orchestration: the Recommender class.

In the original SoundMatch this class was a stub. Now it does the real work of
the RAG system, wiring the four stages together and logging each one:

    parse (LLM)  ->  retrieve (scorer)  ->  generate (LLM)  ->  verify (guardrail)

A single call to `recommend()` returns a RecommendationResult that carries
everything needed to display the answer and to test it: the parsed profile, the
retrieved candidates, the final grounded pick, and which backend produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .generate import generate_recommendation
from .llm import get_backend
from .parse import parse_query
from .recommender import ScoredSong, Song, UserProfile, load_songs, recommend_songs

logger = logging.getLogger("soundmatch.pipeline")

DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


@dataclass
class RecommendationResult:
    query: str
    profile: UserProfile
    candidates: list[ScoredSong]
    pick: str
    blurb: str
    grounded: bool
    source: str  # "llm" or "template-fallback"
    backend: str  # "ollama" or "stub"
    extra: dict = field(default_factory=dict)


class Recommender:
    """Owns the catalog and runs the full retrieve-then-generate pipeline."""

    def __init__(self, catalog_path: str | Path = DEFAULT_CATALOG, top_k: int = 5):
        self.songs: list[Song] = load_songs(catalog_path)
        self.top_k = top_k
        logger.info("Recommender loaded %d songs from %s", len(self.songs), catalog_path)

    def recommend(self, query: str) -> RecommendationResult:
        logger.info("=== recommend: %r ===", query)

        # 1. PARSE (LLM, with validation + fallback inside)
        profile = parse_query(query)

        # 2. RETRIEVE (deterministic scorer)
        candidates = recommend_songs(self.songs, profile, top_k=self.top_k)
        logger.info(
            "retrieved: %s",
            [f"{c.song.title}({c.score:.2f})" for c in candidates],
        )

        # 3 + 4. GENERATE then VERIFY (guardrail lives in generate)
        result = generate_recommendation(profile, candidates)

        rec = RecommendationResult(
            query=query,
            profile=profile,
            candidates=candidates,
            pick=result["pick"],
            blurb=result["blurb"],
            grounded=result["grounded"],
            source=result["source"],
            backend=get_backend(),
        )
        logger.info(
            "result: pick=%r grounded=%s source=%s backend=%s",
            rec.pick, rec.grounded, rec.source, rec.backend,
        )
        return rec
