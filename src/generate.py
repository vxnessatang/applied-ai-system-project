"""Grounded generation + the anti-hallucination guardrail.

This is the second half of RAG. The retriever (recommender.py) has already
picked a small set of candidate songs. We hand ONLY those rows to the LLM and
ask it to write the final recommendation. The model is instructed to choose one
of the given titles and explain why.

The guardrail (`verify_grounded`) then checks that the song the model named
actually exists in the retrieved candidate set. If the model hallucinated a
title that is not in the catalog subset, we reject its answer and fall back to a
deterministic template built from the retriever's own reasons. This is what
makes the feature RAG rather than "LLM output printed next to a score": the
model literally cannot recommend a song outside the retrieved evidence.
"""

from __future__ import annotations

import json
import logging
import re

from .llm import complete
from .parse import _extract_json
from .recommender import ScoredSong, UserProfile

logger = logging.getLogger("soundmatch.generate")

_GEN_PROMPT = """TASK: GENERATE

You are a music concierge. Recommend ONE song to the listener, chosen ONLY from
the candidate list below. Do not invent songs. Respond with ONLY a JSON object:
{{"pick": "<exact title from the list>", "blurb": "<2 sentence reason>"}}

Listener wanted: {intent}

Candidates (title, artist, and why the ranking engine surfaced it):
{candidates}

JSON:"""


def _format_candidates(candidates: list[ScoredSong]) -> str:
    lines = []
    for cs in candidates:
        why = "; ".join(cs.reasons) if cs.reasons else "overall vibe fit"
        lines.append(
            f'- "{cs.song.title}" by {cs.song.artist} '
            f"(genre={cs.song.genre}, mood={cs.song.mood}, "
            f"energy={cs.song.energy:.2f}, score={cs.score:.2f}; {why})"
        )
    return "\n".join(lines)


def _template_explanation(candidates: list[ScoredSong]) -> dict:
    """Deterministic fallback used when the LLM answer fails the guardrail."""
    top = candidates[0]
    why = "; ".join(top.reasons) if top.reasons else "it was the closest overall fit"
    return {
        "pick": top.song.title,
        "blurb": f'"{top.song.title}" by {top.song.artist} is the top match because {why}.',
        "grounded": True,
        "source": "template-fallback",
    }


def verify_grounded(pick: str, candidates: list[ScoredSong]) -> bool:
    """True if `pick` matches a candidate title (case/space insensitive)."""
    if not isinstance(pick, str) or not pick.strip():
        return False
    norm = _norm(pick)
    return any(_norm(cs.song.title) == norm for cs in candidates)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def generate_recommendation(
    profile: UserProfile, candidates: list[ScoredSong]
) -> dict:
    """Produce the final grounded recommendation dict.

    Returns keys: pick, blurb, grounded (bool), source (llm|template-fallback).
    """
    if not candidates:
        raise ValueError("no candidates to generate from")

    intent = profile.notes or profile.raw_query or "a good match for their taste"
    prompt = _GEN_PROMPT.format(
        intent=intent, candidates=_format_candidates(candidates)
    )
    raw = complete(prompt)

    try:
        data = _extract_json(raw)
        pick = data.get("pick", "")
        blurb = data.get("blurb", "")
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("generate: unreadable JSON (%s); using template", exc)
        return _template_explanation(candidates)

    # THE GUARDRAIL: reject any pick not present in the retrieved evidence.
    if not verify_grounded(pick, candidates):
        logger.warning(
            "generate: model picked ungrounded title %r; using template", pick
        )
        return _template_explanation(candidates)

    logger.info("generate: grounded pick=%r", pick)
    return {
        "pick": pick,
        "blurb": blurb or f'"{pick}" is the recommended track.',
        "grounded": True,
        "source": "llm",
    }
