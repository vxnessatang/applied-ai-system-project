"""Query understanding: free text -> structured UserProfile.

This is the first half of the RAG pipeline. The listener types something like
"chill beats to study to, but nothing that puts me to sleep". We ask the LLM to
translate that into the numeric/categorical fields the retriever understands.

Guardrails live here, not in the model:
  - We demand strict JSON and extract the first JSON object even if the model
    wraps it in prose or code fences.
  - Every field is validated and clamped. Genres/moods outside the catalog
    vocabulary are dropped (set to None) rather than trusted blindly.
  - If parsing fails for any reason, we fall back to a keyword parser so the
    system never crashes on a bad model response.
"""

from __future__ import annotations

import json
import logging
import re

from .llm import complete
from .recommender import UserProfile

logger = logging.getLogger("soundmatch.parse")

# The catalog vocabulary the retriever can actually match against.
KNOWN_GENRES = {"lofi", "rock", "pop", "synthpop"}

_NUMERIC_FIELDS = ("energy", "valence", "danceability", "acousticness", "tempo")

_PARSE_PROMPT = """TASK: PARSE

You convert a music listener's request into a JSON profile. Respond with ONLY a
JSON object, no prose, no code fences.

Allowed genres: lofi, rock, pop, synthpop (use null if none clearly fits).
Fields (use null when the request gives no signal):
  genre: one of the allowed genres or null
  mood: a single lowercase word or null
  energy: 0.0 (very calm) to 1.0 (very intense) or null
  valence: 0.0 (sad) to 1.0 (happy) or null
  danceability: 0.0 to 1.0 or null
  acousticness: 0.0 to 1.0 or null
  tempo: 0.0 (slow) to 1.0 (fast) or null
  notes: a short phrase capturing the listener's intent

Listener request: "{query}"
JSON:"""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response. Raises ValueError."""
    # Strip code fences if present.
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Find the first balanced-looking {...} block.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


def _clamp_unit(value) -> float | None:
    """Coerce to a float in [0,1], or None if it is not a usable number."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))


def _keyword_fallback(query: str) -> UserProfile:
    """Deterministic backup parser used when the LLM output is unusable."""
    q = query.lower()
    genre = next((g for g in KNOWN_GENRES if g in q), None)
    energy = None
    if any(k in q for k in ["sleep", "calm", "chill", "study", "relax", "slow"]):
        energy = 0.25
    if any(k in q for k in ["hype", "workout", "gym", "intense", "party", "fast", "loud"]):
        energy = 0.9
    valence = None
    if any(k in q for k in ["happy", "joy", "upbeat", "sunny"]):
        valence = 0.85
    if any(k in q for k in ["sad", "rainy", "blue", "melancholy"]):
        valence = 0.2
    logger.info("parse: used keyword fallback for query=%r", query)
    return UserProfile(
        genre=genre,
        energy=energy,
        valence=valence,
        raw_query=query,
        notes="keyword-fallback",
    )


def parse_query(query: str) -> UserProfile:
    """Turn a free-text request into a validated UserProfile."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    prompt = _PARSE_PROMPT.format(query=query.replace('"', "'"))
    raw = complete(prompt)

    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("parse: could not read JSON (%s); falling back", exc)
        return _keyword_fallback(query)

    # Validate genre against the catalog vocabulary.
    genre = data.get("genre")
    if isinstance(genre, str):
        genre = genre.strip().lower()
        if genre not in KNOWN_GENRES:
            logger.info("parse: dropping out-of-vocabulary genre %r", genre)
            genre = None
    else:
        genre = None

    mood = data.get("mood")
    mood = mood.strip().lower() if isinstance(mood, str) and mood.strip() else None

    numeric = {name: _clamp_unit(data.get(name)) for name in _NUMERIC_FIELDS}

    notes = data.get("notes")
    notes = notes.strip() if isinstance(notes, str) else ""

    profile = UserProfile(
        genre=genre,
        mood=mood,
        energy=numeric["energy"],
        valence=numeric["valence"],
        danceability=numeric["danceability"],
        acousticness=numeric["acousticness"],
        tempo=numeric["tempo"],
        raw_query=query,
        notes=notes,
    )

    # Guardrail: if the model gave us nothing usable, fall back so retrieval is
    # not run on an empty profile (which would score every song 0.0).
    if _is_empty(profile):
        logger.warning("parse: model produced an empty profile; falling back")
        return _keyword_fallback(query)

    logger.info("parse: %r -> %s", query, profile)
    return profile


def _is_empty(p: UserProfile) -> bool:
    return all(
        getattr(p, f) is None
        for f in ("genre", "mood", "energy", "valence", "danceability", "acousticness", "tempo")
    )
