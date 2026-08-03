"""Thin language-model client for SoundMatch.

The system talks to a local Ollama model. To keep the project reproducible for
anyone who does not have Ollama installed, this module degrades gracefully:

  - If SOUNDMATCH_BACKEND=stub, or the `ollama` package is missing, or the
    Ollama server is unreachable, we fall back to a deterministic STUB model.

The stub is not a third AI feature. It exists so the pipeline, the CLI demo,
and the test suite all still run end to end without a network or a GPU. Its
answers are intentionally simple keyword-driven responses.

Public surface:
    get_backend() -> str        # "ollama" or "stub", after probing
    complete(prompt, ...) -> str
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger("soundmatch.llm")

DEFAULT_MODEL = os.environ.get("SOUNDMATCH_MODEL", "llama3.2")
_FORCED_BACKEND = os.environ.get("SOUNDMATCH_BACKEND", "").strip().lower()

# Cache the probe result so we do not hit the network on every call.
_resolved_backend: str | None = None


def _probe_ollama(model: str) -> bool:
    """Return True if we can actually reach an Ollama server with a model."""
    try:
        import ollama  # type: ignore
    except ImportError:
        logger.info("ollama package not installed; using stub backend")
        return False
    try:
        # A cheap call that fails fast if the server is down.
        ollama.list()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means fall back
        logger.warning("Ollama unreachable (%s); using stub backend", exc)
        return False


def get_backend(model: str = DEFAULT_MODEL) -> str:
    """Resolve which backend is live: 'ollama' or 'stub'. Cached after first call."""
    global _resolved_backend
    if _resolved_backend is not None:
        return _resolved_backend

    if _FORCED_BACKEND == "stub":
        _resolved_backend = "stub"
    elif _FORCED_BACKEND == "ollama":
        # User insists on Ollama; still probe so we log a clear warning if down.
        _resolved_backend = "ollama" if _probe_ollama(model) else "stub"
    else:
        _resolved_backend = "ollama" if _probe_ollama(model) else "stub"

    logger.info("LLM backend resolved to: %s (model=%s)", _resolved_backend, model)
    return _resolved_backend


def complete(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.2) -> str:
    """Return a completion string for `prompt` from whichever backend is live."""
    backend = get_backend(model)
    if backend == "ollama":
        try:
            import ollama  # type: ignore

            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
            return resp["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama call failed mid-run (%s); using stub reply", exc)
            return _stub_complete(prompt)
    return _stub_complete(prompt)


# ---------------------------------------------------------------------------
# Deterministic stub backend
# ---------------------------------------------------------------------------

_GENRE_HINTS = {
    "lofi": ["lofi", "lo-fi", "study", "chill", "beats", "cafe"],
    "rock": ["rock", "guitar", "heavy", "loud", "intense", "angry", "mosh"],
    "pop": ["pop", "radio", "catchy", "upbeat", "happy", "sing"],
    "synthpop": ["synth", "80s", "retro", "neon", "electro", "dance"],
}
_MOOD_HINTS = {
    "calm": ["calm", "relax", "peaceful", "mellow"],
    "sleepy": ["sleep", "sleepy", "drowsy", "bedtime"],
    "happy": ["happy", "joy", "sunny", "cheer"],
    "focused": ["focus", "study", "work", "concentrate"],
    "intense": ["intense", "hype", "workout", "gym", "pump"],
    "melancholy": ["sad", "melancholy", "rainy", "blue", "cry"],
}


def _stub_complete(prompt: str) -> str:
    """A tiny rule-based stand-in for a real model.

    The pipeline only ever asks this module to do two jobs, and each job embeds
    a marker in the prompt so the stub knows which JSON shape to emit:
      - "TASK: PARSE"    -> return a UserProfile-shaped JSON object
      - "TASK: GENERATE" -> return a short grounded blurb as JSON
    """
    text = prompt.lower()

    if "task: parse" in text:
        # Only look at the listener's actual request, not the prompt template
        # (otherwise template words like "calm"/"sad"/"intense" leak in).
        m = re.search(r'listener request:\s*"([^"]*)"', text)
        query = m.group(1) if m else text
        return _stub_parse(query)
    if "task: generate" in text:
        return _stub_generate(prompt)
    return "OK"


def _stub_parse(text: str) -> str:
    genre = next((g for g, kws in _GENRE_HINTS.items() if any(k in text for k in kws)), None)
    mood = next((m for m, kws in _MOOD_HINTS.items() if any(k in text for k in kws)), None)

    # Energy heuristics from obvious words.
    energy = None
    if any(k in text for k in ["sleep", "calm", "chill", "study", "relax", "slow"]):
        energy = 0.25
    if any(k in text for k in ["hype", "workout", "gym", "intense", "party", "loud", "fast"]):
        energy = 0.9

    valence = None
    if any(k in text for k in ["happy", "joy", "sunny", "upbeat"]):
        valence = 0.85
    if any(k in text for k in ["sad", "rainy", "blue", "melancholy"]):
        valence = 0.2

    profile = {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "valence": valence,
        "danceability": None,
        "acousticness": None,
        "tempo": None,
        "notes": "stub-parsed from keywords",
    }
    return json.dumps(profile)


def _stub_generate(prompt: str) -> str:
    """Pick the first candidate title out of the prompt and praise it briefly."""
    # The generate prompt lists candidates as `- "Title" by Artist ...`.
    titles = re.findall(r'-\s*"([^"]+)"', prompt)
    top = titles[0] if titles else "the top match"
    blurb = (
        f'"{top}" lines up best with what you asked for, based on its energy '
        f"and mood in the catalog."
    )
    return json.dumps({"pick": top, "blurb": blurb})
