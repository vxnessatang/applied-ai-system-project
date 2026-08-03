"""Reliability and guardrail test suite for SoundMatch RAG.

This is the project's Reliability/Evaluation component. It runs against the
deterministic stub backend so it passes with zero external setup, while still
exercising the exact same pipeline code the Ollama backend uses.

Coverage:
  1. Grounding: the model can never recommend a song outside the retrieved set.
  2. Robustness: malformed model output falls back cleanly instead of crashing.
  3. Consistency: the same query yields the same retrieved set every run.
  4. Guardrail unit test: verify_grounded accepts real titles, rejects fakes.
  5. End-to-end: all demo queries return a valid, grounded recommendation.
"""

import os

import pytest

# Force the deterministic stub backend so tests never depend on a live model.
os.environ["SOUNDMATCH_BACKEND"] = "stub"

from src.generate import generate_recommendation, verify_grounded
from src.main import DEMO_QUERIES
from src.parse import KNOWN_GENRES, parse_query
from src.pipeline import Recommender
from src.recommender import UserProfile, load_songs, recommend_songs


@pytest.fixture(scope="module")
def rec():
    return Recommender()


@pytest.fixture(scope="module")
def songs():
    from src.pipeline import DEFAULT_CATALOG
    return load_songs(DEFAULT_CATALOG)


# 1. Grounding guardrail across every demo query -----------------------------
@pytest.mark.parametrize("query", DEMO_QUERIES)
def test_recommendation_is_grounded(rec, query):
    result = rec.recommend(query)
    titles = {c.song.title for c in result.candidates}
    assert result.pick in titles, f"picked {result.pick!r} not in retrieved set"
    assert result.grounded is True


# 2. Robustness: a broken model reply must not crash the generator -----------
def test_generation_survives_garbage_model_output(monkeypatch, songs):
    import src.generate as gen

    monkeypatch.setattr(gen, "complete", lambda *_a, **_k: "this is not json at all {{{")
    profile = UserProfile(genre="lofi", energy=0.3, raw_query="test")
    candidates = recommend_songs(songs, profile, top_k=5)

    result = gen.generate_recommendation(profile, candidates)
    assert result["grounded"] is True
    assert result["source"] == "template-fallback"
    assert result["pick"] == candidates[0].song.title


def test_generation_rejects_hallucinated_pick(monkeypatch, songs):
    import src.generate as gen

    monkeypatch.setattr(
        gen, "complete",
        lambda *_a, **_k: '{"pick": "A Song That Does Not Exist", "blurb": "trust me"}',
    )
    profile = UserProfile(genre="rock", energy=0.9, raw_query="test")
    candidates = recommend_songs(songs, profile, top_k=5)

    result = gen.generate_recommendation(profile, candidates)
    # Guardrail must catch the fake title and fall back to a real candidate.
    assert result["source"] == "template-fallback"
    assert result["pick"] in {c.song.title for c in candidates}


# 3. Consistency: deterministic retrieval --------------------------------------
def test_retrieval_is_consistent(rec):
    q = "chill lofi beats to study to, but nothing that puts me to sleep"
    a = rec.recommend(q)
    b = rec.recommend(q)
    assert [c.song.id for c in a.candidates] == [c.song.id for c in b.candidates]
    assert a.pick == b.pick


# 4. Guardrail unit test -------------------------------------------------------
def test_verify_grounded_unit(songs):
    profile = UserProfile(genre="pop", raw_query="pop")
    candidates = recommend_songs(songs, profile, top_k=5)
    real_title = candidates[0].song.title
    assert verify_grounded(real_title, candidates) is True
    assert verify_grounded("  " + real_title.upper() + " ", candidates) is True  # normalized
    assert verify_grounded("Totally Made Up Track", candidates) is False
    assert verify_grounded("", candidates) is False


# 5. Parser guardrails ---------------------------------------------------------
def test_parser_drops_out_of_vocab_genre(monkeypatch):
    import src.parse as parse

    monkeypatch.setattr(
        parse, "complete",
        lambda *_a, **_k: '{"genre": "polka", "mood": null, "energy": 0.5}',
    )
    profile = parse.parse_query("some polka thing with energy")
    assert profile.genre is None  # polka is not in KNOWN_GENRES
    assert profile.energy == 0.5


def test_parser_clamps_out_of_range_numbers(monkeypatch):
    import src.parse as parse

    monkeypatch.setattr(
        parse, "complete",
        lambda *_a, **_k: '{"genre": "rock", "energy": 4.5, "valence": -2}',
    )
    profile = parse.parse_query("super intense rock")
    assert profile.energy == 1.0
    assert profile.valence == 0.0


def test_parser_empty_query_raises():
    with pytest.raises(ValueError):
        parse_query("   ")


def test_known_genres_match_catalog(songs):
    catalog_genres = {s.genre for s in songs}
    assert catalog_genres == KNOWN_GENRES
