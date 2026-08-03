"""Core retrieval engine for SoundMatch.

This is the deterministic backbone carried over from the original SoundMatch
simulation. It scores every song against a UserProfile using a weighted
closeness rule and returns a ranked list. In the extended RAG system this
module plays the role of the RETRIEVER: given a structured profile, it
surfaces the candidate songs the language model is then allowed to talk about.

No machine learning here on purpose. The scores are reproducible: the same
profile always yields the same ranking, which is what makes the grounding
guardrail (see generate.py) and the consistency tests meaningful.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Tempo is stored in BPM but scored on a 0-1 scale. These bounds normalize it.
TEMPO_MIN_BPM = 60.0
TEMPO_MAX_BPM = 180.0

# Weights for the scoring recipe. Energy leads because listeners tend to pick by
# overall vibe first; genre and mood are strong categorical signals.
WEIGHTS = {
    "genre": 2.0,
    "mood": 2.0,
    "energy": 3.0,
    "valence": 2.0,
    "danceability": 1.0,
    "acousticness": 1.0,
    "tempo": 1.0,
}


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

    @property
    def tempo_norm(self) -> float:
        """BPM mapped into 0-1, clamped to the configured tempo range."""
        span = TEMPO_MAX_BPM - TEMPO_MIN_BPM
        value = (self.tempo_bpm - TEMPO_MIN_BPM) / span
        return max(0.0, min(1.0, value))


@dataclass
class UserProfile:
    """A structured description of what the listener wants right now.

    Any field may be None, meaning "no preference"; None fields are skipped in
    scoring so they neither help nor hurt a song. The LLM parser in parse.py
    produces these objects from free text.
    """

    genre: Optional[str] = None
    mood: Optional[str] = None
    energy: Optional[float] = None
    valence: Optional[float] = None
    danceability: Optional[float] = None
    acousticness: Optional[float] = None
    tempo: Optional[float] = None  # already normalized to 0-1
    raw_query: str = ""
    notes: str = ""  # free-text intent the LLM extracted, used later for generation


@dataclass
class ScoredSong:
    song: Song
    score: float
    reasons: list[str] = field(default_factory=list)


def load_songs(csv_path: str | Path) -> list[Song]:
    """Load the catalog from CSV. Raises FileNotFoundError if the path is wrong."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Song catalog not found at {path}")

    songs: list[Song] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            songs.append(
                Song(
                    id=int(row["id"]),
                    title=row["title"],
                    artist=row["artist"],
                    genre=row["genre"].strip().lower(),
                    mood=row["mood"].strip().lower(),
                    energy=float(row["energy"]),
                    tempo_bpm=float(row["tempo_bpm"]),
                    valence=float(row["valence"]),
                    danceability=float(row["danceability"]),
                    acousticness=float(row["acousticness"]),
                )
            )
    if not songs:
        raise ValueError(f"Song catalog at {path} is empty")
    return songs


def _closeness(target: float, value: float) -> float:
    """1.0 when identical, down to 0.0 when maximally apart on a 0-1 scale."""
    return 1.0 - abs(target - value)


def score_song(song: Song, profile: UserProfile) -> ScoredSong:
    """Score one song against a profile, returning the score and human reasons.

    The score is a weighted average over only the dimensions the profile
    actually specifies, so it always lands in 0-1 regardless of how many
    preferences the user gave.
    """
    total = 0.0
    weight_sum = 0.0
    reasons: list[str] = []

    if profile.genre is not None:
        w = WEIGHTS["genre"]
        weight_sum += w
        if song.genre == profile.genre.strip().lower():
            total += w * 1.0
            reasons.append(f"genre matches ({song.genre})")

    if profile.mood is not None:
        w = WEIGHTS["mood"]
        weight_sum += w
        if song.mood == profile.mood.strip().lower():
            total += w * 1.0
            reasons.append(f"mood matches ({song.mood})")

    numeric = [
        ("energy", profile.energy, song.energy),
        ("valence", profile.valence, song.valence),
        ("danceability", profile.danceability, song.danceability),
        ("acousticness", profile.acousticness, song.acousticness),
        ("tempo", profile.tempo, song.tempo_norm),
    ]
    for name, target, value in numeric:
        if target is None:
            continue
        w = WEIGHTS[name]
        weight_sum += w
        c = _closeness(target, value)
        total += w * c
        if c >= 0.85:
            reasons.append(f"{name} is a close fit ({value:.2f} vs wanted {target:.2f})")

    score = (total / weight_sum) if weight_sum > 0 else 0.0
    return ScoredSong(song=song, score=round(score, 4), reasons=reasons)


def recommend_songs(
    songs: list[Song], profile: UserProfile, top_k: int = 5
) -> list[ScoredSong]:
    """Rank the whole catalog and return the top_k candidates.

    Ties are broken by song id so the ordering is fully deterministic.
    """
    scored = [score_song(s, profile) for s in songs]
    scored.sort(key=lambda ss: (-ss.score, ss.song.id))
    return scored[:top_k]
