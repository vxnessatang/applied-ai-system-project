"""CLI entry point for the SoundMatch RAG recommender.

Usage:
    python -m src.main "chill beats to study to, but not sleepy"
    python -m src.main --demo          # runs the built-in example queries
    python -m src.main                 # interactive prompt

Set logging to DEBUG for a full trace:
    python -m src.main --demo -v
"""

from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import Recommender, RecommendationResult

# A small suite of example queries, mixing realistic requests with adversarial
# edge cases carried over from the original SoundMatch test profiles.
DEMO_QUERIES = [
    "high energy pop to sing along to in the car",
    "chill lofi beats to study to, but nothing that puts me to sleep",
    "deep intense rock for an angry workout",
    "something sad but also really high energy",          # contradictory wishes
    "polka accordion folk music",                          # not in catalog
    "the most extreme, over the top, maximum everything track you have",
]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _print_result(result: RecommendationResult) -> None:
    print(f'\nQuery: "{result.query}"')
    print(f"Parsed profile: {_profile_summary(result.profile)}")
    print(f"Backend: {result.backend} | grounded: {result.grounded} | source: {result.source}")
    print("Top candidates the model could choose from:")
    for cs in result.candidates:
        print(f"  - {cs.song.title} by {cs.song.artist}  (score {cs.score:.2f})")
    print(f"\n>>> Recommendation: {result.pick}")
    print(f"    {result.blurb}\n")
    print("-" * 68)


def _profile_summary(profile) -> str:
    fields = ["genre", "mood", "energy", "valence", "danceability", "acousticness", "tempo"]
    parts = [f"{f}={getattr(profile, f)}" for f in fields if getattr(profile, f) is not None]
    return ", ".join(parts) if parts else "(no preferences extracted)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SoundMatch RAG recommender")
    parser.add_argument("query", nargs="*", help="free-text music request")
    parser.add_argument("--demo", action="store_true", help="run built-in example queries")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    try:
        rec = Recommender()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1

    if args.demo:
        for q in DEMO_QUERIES:
            _print_result(rec.recommend(q))
        return 0

    if args.query:
        query = " ".join(args.query)
        _print_result(rec.recommend(query))
        return 0

    # Interactive mode.
    print("SoundMatch RAG. Type a request, or blank line to quit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        _print_result(rec.recommend(query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
