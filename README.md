# SoundMatch RAG (AI110 Project 4: Applied AI System)

## What this is

This project extends **SoundMatch**, my Module 3 content-based music recommender.
The original SoundMatch was a command-line tool that scored a catalog of 19
songs against a hand-built `UserProfile` (genre, mood, energy, valence, and so
on) using a weighted closeness rule, then printed the top 5 matches with a
plain-language reason for each. It used no machine learning, just a scoring
formula and a sort, and the user had to describe their taste as exact numbers
and category names.

**The problem I wanted to solve:** real people do not describe music in
0-to-1 feature vectors. They say things like "chill lofi to study to, but
nothing that puts me to sleep." The old system could not accept that. It also
matched genres by exact string, so "rap" would never match "hip-hop."

**The extension:** SoundMatch RAG puts a language model on both ends of the old
scorer so the system now accepts free text and explains its pick in natural
language, while the deterministic scorer still does the actual retrieval. The
scorer stays the trustworthy backbone; the model only interprets and describes.

## New AI feature: Retrieval-Augmented Generation (RAG)

The system runs a four-stage pipeline:

1. **Parse (LLM).** The free-text query is turned into a structured
   `UserProfile`. Output is validated as JSON, genres are checked against the
   catalog vocabulary, and every number is clamped to 0 to 1.
2. **Retrieve (scorer).** The original `score_song` / `recommend_songs` logic
   ranks the whole catalog and returns the top 5 candidates. This is the
   retrieval step, and it is fully deterministic.
3. **Generate (LLM).** The model is given **only** those 5 retrieved songs and
   asked to recommend one and explain why.
4. **Verify (guardrail).** The system checks that the song the model named is
   actually one of the retrieved candidates. If it is not, the model answer is
   thrown out and a template built from the scorer's own reasons is used
   instead.

Because generation is restricted to the retrieved rows, the model cannot
recommend a song that is not in the catalog. The retrieved data actively shapes
the answer rather than sitting next to it, which is the point of RAG.

## Reliability and guardrails

This is a separate, functional part of the system, not just tests on the side:

- **Input validation** in the parser: bad JSON, out-of-vocabulary genres, and
  out-of-range numbers are all caught and corrected or dropped.
- **Output guardrail** in the generator: the grounding check
  (`verify_grounded`) rejects any recommendation the model invents.
- **Graceful fallbacks** everywhere: a keyword parser backs up the LLM parse,
  and a template backs up the LLM generation, so the system never crashes on a
  bad model response.
- **Evaluation suite** in `tests/test_reliability.py`: 14 tests covering
  grounding, robustness to garbage model output, deterministic retrieval, and
  the guardrail itself.

See [reliability examples](#reliability-examples-input-behavior-result) below
for input-and-result walkthroughs.

## Architecture

The data flow is in
[assets/diagrams/architecture.mmd](assets/diagrams/architecture.mmd)
as Mermaid source. Summary:

```
free-text query
  -> Parse (LLM + validation, keyword fallback)
  -> Retrieve (weighted scorer over songs.csv, top 5)
  -> Generate (LLM sees only the 5 candidates)
  -> Verify (grounding guardrail, template fallback)
  -> grounded recommendation
```

## Setup

Requires Python 3.10 or newer. From this folder:

```bash
# 1. create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2. install (only needed for real-model mode and pytest)
pip install -r requirements.txt
```

The core system needs **no third-party packages**. Out of the box it uses a
built-in deterministic stub model, so it runs on a plain Python install.

## Running the system

Run a single query:

```bash
python -m src.main "high energy pop to sing along to in the car"
```

Run the built-in demo (six example queries, three realistic and three
adversarial):

```bash
python -m src.main --demo
```

Interactive mode (type queries until you enter a blank line):

```bash
python -m src.main
```

Add `-v` for a full debug trace of every pipeline stage.

### Running with a real model (Ollama)

By default the system tries Ollama and silently falls back to the stub if it is
not available. To use a real local model:

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.2
python -m src.main --demo            # now uses Ollama automatically
```

You can pin the model or force a backend with environment variables:

```bash
SOUNDMATCH_MODEL=llama3.2 python -m src.main "sad but high energy"
SOUNDMATCH_BACKEND=stub python -m src.main --demo   # force the offline stub
```

## Sample output

Running `python -m src.main --demo` (stub backend, so this is reproducible for
anyone with no setup):

```text
Query: "high energy pop to sing along to in the car"
Parsed profile: genre=pop
Backend: stub | grounded: True | source: llm
Top candidates the model could choose from:
  - Rooftop Sunrise by Ella Brooke  (score 1.00)
  - Glass Hearts by Ella Brooke  (score 1.00)
  - Golden Hour by Ella Brooke  (score 1.00)
  - Wildfire by Ellabrooke  (score 1.00)
  - Midnight Drive by Neon Vale  (score 0.00)

>>> Recommendation: Rooftop Sunrise
    "Rooftop Sunrise" lines up best with what you asked for, based on its energy and mood in the catalog.
--------------------------------------------------------------------
Query: "chill lofi beats to study to, but nothing that puts me to sleep"
Parsed profile: genre=lofi, mood=sleepy, energy=0.25
Backend: stub | grounded: True | source: llm
Top candidates the model could choose from:
  - Slow Tide by Kaito Mori  (score 0.97)
  - Rainy Commute by Nora Lin  (score 0.71)
  - Paper Lanterns by Kaito Mori  (score 0.70)
  - Soft Focus by Nora Lin  (score 0.69)
  - Cafe Window by Nora Lin  (score 0.69)

>>> Recommendation: Slow Tide
    "Slow Tide" lines up best with what you asked for, based on its energy and mood in the catalog.
--------------------------------------------------------------------
Query: "deep intense rock for an angry workout"
Parsed profile: genre=rock, mood=focused, energy=0.9
Backend: stub | grounded: True | source: llm
Top candidates the model could choose from:
  - Deep End by Greyline  (score 0.71)
  - Static Bloom by Greyline  (score 0.70)
  - Concrete Jungle by Greyline  (score 0.69)
  - Ashfall by Greyline  (score 0.65)
  - Study Loop by Kaito Mori  (score 0.48)

>>> Recommendation: Deep End
    "Deep End" lines up best with what you asked for, based on its energy and mood in the catalog.
--------------------------------------------------------------------
```

The three adversarial demo queries (contradictory taste, a genre not in the
catalog, and impossible extremes) also return a valid grounded pick. The
out-of-catalog case parses to an empty profile, all songs score 0, and the
system still returns a real catalog song rather than inventing one.

## Design decisions and trade-offs

- **Keep the old scorer as the retriever, do not replace it.** The deterministic
  scorer is the part I trust, so I made it the retrieval step and let the model
  only interpret input and describe output. Trade-off: the model cannot override
  a bad ranking, but that is the point, it also cannot invent one.
- **Ground generation to the retrieved set.** Handing the model only the top 5
  songs is what makes this real RAG. Trade-off: the model can never surprise you
  with a song outside the candidates, which is exactly the safety I wanted.
- **Ollama with a deterministic stub fallback.** A local model keeps it free and
  offline, and the stub means the project runs and all tests pass with zero
  setup. Trade-off: the stub parses more crudely than a real model, so the demo
  output is honest about being simple rather than polished.
- **Fail safe, never crash.** Every model call has a fallback (keyword parser,
  template generator). Trade-off: a fallback answer is less rich than a good
  model answer, but the system stays reliable no matter what the model returns.

## Reliability examples (input, behavior, result)

**1. Model hallucinates a song that does not exist.**

- Input: retrieval returns 5 rock songs; the model replies
  `{"pick": "A Song That Does Not Exist", "blurb": "trust me"}`.
- Behavior: `verify_grounded` finds no matching title in the retrieved set and
  rejects the answer.
- Result: the system falls back to the top-scored real song and reports
  `source=template-fallback`. Covered by
  `test_generation_rejects_hallucinated_pick`.

**2. Model returns unparseable text.**

- Input: the model replies `this is not json at all {{{`.
- Behavior: JSON extraction fails, the generator logs a warning.
- Result: template fallback fires, the system still returns a grounded pick and
  does not crash. Covered by `test_generation_survives_garbage_model_output`.

**3. Model invents a genre outside the catalog.**

- Input: the parser model replies `{"genre": "polka", "energy": 0.5}`.
- Behavior: `polka` is not in the catalog vocabulary, so it is dropped.
- Result: the profile keeps `energy=0.5` and sets `genre=None`. Covered by
  `test_parser_drops_out_of_vocab_genre`.

**4. Model returns numbers outside 0 to 1.**

- Input: `{"genre": "rock", "energy": 4.5, "valence": -2}`.
- Behavior: values are clamped.
- Result: `energy=1.0`, `valence=0.0`. Covered by
  `test_parser_clamps_out_of_range_numbers`.

## Testing summary

```bash
python -m pytest tests/ -q
```

Result: **14 out of 14 tests pass** in under a second, against the stub backend,
so they need no network, no API key, and no GPU.

| Test area | What it checks | Result |
| --- | --- | --- |
| Grounding (6 queries) | Every recommendation is a real retrieved song | Pass |
| Hallucination rejection | Invented title is caught and replaced | Pass |
| Garbage-output robustness | Unparseable model reply falls back, no crash | Pass |
| Deterministic retrieval | Same query gives the same ranking | Pass |
| Input validation | Out-of-vocab genre dropped, numbers clamped | Pass |

What worked: the deterministic scorer and the grounding guardrail were rock
solid, so no recommendation ever escaped the catalog. What did not work at first:
the stub parser matched words in my prompt template instead of the user query,
which made every result identical until I fixed it (see model_card.md). What I
learned: the guardrail matters most exactly when the model is weakest, and tests
are what proved that rather than a demo that happened to look fine.

## Project layout

```
data/songs.csv        19-song catalog (the retrieval source)
src/recommender.py    scoring engine = the RETRIEVER (from original SoundMatch)
src/llm.py            Ollama client with deterministic stub fallback
src/parse.py          free text -> UserProfile, with validation guardrails
src/generate.py       grounded generation + the anti-hallucination guardrail
src/pipeline.py       Recommender class: parse -> retrieve -> generate -> verify
src/main.py           command-line interface
tests/test_reliability.py   the evaluation and guardrail suite
assets/diagrams/architecture.mmd   Mermaid architecture diagram
model_card.md         intended use, data, algorithm, biases, evaluation,
                      and the graded Responsible AI Reflection
```
