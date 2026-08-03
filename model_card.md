# Model Card: SoundMatch RAG

## Name
SoundMatch RAG, a retrieval-augmented music recommender that extends the
Module 3 SoundMatch simulation.

## Intended use
Take a listener's request in plain English and recommend one song from a fixed
19-song catalog, with a short reason grounded in that song's real features.
Built as a learning project for AI110 Project 4.

## Not intended for
Real production music delivery, discovering music outside the small demo
catalog, or any setting where a wrong or invented recommendation carries real
cost. The catalog is tiny and made up.

## Data
`data/songs.csv`, 19 fictional songs. Columns: id, title, artist, genre, mood,
energy, tempo_bpm, valence, danceability, acousticness. The four numeric feeling
columns are 0 to 1 floats. Genres are limited to lofi, rock, pop, and synthpop.

## How it works
Four stages: parse, retrieve, generate, verify.

- **Parse:** a language model (local Ollama, or a deterministic stub when Ollama
  is absent) converts the free-text query into a structured `UserProfile`.
- **Retrieve:** the original SoundMatch scorer ranks the catalog. Each specified
  preference contributes a weighted term. Genre and mood are exact-match bonuses
  (weight 2.0 each). Numeric features are scored by closeness `1 - abs(target -
  value)`: energy weight 3.0, valence 2.0, danceability 1.0, acousticness 1.0,
  tempo 1.0 after normalizing BPM to 0 to 1. The final score is the weighted
  average over only the specified preferences, so it always lands in 0 to 1.
- **Generate:** the model receives only the top 5 retrieved songs and writes the
  recommendation.
- **Verify:** the system confirms the recommended title is one of those 5,
  otherwise it substitutes a template built from the scorer's own reasons.

## Reliability mechanisms
Input validation (JSON schema check, genre vocabulary check, numeric clamping),
an output grounding guardrail, keyword and template fallbacks so a bad model
response never crashes the run, structured logging of every stage, and a
14-test evaluation suite.

## Responsible AI Reflection

### What are the limitations or biases in your system?
- **Energy-gap filter bubble.** Because energy carries the highest weight,
  mid-energy songs match almost everyone and users with extreme taste get
  under-served. Carried over from the original SoundMatch.
- **Dataset skew.** Lofi and chill songs are over-represented in the catalog, so
  calm queries have more good matches than intense ones.
- **Sparse mood matches.** Most songs have a unique mood word, so exact mood
  matching rarely fires and mostly acts as a tie-breaker.
- **Exact-match unfairness.** Genre matching is exact, so a near-synonym like
  "rap" against "hip-hop" would score zero on genre. The LLM parse step reduces
  this by mapping loose language onto catalog vocabulary, but only for the four
  genres that exist.
- **Small model parse errors.** With the stub backend, and sometimes with a
  small Ollama model, the parsed mood can be slightly off (for example tagging a
  study request as "sleepy"). The guardrail still keeps the final pick grounded,
  but the parse is the weakest link.
- **Empty-profile queries.** A request with nothing the parser recognizes (for
  example a genre not in the catalog) produces an all-zero ranking, and the
  system returns the lowest-id song. It stays grounded but the pick is
  effectively arbitrary.

### Could your AI be misused, and how would you prevent that?
The main risk is presenting a made-up recommendation as if it were real, since
the model writes confident natural-language blurbs. Someone could also point the
parser at offensive or manipulative input to try to steer the output. I limit
both in the system itself. The grounding guardrail means the model can only ever
recommend a song that actually exists in the catalog, so it cannot invent a
track or endorse something outside the fixed data. Input is validated and
clamped before it reaches the scorer, so a hostile or malformed query cannot push
values out of range. And because the catalog is small, fixed, and fictional,
there is no personal or sensitive data to leak. In a real product I would add
rate limiting, content filtering on the query, and a visible label telling users
the blurb is generated, not editorial.

### What surprised you while testing your AI's reliability?
What surprised me was how convincingly wrong AI-written code can look. My offline
stub model was supposed to read the user's query, but the code fed it the entire
prompt, including my instruction text. My instructions contained words like
"calm" and "intense" as part of the field descriptions, so the stub matched those
instruction words instead of the actual request. Every single query returned the
exact same recommendation. The code read as completely reasonable and I only
caught it by running the demo and seeing the same song six times in a row. It
taught me that "looks right" is not "is right," and that the guardrail and tests
are what actually prove the system works.

### Describe your collaboration with AI during this project.
I used an AI assistant as a design partner and pair programmer. I described my
original SoundMatch project and the problem I wanted to solve, and we talked
through directions before settling on RAG plus a reliability harness. It helped
me scaffold the four-stage pipeline, draft the scoring code, and write the tests.

- **One helpful suggestion.** The AI proposed the grounding guardrail: give the
  model only the retrieved songs, then verify its pick is one of them and fall
  back to a template otherwise. That single check is what makes this real RAG
  instead of a chatbot with a scorer attached, and it became the core of my
  reliability component. I would not have framed the project around it on my own.
- **One flawed suggestion.** The AI's first version of the offline stub was
  broken, as described above: it parsed the whole prompt instead of just the
  user's query, so every request produced an identical profile. The fix was to
  read only the text after "listener request:". It was a clear reminder that I
  have to run and verify AI-written code, not just read it.

## Evaluation
`tests/test_reliability.py` runs the full pipeline on the stub backend across
all six demo queries and asserts every recommendation is grounded. It also tests
hallucination rejection, garbage-output robustness, deterministic retrieval,
numeric clamping, and vocabulary filtering. All 14 tests pass.

## Future work
Fuzzy genre matching with synonyms, a larger and more balanced catalog, an
agentic step that re-queries when the top scores are all low, and swapping the
stub for a small fine-tuned parser so the parse step is more reliable offline.
