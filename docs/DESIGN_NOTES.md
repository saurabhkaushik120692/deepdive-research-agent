# Key design decisions

Short notes on the choices that shape DeepDive, and why. The full design is in
`docs/superpowers/specs/2026-10-18-deepdive-design.md`.

## Verification

- **Atomic claims, judged one at a time against only their cited chunk.** A verifier that sees the
  whole report or all sources can "find" support somewhere else; isolation makes each verdict
  about one sentence and one piece of text.
- **The verifier must quote, and code checks the quote.** LLM verifiers can hallucinate support
  as easily as writers hallucinate facts. `quote_in_chunk` normalizes text (quotes, spaces, case),
  tries a punctuation-insensitive exact match, then fuzzy matching for long quotes, and requires
  every number in the quote to appear in the source. A one-digit change otherwise scores ~97%
  similar and would pass. The fuzzy fallback may not change meaning: every quoted word must occur
  in the matched source passage, with the same negations ("was not effective" never matches "was
  effective"). A failed check turns the verdict into `unsupported`.
- **Fail closed.** An unparseable verifier reply rejects that claim instead of passing it.
- **Cross-family verifier.** The drafter and verifier come from different model families
  (`openai/*` vs `anthropic/*`), enforced by a settings validator, so a model doesn't approve
  its own style of mistake.
- **Drafter and Synthesizer are separate agents.** The synthesizer only ever receives verified
  claims. Code then drops any paragraph that cites a claim outside that set, or that states a
  number or year its cited claims don't contain; the summary gets the same figure check. The
  prose wording itself is not verified word for word: a paragraph can still mis-phrase its
  claims, which is why every paragraph carries its citations and the claims are kept in the trail.
- **Re-research can't count the same fact twice.** Later rounds tell the drafter what was already
  drafted, and claims matching an existing one (ignoring case and punctuation) are dropped, so a
  thin sub-question only passes the loop rule with genuinely new supported facts.

## The loop

- **The loop rule is plain code, not an LLM decision** (`app/graph/rules.py`): fewer than 2
  supported claims for a sub-question triggers re-research, capped at 2 extra rounds. It's
  predictable, testable and visible in the trail (`loop_decision` events).
- Re-research uses a **refined query written from the rejected claims and their reasons**,
  skips URLs already used for that sub-question, and only the thin sub-questions are re-run.

## Architecture

- **Thin graph state, database as the source of truth.** LangGraph state carries ids only;
  everything else is in SQLite. The audit trail, the live SSE timeline and the eval all read the
  same `Event` rows, so what you see live is exactly what was recorded.
- **Dependency injection for every node** (`Deps`): the whole graph, including the real loop,
  runs in tests with scripted fakes and no network. 100+ tests run in a few seconds.
- **SQLite + Chroma (embedded)**: zero setup for judges and a single Docker volume in production.
- **Every model call goes through one Mesh client** (`app/llm.py`), with JSON-schema
  instructions, one repair retry on invalid JSON, backoff on transient errors, bounded
  concurrency and per-call token logging. Chroma never uses its built-in local embedding model;
  embeddings are computed through Mesh too (hackathon rule).
- **Search and fetch results are cached** in SQLite: eval re-runs are reproducible and cheaper.

## Evaluation

- **Planted false claims** measure what matters most, whether the verifier catches unsupported
  claims, per kind of error (number, year, direction, entity).
- **Wrongful rejection rate** guards against a verifier that looks good by rejecting everything.
- **A third-family judge** re-checks final-report claims independently of the verifier.
- **Hand labels** give a ground truth that no model produced.

## Engineering constraints

- **Python 3.11 syntax only.** The organizers' CI compiles every file with Python 3.11; ruff's
  `target-version = "py311"`, a pre-commit compile hook and a Claude Code edit hook catch
  3.12-only syntax early.
- **Public demo protection**: an access-code cookie gate, an hourly run limit, at most 2
  concurrent runs, and a moderation screen on questions (Mesh's free moderation model).

## Model changes after the day-1 smoke test

_Record here any model ID that had to change after `scripts/mesh_smoke.py` ran with the real key._
