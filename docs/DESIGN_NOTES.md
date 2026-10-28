# Key design decisions

Short notes on the choices that shape DeepDive, and why.

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
- **Names are checked in code too.** A valid quote only proves the quote is in the source; the
  claim can still name someone else ("BloombergNEF says…" approved on a quote where the IEA says
  it), which is how 1 in 4 swapped entities got through. `missing_names` requires every name in
  the claim to appear in the cited text: capitalized words after the first, acronyms, and a first
  word that is clearly a name. It accepts forms of the same name (China/Chinese, RBI/Reserve Bank
  of India, US/U.S., Fed/Federal Reserve) and ignores months, currencies, units and company
  suffixes, which an earlier version flagged in 2% of approved claims. A missing name turns the
  verdict into `unsupported` with the name as the reason.
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

## Follow-up questions

- **A second graph, not a chat prompt.** A follow-up runs scope check → draft → verify → (one web
  search if thin) → answer, as its own LangGraph graph. A chat model answering from the report text
  would be fast but unchecked; this keeps every sentence tied to a verified claim.
- **Reuse before searching.** The run's saved chunks are searched first, and the report's verified
  claims that cite those chunks can be cited directly instead of being verified again. Verifying a
  fact twice could give it two different verdicts on one page.
- **The loop counts only new facts.** The user is asking for more than the report says, so the
  web search is triggered when fewer than 2 *new* claims are verified, whatever the report holds.
- **No back door for rejected claims.** A drafted claim matching any report claim (supported or
  rejected) is dropped, so a follow-up can't bring back something the verifier threw out.
- **Scope is checked by a model, the limits by code.** Whether a question is on topic needs
  language understanding; who may ask (the report's owner, not on examples, within 30 days, 10 per
  report, one at a time) is plain code. Off-topic questions cost one cheap call.
- **Fast enough to feel conversational.** The scope check, follow-up drafting and answer use low
  reasoning effort; the verifier keeps its default. Follow-up rows carry a `followup_id`, and every
  report query filters on it, so the report, its counts and its trail never change afterwards.

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
  `target-version = "py311"` and a pre-commit compile hook catch 3.12-only syntax early.
- **Public demo protection**: Google sign-in to start a run, an hourly run limit, at most 2
  concurrent runs, and a moderation screen on questions (Mesh's free moderation model).
- **Security review before going public**: full-history secret scan (no keys ever committed),
  `pip-audit` on the locked dependencies (certifi upgraded; the open chromadb advisories concern
  Chroma's standalone HTTP server, which DeepDive never runs; it uses the embedded client), no
  unescaped template output, parameterized SQL only, and the reverse proxy receives only its
  hostname, never the API keys.

## Model changes after the day-1 smoke test

None. All configured models (`openai/gpt-5-nano`, `anthropic/claude-haiku-5.5`,
`google/gemini-2.5-flash-lite`, `google/embeddinggemma-300m`, `openai/omni-moderation-latest`)
answered in JSON mode through Mesh on the first smoke test.

## What live runs taught us

Each of these passed the unit tests and only showed up against real services. Each now has a
regression test.

| Found in | Problem | Fix |
|---|---|---|
| First live run | Search results arrive as markdown full of links: 600 "words" were ~14k characters, over the embedding model's 2048-token input limit | Strip links/URLs, cap chunks at 3,000 characters, truncate embedding input to 1,000 |
| Second live run | 9.5 min per run: sub-questions were drafted one after another, each a 45–120 s reasoning call | Draft and check contradictions for all sub-questions concurrently → ~4 min |
| Second live run | The synthesizer copied internal "(partial)" markers and added attributions ("according to the IEA") that its claims didn't contain | Prompt rules + a code check that prose figures come from the cited claims |
| Live runs | A 60 s timeout cut off normal 60–120 s reasoning calls, which were retried (and billed) from scratch | 180 s timeout, still bounding truly hung calls |
| First full eval | Mesh allows 100 requests/minute; two runs at once burst past it and short retries gave up (12 of 24 runs failed) | Shared sliding-window limiter at 80/min; 429s wait for the window to reset |
| First full eval | Number tables, leader-dot tables of contents and SVG markup pack >1 token per character (3 runs failed) | Drop chunks that are mostly digits/symbols; they held no quotable evidence anyway |
| Second full eval | An empty verifier reply was echoed back in the JSON-repair turn; Anthropic rejects empty messages, failing the call | Never send an empty assistant turn; eval counts per-claim API errors instead of aborting |
