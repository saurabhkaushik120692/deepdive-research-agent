# DeepDive self-evaluation results

Generated 2026-10-21 15:33 UTC on 24 questions from `eval/questions.jsonl`.

| Metric | Value |
|---|---|
| Runs completed | 24 / 24 |
| Verifier catch rate (planted false claims) | 87.0% of 54 |
|   · catch rate: entity | 78.9% of 19 |
|   · catch rate: negation | 94.7% of 19 |
|   · catch rate: number | 100.0% of 7 |
|   · catch rate: year | 77.8% of 9 |
| Wrongful rejection rate (re-verified supported claims) | 0.0% of 60 |
| Probe / judge calls that errored (excluded from the rates above) | 0 / 0 |
| Independent groundedness (judge: google/gemini-2.5-flash-lite) | 94.0% of 100 |
| Agreement with hand labels (accuracy) | 86.0% of 50 |
| Agreement with hand labels (Cohen's kappa) | 0.70 |
| Quote validity rate | 99.5% |
| Drafted claims supported / partial / rejected | 1200 / 125 / 108 of 1433 |
| Draft claims rejected by verifier | 7.5% |
| Report paragraphs dropped for citing unverified claims | 15 |
| Runs that looped back to research | 20.8% |
| Mean extra research rounds | 0.21 |
| Sub-questions with enough verified support | 100.0% of 115 |
| Cost per run (mean / p95, USD) | 0.0389 / 0.0557 |
| Latency per run (p50 / p95, s) | 242 / 383 |

## What these numbers mean

- **Catch rate**: we take claims the verifier supported, alter one fact (a number, a year, the
  direction of a change, or the named entity) and verify the altered claim against the same
  source. A good verifier rejects them.
- **Wrongful rejection rate**: the same supported claims verified again, unaltered. A verifier
  that "catches" everything by rejecting everything would score high here.
- **Independent groundedness**: a judge from a third model family (not the drafter's, not the
  verifier's) checks claims cited in final reports against their sources.
- **Agreement with hand labels**: the owner labelled claims by reading the source text; n/a until
  `eval/gold/claims_gold.jsonl` is filled in.

## Models

- planner: `openai/gpt-5-nano`
- drafter: `openai/gpt-5-nano`
- verifier: `anthropic/claude-haiku-5.5`
- synthesizer: `openai/gpt-5-nano`
- contradictions: `openai/gpt-5-nano`
- embed: `google/embeddinggemma-300m`
- judge: `google/gemini-2.5-flash-lite`

Failed runs: none

## How to reproduce

```bash
uv sync
cp .env.example .env   # MESH_API_KEY, TAVILY_API_KEY
uv run python -m eval.run                       # research all questions + evaluate
uv run python -m eval.label --n 50              # build the hand-labelling set
uv run python -m eval.run --skip-research       # re-evaluate stored runs (e.g. after labelling)
```

Raw numbers: the matching JSON file in `eval/results/`.
