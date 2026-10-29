# DeepDive self-evaluation results

Generated 2026-10-29 05:13 UTC on 24 questions from `eval/questions.jsonl`.

| Metric | Value |
|---|---|
| Runs completed | 24 / 24 |
| Verifier catch rate (planted false claims) | 94.7% of 57 |
|   · catch rate: entity | 95.0% of 20 |
|   · catch rate: negation | 94.1% of 17 |
|   · catch rate: number | 100.0% of 8 |
|   · catch rate: year | 91.7% of 12 |
| Wrongful rejection rate (re-verified supported claims) | 0.0% of 60 |
| Probe / judge calls that errored (excluded from the rates above) | 0 / 0 |
| Independent groundedness (judge: google/gemini-2.5-flash-lite) | 97.0% of 100 |
| Agreement with hand labels (accuracy) | 86.0% of 50 |
| Agreement with hand labels (Cohen's kappa) | 0.70 |
| Quote validity rate | 99.3% |
| Drafted claims supported / partial / rejected | 1156 / 155 / 165 of 1476 |
| Draft claims rejected by verifier | 11.2% |
| Report paragraphs dropped for citing unverified claims | 29 |
| Runs that looped back to research | 37.5% |
| Mean extra research rounds | 0.50 |
| Sub-questions with enough verified support | 99.1% of 109 |
| Latency per run (p50 / p95, s) | 211 / 392 |

## What these numbers mean

- **Catch rate**: we take claims the verifier supported, alter one fact (a number, a year, the
  direction of a change, or the named entity) and verify the altered claim against the same
  source. A good verifier rejects them.
- **Wrongful rejection rate**: the same supported claims verified again, unaltered. A verifier
  that "catches" everything by rejecting everything would score high here.
- **Independent groundedness**: a judge from a third model family (not the drafter's, not the
  verifier's) checks claims cited in final reports against their sources.
- **Agreement with hand labels**: the owner labelled 50 claims by reading the source text, without seeing the verifier's answer (`eval/annotate.py`); partial verdicts count as supported.

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
