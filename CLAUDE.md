# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DeepDive is a hackathon entry (Agent Arena 2026, Track 01): a multi-agent research system that produces cited reports in which every claim has been verified against the exact source text. The design is in `docs/superpowers/specs/2026-10-18-deepdive-design.md`. The task-by-task build plan is in `docs/superpowers/plans/2026-10-18-deepdive-research-agent.md`; check the plan's progress ledger at `.superpowers/sdd/2026-10-18-deepdive-research-agent/progress.md` before starting work.

## Commands

```bash
uv sync                                   # install (Python 3.11 is pinned)
uv run pytest -q                          # all tests (no network; fakes in tests/fakes.py)
uv run pytest tests/test_config.py::test_mesh_base_url_literal -q   # single test
uv run ruff check . && uv run ruff format .                         # lint + format
uv run python -m compileall -q app eval tests scripts  # same compile check the hackathon CI runs on 3.11
uv run uvicorn app.main:create_app --factory --reload               # dev server
uv export --no-hashes --no-dev -o requirements.txt                  # after ANY dependency change
```

## Hard rules (judged; breaking these can disqualify the submission)

- All `.py` files must compile on **Python 3.11**. No 3.12-only syntax: no same-quote nesting inside f-strings (`f"{d["k"]}"`), no `type X = ...`, no `def f[T]()`. ruff `target-version = "py311"`.
- **Every** LLM, embedding and moderation call goes through `app/llm.py` → Mesh at base URL `https://api.meshapi.ai/v1` (literal string in `app/config.py`), key from env `MESH_API_KEY`. Never let Chroma use its built-in local embedding function; always pass embeddings explicitly.
- Tavily is used for raw search results only. Never pass `include_answer=True`.
- Model IDs come from settings/env, never hardcoded at call sites. The Verifier's model family must differ from the Drafter's (enforced by a `Settings` validator).
- `.github/workflows/agent-arena-2026-checks.yml` is the organizers' file and must stay byte-for-byte unchanged.
- `.env` is never committed; secrets are never baked into the Docker image.
- No stubbed features: every agent is called, the loop really loops, and eval numbers come from real runs. The judges' automated reviewer penalizes fakes.

## Architecture (big picture)

- **Orchestration:** a LangGraph `StateGraph` (`app/graph/build.py`) is the coordinator: `plan → research (parallel via Send, one per sub-question) → draft → verify → route`. The route either loops back through `refine → research` for "thin" sub-questions (fewer than `MIN_SUPPORTED_CLAIMS` supported claims, at most `MAX_EXTRA_ROUNDS` extra rounds) or continues to `contradictions → synthesize`.
- **State is thin; the DB is the source of truth.** `ResearchState` carries only `run_id`, `question`, `round`, `pending_sq_ids` and `researched`. Sources, chunks, claims, verdicts and the event log live in SQLite (SQLModel, `app/models.py`); chunk embeddings live in Chroma, keyed by chunk id.
- **Dependency injection:** nodes get collaborators from `config["configurable"]["deps"]` (`app/deps.py`: llm, search, fetcher, vectors, session_factory). Tests swap in the fakes from `tests/fakes.py`; real wiring is `build_deps(settings)`.
- **Verification core:** the Drafter emits atomic claims citing chunk ids (claims citing ids it wasn't given are dropped). The Verifier judges each claim against only its cited chunk and must return a verbatim quote. `app/tools/quote_match.py` then checks that the quote really exists in the chunk; if it doesn't, the claim becomes `unsupported`. The Synthesizer may cite only supported/partial claims, and paragraphs citing anything else are dropped.
- **Audit trail:** every node writes `Event` rows via `log_event()`. The same rows feed the SSE live timeline (`/api/runs/{id}/events`, replayed from the DB) and the trail page. LLM usage is logged as `llm_call` events.
- **Eval** (`eval/`): planted false claims (mutations) measure the verifier's catch rate; a third-family judge model checks groundedness independently; ~50 hand-labelled claims in `eval/gold/` measure agreement. Results go to `eval/results/*.json` and `eval/RESULTS.md`.
- **Deploy:** Docker (`docker-compose.yml`, `./data` volume) → GitHub Actions `deploy.yml` (OIDC role if `AWS_ROLE_ARN` is set, else the scoped deploy user's access keys → ECR → EC2 via SSM) behind Caddy. See `docs/DEPLOY.md`.
