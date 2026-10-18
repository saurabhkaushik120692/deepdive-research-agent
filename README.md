# DeepDive Research Agent

A self-verifying multi-agent research analyst (Agent Arena 2026, Track 01 - DeepDive).
Every claim in the final report is checked against the exact source text it cites.

Work in progress. See `docs/superpowers/specs/2026-10-18-deepdive-design.md` for the design.

## Run locally

```bash
uv sync
cp .env.example .env   # fill in MESH_API_KEY and TAVILY_API_KEY
uv run uvicorn app.main:create_app --factory --reload
```
