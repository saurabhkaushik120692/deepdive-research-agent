# DeepDive Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DeepDive, a FastAPI + LangGraph multi-agent research system that writes cited reports in which every claim has been verified against the exact source text. It includes a loop back to research, a full audit trail, a measured self-evaluation, and a Docker → ECR → EC2 deployment.

**Architecture:** A LangGraph `StateGraph` coordinates five agents: Planner, parallel Researchers (via `Send`), Drafter, Verifier and Synthesizer. Graph state carries only IDs and counters. All content (sources, chunks, claims, verdicts, events) lives in SQLite through SQLModel, and chunk embeddings live in Chroma. Every model call goes through one Mesh client (`app/llm.py`). Collaborators are injected through a `Deps` object, so every node can be tested with fakes and without network access.

**Tech Stack:** Python 3.11, uv, FastAPI, LangGraph, openai SDK (pointed at Mesh), SQLModel/SQLite, Chroma (persistent), Tavily, httpx, trafilatura, rapidfuzz, Jinja2 + HTMX (SSE extension) + Pico.css, sse-starlette, pytest + pytest-asyncio, ruff, Docker, GitHub Actions, AWS ECR/EC2/SSM.

**Spec:** `docs/superpowers/specs/2026-10-18-deepdive-design.md`

**Schedule (spec §11):** Tasks 1–3 → day 1 · 4–5 → day 2 · 6–9 → days 3–4 · 10–12 → days 5–6 · 13–14 → day 7 · 15–16 → days 8–9 · 17 → day 10 · 18 → day 11.

## Global Constraints

- All `.py` files must compile on **Python 3.11**. No 3.12-only syntax: no same-quote nesting inside f-strings (`f"{d["k"]}"`), no `type X = ...`, no `def f[T]()`. ruff `target-version = "py311"`.
- **Every** LLM, embedding and moderation call goes through `app/llm.py` → Mesh at base URL `https://api.meshapi.ai/v1` (literal string in `app/config.py`), key from env `MESH_API_KEY`.
- Tavily is used for raw search results only. Never pass `include_answer=True`.
- Model IDs come from settings/env, never hardcoded at call sites. Defaults (spec §4.5): planner/drafter/synthesizer `openai/gpt-5-nano`, verifier `anthropic/claude-haiku-5.5`, judge `google/gemini-2.5-flash-lite`, embed `google/embeddinggemma-300m`, moderation `openai/omni-moderation-latest`.
- The Verifier's model family must differ from the Drafter's. `Settings` validation fails at startup if the provider prefix (text before `/`) is the same for both.
- `MIN_SUPPORTED_CLAIMS = 2`, `MAX_EXTRA_ROUNDS = 2`, `QUOTE_MATCH_THRESHOLD = 90`.
- `.github/workflows/agent-arena-2026-checks.yml` is committed byte-for-byte unchanged from `E:\Krish\Hackathon\agent-arena-2026-checks.yml`.
- `.env` is never committed; secrets are never baked into the Docker image.
- No stubbed features: every agent is called, the loop really loops, and eval numbers come from real runs.
- Tests never touch the network. Use `tests/fakes.py` and `tmp_path`.
- Whenever dependencies change, regenerate `requirements.txt` with `uv export --no-hashes --no-dev -o requirements.txt` in the same commit.

## Review Focus

1. **A fetched page is empty, paywalled, a PDF, or returns 403/timeout.** The researcher skips that source with a `source_skipped` event and the run continues. If every source fails, the sub-question becomes thin rather than crashing. Pinned in Task 8 and Task 10.
2. **Mesh errors mid-run (429/5xx after retries, timeout).** The run ends with `status="failed"` and a `run_failed` event carrying the error message. The UI shows the failure instead of an endless spinner. Pinned in Task 12 and Task 13.
3. **Quotes with curly quotes, non-breaking spaces, line breaks or changed case.** These still match. Short fabricated quotes (< 20 chars) don't pass on fuzziness. Pinned in Task 6.
4. **An obscure question with zero search results.** After the loop cap, the report states the evidence is insufficient for each empty sub-question. It is never an empty report or a crash. Pinned in Task 12.
5. **The SSE client connects after the run has finished, or reconnects mid-run.** It receives every event from the start (or after `Last-Event-ID`) and then a closing `done` event. Pinned in Task 13.

---

## File Structure

```
app/
  config.py           Settings (pydantic-settings); MESH base URL literal
  main.py             FastAPI app factory, mounts routers/static/templates
  db.py               engine, init_db(), get_session()
  models.py           SQLModel tables
  events.py           log_event()
  llm.py              MeshLlm: chat_json / embed / moderate (+ usage events)
  deps.py             Deps dataclass + build_deps(settings)
  cli.py              `python -m app.cli "question"`
  report.py           render_report() → markdown
  tools/
    quote_match.py    normalize(), quote_in_chunk()
    tiers.py          source_tier()
    chunking.py       chunk_text()
    search.py         TavilySearch (+cache)
    fetch.py          HttpFetcher (+cache)
    cache.py          cache_get()/cache_put()
    vectorstore.py    ChromaStore
  graph/
    schemas.py        Pydantic I/O schemas for every agent
    state.py          ResearchState
    rules.py          thin_sub_questions(), should_loop()
    nodes/planner.py, researcher.py, drafter.py, verifier.py, synthesizer.py, contradictions.py
    build.py          build_graph()
    runner.py         create_run(), run_research()
  prompts/            planner.md, refine.md, drafter.md, verifier.md, synthesizer.md, contradictions.md, judge.md
  api/runs.py, api/pages.py, api/guard.py
  templates/          base.html, ask.html, run.html, report.html, trail.html, eval.html, gate.html
  static/             app.css, print.css
eval/
  questions.jsonl, mutate.py, metrics.py, judge.py, run.py, label.py, gold/claims_gold.jsonl, results/, RESULTS.md
tests/                conftest.py, fakes.py, test_*.py
deploy/               ec2-bootstrap.sh, Caddyfile, docker-compose.prod.yml
docs/                 DEPLOY.md, DESIGN_NOTES.md
.github/workflows/    agent-arena-2026-checks.yml, ci.yml, deploy.yml
Dockerfile, docker-compose.yml, .dockerignore, pyproject.toml, uv.lock, requirements.txt,
.env.example, .gitignore, .pre-commit-config.yaml, README.md, CLAUDE.md
```

---

## Phase A: Foundation (day 1)

### Task 0: Accounts (owner does these by hand; not code)

- [ ] **Step 1:** Register for Agent Arena 2026 (closes **24 Oct 23:00 IST**). Copy the `SUBMISSION_TOKEN` from the dashboard.
- [ ] **Step 2:** Create a Mesh account and API key (`rsk_…`); top up about $5. In the Mesh dashboard, confirm the exact IDs of the six default models in Global Constraints exist.
- [ ] **Step 3:** Create a Tavily account and API key (free tier).
- [ ] **Step 4:** Create a **private** GitHub repo `deepdive-research-agent`. Add the repo secrets `MESH_API_KEY` and `SUBMISSION_TOKEN`.
- [ ] **Step 5:** Create the AWS account. Immediately enable root MFA and create a **$10 monthly budget alert** (Billing → Budgets). The rest of the AWS setup is in Task 5.

### Task 1: Repo scaffold, tooling, settings, health endpoint

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.dockerignore`, `.env.example`, `.pre-commit-config.yaml`, `CLAUDE.md`, `README.md` (stub), `app/__init__.py`, `app/config.py`, `app/main.py`, `tests/conftest.py`, `tests/test_health.py`, `tests/test_config.py`, `.github/workflows/agent-arena-2026-checks.yml` (copied), `.github/workflows/ci.yml`, `requirements.txt` (generated)

**Interfaces:**
- Produces: `class Settings(BaseSettings)` with the fields below; `get_settings() -> Settings` (lru_cached); `create_app(settings: Settings | None = None) -> FastAPI`; `GET /healthz → {"status": "ok"}`.

`Settings` fields (env names are the upper-case field names): `mesh_api_key: str = ""`, `mesh_base_url: str = "https://api.meshapi.ai/v1"`, `tavily_api_key: str = ""`, `data_dir: Path = Path("data")`, `database_url: str = "sqlite:///data/deepdive.db"`, `model_planner`, `model_drafter`, `model_synthesizer`, `model_contradictions` (default `"openai/gpt-5-nano"`), `model_verifier: str = "anthropic/claude-haiku-5.5"`, `model_judge: str = "google/gemini-2.5-flash-lite"`, `model_embed: str = "google/embeddinggemma-300m"`, `model_moderation: str = "openai/omni-moderation-latest"`, `min_supported_claims: int = 2`, `max_extra_rounds: int = 2`, `quote_match_threshold: int = 90`, `search_results_per_query: int = 5`, `chunk_words: int = 600`, `chunk_overlap_words: int = 75`, `max_chunks_per_source: int = 40`, `top_k_chunks: int = 6`, `llm_max_retries: int = 3`, `llm_concurrency: int = 8`, `demo_access_code: str = ""` (empty = no gate), `runs_per_hour: int = 10`, `max_concurrent_runs: int = 2`.

- [ ] **Step 1:** `git init` in `E:\Krish\Hackathon\deepdive-research-agent`, then `uv init --python 3.11 --no-workspace`. Pin `requires-python = ">=3.11,<3.12"`. Runtime deps: `fastapi`, `uvicorn[standard]`, `openai`, `langgraph`, `sqlmodel`, `pydantic-settings`, `chromadb`, `tavily-python`, `httpx`, `trafilatura`, `rapidfuzz`, `jinja2`, `sse-starlette`, `python-multipart`. Dev group: `pytest`, `pytest-asyncio`, `ruff`, `pre-commit`. Add `[tool.ruff] target-version = "py311"` and `[tool.pytest.ini_options] asyncio_mode = "auto"`.
- [ ] **Step 2:** Copy the checks workflow unchanged:
  ```bash
  mkdir -p .github/workflows && cp ../agent-arena-2026-checks.yml .github/workflows/agent-arena-2026-checks.yml
  ```
- [ ] **Step 3:** Write the failing tests:
  ```python
  # tests/test_config.py
  import pytest
  from app.config import Settings

  def test_mesh_base_url_literal():
      assert Settings().mesh_base_url == "https://api.meshapi.ai/v1"

  def test_verifier_must_differ_from_drafter_family():
      with pytest.raises(ValueError, match="different model family"):
          Settings(model_drafter="openai/gpt-5-nano", model_verifier="openai/gpt-4.1-nano")

  # tests/test_health.py
  from fastapi.testclient import TestClient
  from app.main import create_app

  def test_healthz():
      r = TestClient(create_app()).get("/healthz")
      assert r.status_code == 200 and r.json() == {"status": "ok"}
  ```
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: FAIL (`ModuleNotFoundError: app.config`).
- [ ] **Step 5:** Implement `app/config.py` (with a `model_validator` comparing `model_drafter.split("/")[0]` and `model_verifier.split("/")[0]`) and `app/main.py`.
- [ ] **Step 6:** Run `uv run pytest -q`. Expected: 3 passed.
- [ ] **Step 7:** Add the remaining tooling:
  - `.gitignore`: `.env`, `data/`, `.venv/`, `__pycache__/`, `.superpowers/`, `checks.py`
  - `.env.example`: every Settings env name, with empty secrets
  - `.pre-commit-config.yaml`: ruff, ruff-format, and a local hook running `python -m compileall -q app eval tests`
  - `ci.yml`: on push, set up uv + Python 3.11, then `uv sync --frozen`, `uv run ruff check .`, `uv run python -m compileall -q .`, `uv run pytest -q`
  - generate `requirements.txt` with `uv export --no-hashes --no-dev -o requirements.txt`
- [ ] **Step 8:** Run the `init` skill to create `CLAUDE.md`. Make sure it states the Global Constraints above verbatim. Then use the `update-config` skill to add a PostToolUse hook on Edit/Write that runs `uv run ruff check --quiet` and `uv run python -m compileall -q app eval tests`.
- [ ] **Step 9:** Commit, add the GitHub remote, and push. Verify in the GitHub Actions tab that **Agent Arena Checks** has all critical checks green and **ci** is green.
  ```bash
  git add -A && git commit -m "chore: scaffold project, settings, health endpoint, CI"
  ```

### Task 2: Database models and event log

**Files:**
- Create: `app/db.py`, `app/models.py`, `app/events.py`, `tests/test_models.py`

**Interfaces:**
- Produces: `init_db(engine) -> None`; `make_engine(database_url: str) -> Engine`; `get_session()` FastAPI dependency; `log_event(session, run_id: str, node: str, type: str, payload: dict | None = None) -> Event` (commits).
- Tables, using spec §6.2 field names:
  - `Run(id: str = uuid4 hex PK, question, status: Literal["queued","running","completed","failed"], models: dict JSON, error: str|None, created_at, finished_at, prompt_tokens: int = 0, completion_tokens: int = 0)`
  - `SubQuestion(id: int PK, run_id FK, text, query, round: int = 0, status: Literal["pending","researched","thin","ok"])`
  - `SearchRecord(id, sub_question_id FK, query, round, provider, n_results)`
  - `Source(id, run_id FK, url, domain, title, tier: int, fetched_at)`
  - `EvidenceChunk(id, source_id FK, run_id FK, sub_question_id FK, text, position: int)`
  - `Claim(id, run_id FK, sub_question_id FK, text, chunk_ids: list[int] JSON, round: int, in_draft: bool = True)`
  - `Verdict(id, claim_id FK unique, label: Literal["supported","partial","unsupported"], quote: str, quote_found: bool, reason: str, model: str)`
  - `Event(id: int PK autoincrement, run_id FK, node, type, ts, payload: dict JSON)`
  - `Report(run_id PK FK, markdown, metrics: dict JSON)`
  - `CacheEntry(key: str PK, value: str, created_at)`

- [ ] **Step 1:** Write the failing tests:
  ```python
  def test_run_roundtrip_with_claim_and_verdict(session):
      run = Run(question="EV battery risks 2026?", status="queued", models={})
      session.add(run); session.commit()
      sq = SubQuestion(run_id=run.id, text="Market size?", query="EV battery market size 2026")
      session.add(sq); session.commit()
      c = Claim(run_id=run.id, sub_question_id=sq.id, text="Demand rose 28%.", chunk_ids=[1, 2], round=0)
      session.add(c); session.commit()
      session.add(Verdict(claim_id=c.id, label="supported", quote="rose 28%", quote_found=True, reason="ok", model="m"))
      session.commit()
      got = session.get(Claim, c.id)
      assert got.chunk_ids == [1, 2]

  def test_log_event_is_ordered(session):
      run = Run(question="q", status="running", models={}); session.add(run); session.commit()
      e1 = log_event(session, run.id, "planner", "node_started")
      e2 = log_event(session, run.id, "planner", "node_finished", {"n": 3})
      assert e2.id > e1.id and e2.payload == {"n": 3}
  ```
  Put a `session` fixture in `tests/conftest.py`: an in-memory SQLite engine (`sqlite://` with `StaticPool`) plus `init_db`.
- [ ] **Step 2:** Run `uv run pytest tests/test_models.py -q`. Expected: FAIL (import error).
- [ ] **Step 3:** Implement `app/models.py` (JSON columns via `sa_column=Column(JSON)`), `app/db.py`, and `app/events.py`. `init_db` creates `settings.data_dir` if it's missing.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: SQLModel schema and event log`.

### Task 3: Mesh LLM client

**Files:**
- Create: `app/llm.py`, `tests/test_llm.py`, `tests/fakes.py` (start it here), `scripts/mesh_smoke.py`

**Interfaces:**
- Consumes: `Settings`, `log_event`.
- Produces:
  ```python
  Role = Literal["planner", "drafter", "verifier", "synthesizer", "judge", "contradictions"]
  class LlmOutputError(Exception): ...
  class LlmClient(Protocol):
      async def chat_json(self, *, role: Role, system: str, user: str, schema: type[T], run_id: str | None = None) -> T: ...
      async def embed(self, texts: list[str]) -> list[list[float]]: ...
      async def moderate(self, text: str) -> bool: ...   # True = flagged
  class MeshLlm(LlmClient):
      def __init__(self, settings: Settings, client: AsyncOpenAI | None = None, on_usage: Callable[[str | None, dict], None] | None = None)
  ```
- `chat_json` behaviour: model = `getattr(settings, f"model_{role}")`. Send `response_format={"type": "json_object"}`, with the JSON schema of `schema` appended to the system prompt. Parse with `schema.model_validate_json`. On a `ValidationError` or JSON error, make **one** repair call that includes the error text. If that fails too, raise `LlmOutputError`. Retry 429/5xx/timeouts with exponential backoff up to `llm_max_retries`, and use a global `asyncio.Semaphore(llm_concurrency)`. After each call, invoke `on_usage(run_id, {"role", "model", "prompt_tokens", "completion_tokens", "latency_ms"})`.
- `tests/fakes.py` → `FakeOpenAI` with scripted `chat.completions.create` responses (a list of strings or exceptions), `embeddings.create`, and `moderations.create`; it records calls.

- [ ] **Step 1:** Write the failing tests:
  ```python
  class Out(BaseModel):
      answer: str

  async def test_chat_json_parses_and_uses_role_model(settings):
      fake = FakeOpenAI(chat=['{"answer": "hi"}'])
      llm = MeshLlm(settings, client=fake)
      out = await llm.chat_json(role="verifier", system="s", user="u", schema=Out)
      assert out.answer == "hi"
      assert fake.chat_calls[0]["model"] == settings.model_verifier

  async def test_repair_retry_once_then_succeeds(settings):
      fake = FakeOpenAI(chat=['not json', '{"answer": "fixed"}'])
      out = await MeshLlm(settings, client=fake).chat_json(role="planner", system="s", user="u", schema=Out)
      assert out.answer == "fixed" and len(fake.chat_calls) == 2

  async def test_two_bad_outputs_raise(settings):
      fake = FakeOpenAI(chat=['nope', '{"wrong": 1}'])
      with pytest.raises(LlmOutputError):
          await MeshLlm(settings, client=fake).chat_json(role="planner", system="s", user="u", schema=Out)

  async def test_usage_callback_receives_tokens(settings):
      seen = []
      fake = FakeOpenAI(chat=['{"answer": "x"}'], usage=(120, 30))
      await MeshLlm(settings, client=fake, on_usage=lambda rid, u: seen.append((rid, u))).chat_json(
          role="drafter", system="s", user="u", schema=Out, run_id="r1")
      assert seen[0][0] == "r1" and seen[0][1]["prompt_tokens"] == 120

  async def test_moderate_returns_flag(settings):
      fake = FakeOpenAI(moderation_flagged=True)
      assert await MeshLlm(settings, client=fake).moderate("bad text") is True
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_llm.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `app/llm.py`. The default client is `AsyncOpenAI(base_url=settings.mesh_base_url, api_key=settings.mesh_api_key)`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Write `scripts/mesh_smoke.py`. With the real key, it calls `chat_json` once per role, `embed(["hello"])` and `moderate("hello")`, then prints the model, latency and tokens. Run `uv run python scripts/mesh_smoke.py`. Expected: every line OK. **If a default model ID or the embeddings endpoint fails, change the default in `Settings`/`.env.example` now and record the change in `docs/DESIGN_NOTES.md`.**
- [ ] **Step 6:** Commit: `feat: Mesh LLM client with JSON repair, retries, usage logging`.

---

## Phase B: Deployment skeleton (day 2, time-boxed to about 1 day)

### Task 4: Docker image and compose

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`

**Interfaces:**
- Produces: image serving `uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000`; compose service `app` with `env_file: .env`, port `8000:8000`, volume `./data:/app/data`, healthcheck on `/healthz`.

- [ ] **Step 1:** Write a multi-stage `Dockerfile`. Builder stage: `ghcr.io/astral-sh/uv:python3.11-bookworm-slim`, `uv sync --frozen --no-dev`. Final stage: `python:3.11-slim`, copy `.venv` and `app/`, run as a non-root user. The image must contain no `.env` (`.dockerignore` lists `.env`, `data/`, `.git/`, `tests/`, `eval/results/`).
- [ ] **Step 2:** Run `docker compose up --build -d`, then `curl -s localhost:8000/healthz`. Expected: `{"status":"ok"}`.
- [ ] **Step 3:** Run `docker compose exec app ls /app/.env`. Expected: "No such file" (the secrets come only from `env_file`).
- [ ] **Step 4:** Commit: `build: Docker image and compose with data volume`.

### Task 5: AWS pipeline (GitHub OIDC → ECR → EC2 via SSM)

**Files:**
- Create: `.github/workflows/deploy.yml`, `deploy/ec2-bootstrap.sh`, `deploy/docker-compose.prod.yml`, `deploy/Caddyfile`, `docs/DEPLOY.md`

**Interfaces:**
- Consumes: the image from Task 4.
- Produces: on push to `main`, jobs `test` (same steps as ci.yml) → `build-push` (tag `${{ github.sha }}` and `latest`) → `deploy` (SSM `send-command` running `cd /opt/deepdive && IMAGE_TAG=<sha> docker compose -f docker-compose.prod.yml pull && ... up -d`). GitHub repo variables: `AWS_REGION=ap-southeast-2`, `AWS_ROLE_ARN`, `ECR_REPOSITORY=deepdive-research-agent`, `EC2_INSTANCE_ID`.

- [ ] **Step 1:** Write `docs/DEPLOY.md` as a click-by-click runbook the **owner** follows in the AWS console:
  - add the IAM OIDC provider `token.actions.githubusercontent.com`
  - create a role trusted only for `repo:<owner>/deepdive-research-agent:ref:refs/heads/main`, with ECR push and `ssm:SendCommand` limited to that instance
  - create the ECR repo
  - launch EC2 t3.small (Amazon Linux 2023) with an instance profile that has `AmazonSSMManagedInstanceCore` and ECR read
  - security group: 80/443 open, 22 closed
  - allocate and attach an Elastic IP
  - put `/opt/deepdive/.env` on the host
- [ ] **Step 2:** Write `deploy/ec2-bootstrap.sh`, which is idempotent: install Docker and the compose plugin, ECR login helper, `mkdir -p /opt/deepdive/data`, copy the prod compose file and Caddyfile. The prod compose file runs `app` plus `caddy` (ports 80/443, `reverse_proxy app:8000`, site `{$SITE_HOST}` such as `<eip-with-dashes>.nip.io`). The data volume is `/opt/deepdive/data:/app/data`.
- [ ] **Step 3:** Write `deploy.yml` with `permissions: id-token: write, contents: read` and `aws-actions/configure-aws-credentials` using `role-to-assume`. Make sure it doesn't touch `agent-arena-2026-checks.yml`.
- [ ] **Step 4:** Push to `main`. Expected: all three jobs green, and `curl -s https://<eip-with-dashes>.nip.io/healthz` returns `{"status":"ok"}`.
- [ ] **Step 5:** Redeploy (an empty commit) and confirm a file in `/opt/deepdive/data` survives. Expected: the file is still present.
- [ ] **Step 6:** Commit: `ci: OIDC deploy pipeline to ECR and EC2 via SSM`.

---

## Phase C: Research pipeline (days 3–6)

### Task 6: Deterministic core (quote matching, source tiers, chunking, loop rules)

**Files:**
- Create: `app/tools/quote_match.py`, `app/tools/tiers.py`, `app/tools/chunking.py`, `app/graph/rules.py`, `tests/test_quote_match.py`, `tests/test_tiers.py`, `tests/test_chunking.py`, `tests/test_rules.py`

**Interfaces:**
- Produces:
  - `normalize(s: str) -> str`
  - `quote_in_chunk(quote: str, chunk: str, threshold: int = 90) -> bool`
  - `source_tier(url: str) -> Literal[1, 2, 3]`
  - `chunk_text(text: str, max_words: int = 600, overlap_words: int = 75) -> list[str]`
  - `thin_sub_questions(supported_counts: dict[int, int], min_supported: int = 2) -> list[int]` (sorted ids)
  - `should_loop(thin_ids: list[int], round_: int, max_extra_rounds: int = 2) -> bool`

- [ ] **Step 1:** Write the failing tests:
  ```python
  CHUNK = "In 2025, global EV battery demand rose 28% to 1.1 TWh, according to the IEA."

  def test_exact_quote(): assert quote_in_chunk("global EV battery demand rose 28% to 1.1 TWh", CHUNK)
  def test_case_and_whitespace(): assert quote_in_chunk("Global  EV battery\ndemand ROSE 28%", CHUNK)
  def test_curly_quotes_and_nbsp():
      chunk = "He said \u201cdemand rose\u00a028%\u201d in 2025."
      assert quote_in_chunk("demand rose 28%", chunk) and quote_in_chunk('"demand rose 28%"', chunk)
  def test_minor_formatting_passes_fuzzy(): assert quote_in_chunk("global EV battery demand rose 28 % to 1.1 TWh", CHUNK)
  def test_fabricated_quote_rejected(): assert not quote_in_chunk("global EV battery demand fell 12% to 0.9 TWh", CHUNK)
  def test_short_quote_requires_exact(): assert not quote_in_chunk("rose 29%", CHUNK)
  def test_empty_quote_rejected(): assert not quote_in_chunk("", CHUNK)

  @pytest.mark.parametrize("url,tier", [
      ("https://www.iea.org/reports/global-ev-outlook-2025", 1),
      ("https://www.sec.gov/Archives/edgar/x.htm", 1),
      ("https://pib.gov.in/release", 1),
      ("https://mit.edu/news/a", 1),
      ("https://www.reuters.com/business/autos/x", 2),
      ("https://medium.com/@someone/ev-take", 3),
      ("https://www.reddit.com/r/ev/x", 3),
  ])
  def test_source_tier(url, tier): assert source_tier(url) == tier

  def test_chunking_overlap_and_coverage():
      words = [f"w{i}" for i in range(1500)]
      chunks = chunk_text(" ".join(words), max_words=600, overlap_words=75)
      assert all(len(c.split()) <= 600 for c in chunks)
      assert chunks[1].split()[:75] == chunks[0].split()[-75:]
      assert chunks[-1].split()[-1] == "w1499"
  def test_chunking_empty_and_short():
      assert chunk_text("") == [] and chunk_text("a b c") == ["a b c"]

  def test_thin_detection(): assert thin_sub_questions({1: 3, 2: 1, 3: 0}, min_supported=2) == [2, 3]
  def test_loop_cap():
      assert should_loop([2], round_=0) and should_loop([2], round_=1)
      assert not should_loop([2], round_=2) and not should_loop([], round_=0)
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_quote_match.py tests/test_tiers.py tests/test_chunking.py tests/test_rules.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement:
  - `normalize`: NFKC, curly → straight quotes, NBSP → space, lowercase, collapse whitespace, strip surrounding quotes/punctuation.
  - `quote_in_chunk`: normalized substring match first. Quotes under 20 normalized characters must match exactly. Otherwise use `rapidfuzz.fuzz.partial_ratio >= threshold`.
  - `source_tier`: suffix rules for tier 1 (`.gov`, `.gov.in`, `.edu`, `.ac.uk`, `.int`, `.mil`) plus a `TIER1_DOMAINS` set (iea.org, worldbank.org, imf.org, oecd.org, who.int, nature.com, science.org, arxiv.org, sciencedirect.com, ncbi.nlm.nih.gov, rbi.org.in, sebi.gov.in); a `TIER2_DOMAINS` news set (reuters.com, bloomberg.com, ft.com, apnews.com, bbc.com, bbc.co.uk, economist.com, wsj.com, nytimes.com, theguardian.com, cnbc.com, thehindu.com, livemint.com, economictimes.indiatimes.com). Match on the registered domain with `www.` stripped; anything else is 3.
  - Chunking works on words. Rules follow the signatures above.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: quote matching, source tiers, chunking, loop rules`.

### Task 7: Agent I/O schemas, graph state, Deps, test fakes

**Files:**
- Create: `app/graph/schemas.py`, `app/graph/state.py`, `app/deps.py`; extend `tests/fakes.py`; create `tests/test_schemas.py`

**Interfaces:**
- Produces (schemas):
  ```python
  class PlannedSubQuestion(BaseModel): text: str; query: str
  class PlanOutput(BaseModel): sub_questions: list[PlannedSubQuestion]   # validator: 3 <= len <= 6
  class RefineOutput(BaseModel): query: str
  class DraftClaim(BaseModel): sub_question_id: int; text: str; chunk_ids: list[int]   # 1 <= len(chunk_ids) <= 2
  class DraftOutput(BaseModel): claims: list[DraftClaim]
  class VerdictOutput(BaseModel): label: Literal["supported","partial","unsupported"]; quote: str; reason: str
  class Paragraph(BaseModel): text: str; claim_ids: list[int]   # min 1
  class Section(BaseModel): heading: str; paragraphs: list[Paragraph]
  class SynthesisOutput(BaseModel): title: str; summary: str; sections: list[Section]
  class Conflict(BaseModel): claim_a: int; claim_b: int; explanation: str
  class ContradictionOutput(BaseModel): conflicts: list[Conflict]
  class JudgeOutput(BaseModel): supported: bool; reason: str
  ```
- State:
  ```python
  class ResearchState(TypedDict):
      run_id: str
      question: str
      round: int
      pending_sq_ids: list[int]
      researched: Annotated[list[int], operator.add]
  ```
- Deps:
  ```python
  @dataclass
  class SearchResult: url: str; title: str; content: str | None
  class SearchClient(Protocol):
      async def search(self, query: str, max_results: int) -> list[SearchResult]: ...
  class Fetcher(Protocol):
      async def fetch_text(self, url: str) -> str | None: ...
  class VectorStore(Protocol):
      async def add(self, run_id: str, sub_question_id: int, chunk_ids: list[int], texts: list[str]) -> None: ...
      async def query(self, run_id: str, sub_question_id: int, text: str, k: int) -> list[int]: ...
  @dataclass
  class Deps:
      settings: Settings; llm: LlmClient; search: SearchClient; fetcher: Fetcher; vectors: VectorStore
      session_factory: Callable[[], Session]
  def build_deps(settings: Settings) -> Deps   # real implementations (Tasks 8–9)
  ```
- Fakes in `tests/fakes.py`:
  - `ScriptedLlm(LlmClient)`: takes `handlers: dict[Role, Callable[[str, str], BaseModel | Exception]]` (receives system/user, returns a schema instance) and records `calls: list[(role, user)]`. `embed` returns deterministic hash vectors; `moderate` returns `self.flagged`.
  - `FakeSearch(results: dict[str, list[SearchResult]])`
  - `FakeFetcher(pages: dict[str, str | None])`
  - `MemoryVectors` returns the first k chunk ids stored for that sub-question.
  - A `deps` fixture in `conftest.py` wires these up with the in-memory session.

- [ ] **Step 1:** Write the failing tests:
  ```python
  def test_plan_requires_3_to_6():
      with pytest.raises(ValidationError): PlanOutput(sub_questions=[PlannedSubQuestion(text="a", query="a")] * 2)
      assert len(PlanOutput(sub_questions=[PlannedSubQuestion(text="a", query="a")] * 3).sub_questions) == 3
  def test_draft_claim_needs_1_or_2_chunks():
      with pytest.raises(ValidationError): DraftClaim(sub_question_id=1, text="x", chunk_ids=[])
      with pytest.raises(ValidationError): DraftClaim(sub_question_id=1, text="x", chunk_ids=[1, 2, 3])
  def test_paragraph_needs_a_claim():
      with pytest.raises(ValidationError): Paragraph(text="x", claim_ids=[])
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_schemas.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the schemas, state, deps and fakes per the interfaces.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: agent schemas, graph state, Deps and test fakes`.

### Task 8: Search, fetch and cache

**Files:**
- Create: `app/tools/cache.py`, `app/tools/search.py`, `app/tools/fetch.py`, `tests/test_search_fetch.py`

**Interfaces:**
- Consumes: `CacheEntry`, `SearchResult`, `Settings`.
- Produces:
  - `cache_get(session, key: str) -> str | None`, `cache_put(session, key: str, value: str) -> None`
  - `class TavilySearch(SearchClient)`: `__init__(settings, session_factory, client=None)`. Calls `client.search(query=..., max_results=..., include_raw_content=True, include_answer=False)` (async via `AsyncTavilyClient`); cache key `"search:" + query`.
  - `class HttpFetcher(Fetcher)`: `__init__(session_factory, http: httpx.AsyncClient | None = None, timeout_s: float = 15)`. Returns `None` for non-2xx responses, a non-`text/html` content type, timeouts, or extracted text shorter than 200 characters. Extracts with `trafilatura.extract`; cache key `"fetch:" + url` (failures are cached as `""` and returned as `None`).

- [ ] **Step 1:** Write the failing tests (Tavily client faked; httpx via `httpx.MockTransport`):
  ```python
  async def test_search_is_cached(session_factory):
      fake = FakeTavily({"q": [{"url": "https://a.com", "title": "A", "raw_content": "text"}]})
      s = TavilySearch(settings, session_factory, client=fake)
      r1 = await s.search("q", 5); r2 = await s.search("q", 5)
      assert r1 == r2 and r1[0].url == "https://a.com" and fake.calls == 1
      assert fake.last_kwargs["include_answer"] is False

  async def test_fetch_extracts_html(session_factory):
      html = "<html><body><article>" + "<p>Battery demand rose sharply this year.</p>" * 20 + "</article></body></html>"
      f = HttpFetcher(session_factory, http=mock_http(200, html, "text/html"))
      assert "Battery demand rose" in await f.fetch_text("https://a.com/x")

  @pytest.mark.parametrize("status,ctype", [(403, "text/html"), (200, "application/pdf")])
  async def test_fetch_failures_return_none(session_factory, status, ctype):
      assert await HttpFetcher(session_factory, http=mock_http(status, "x", ctype)).fetch_text("https://b.com") is None

  async def test_fetch_timeout_returns_none(session_factory):
      assert await HttpFetcher(session_factory, http=mock_http_timeout()).fetch_text("https://c.com") is None
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_search_fetch.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement per the interfaces.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: Tavily search and HTML fetch with caching`.

### Task 9: Chroma vector store

**Files:**
- Create: `app/tools/vectorstore.py`, `tests/test_vectorstore.py`

**Interfaces:**
- Consumes: `LlmClient.embed`.
- Produces: `class ChromaStore(VectorStore)`: `__init__(path: Path, embed: Callable[[list[str]], Awaitable[list[list[float]]]])`. A single collection `"evidence"`; ids `str(chunk_id)`; metadata `{"run_id", "sub_question_id"}`; `query` filters with `where={"$and": [{"run_id": ...}, {"sub_question_id": ...}]}`. Embeddings are always passed explicitly, so Chroma's built-in local embedding function is **never** used (Mesh rule).

- [ ] **Step 1:** Write the failing test, using a fake `embed` that maps texts with the word "battery" near the vector `[1, 0]` and others near `[0, 1]`:
  ```python
  async def test_query_filters_by_run_and_subq(tmp_path):
      store = ChromaStore(tmp_path, embed=fake_embed)
      await store.add("r1", 1, [10, 11], ["battery prices fell", "football scores"])
      await store.add("r2", 1, [20], ["battery prices fell"])
      assert await store.query("r1", 1, "battery cost", k=1) == [10]
      assert 20 not in await store.query("r1", 1, "battery cost", k=5)
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_vectorstore.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement with `chromadb.PersistentClient(path=str(path))`, wrapping the sync Chroma calls in `asyncio.to_thread`. Finish `build_deps` in `app/deps.py`: `MeshLlm(settings, on_usage=...)`, where `on_usage` opens a session and calls `log_event(session, run_id, "llm", "llm_call", usage)` when `run_id` is not None; `TavilySearch`; `HttpFetcher`; `ChromaStore(settings.data_dir / "chroma", embed=llm.embed)`; `session_factory` bound to `make_engine(settings.database_url)`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: Chroma evidence store with Mesh embeddings`.

### Task 10: Planner and Researcher nodes

**Files:**
- Create: `app/prompts/planner.md`, `app/prompts/refine.md`, `app/graph/nodes/planner.py`, `app/graph/nodes/researcher.py`, `tests/test_planner_researcher.py`

**Interfaces:**
- Consumes: `Deps`, schemas, `source_tier`, `chunk_text`, `log_event`.
- Produces:
  - `async def plan_node(state: ResearchState, config: RunnableConfig) -> dict`: creates `SubQuestion` rows (round 0) and returns `{"pending_sq_ids": [...]}`. Deps come from `config["configurable"]["deps"]`.
  - `async def refine_node(state, config) -> dict`: for each thin sub-question, calls role `planner` with `refine.md` (sub-question plus its rejected claims and reasons) → `RefineOutput`. Updates `SubQuestion.query` and `round`, and returns `{"round": state["round"] + 1, "pending_sq_ids": thin_ids}`.
  - `async def research_node(payload: dict, config) -> dict` with payload `{"run_id", "sub_question_id", "round"}`. It searches (`search_results_per_query`), takes `content` or falls back to `fetcher.fetch_text`, skips empty pages (event `source_skipped` with url and reason), and stores `Source` (with tier) and up to `max_chunks_per_source` `EvidenceChunk`s. It calls `vectors.add` and records a `SearchRecord`. It returns `{"researched": [sub_question_id]}`.
  - Events: `node_started`/`node_finished` from each node; `sub_questions_planned`, `search_done`, `source_skipped`, `query_refined`.

- [ ] **Step 1:** Write the failing tests:
  ```python
  async def test_plan_creates_sub_questions(deps, run):
      deps.llm.handlers["planner"] = lambda s, u: PlanOutput(sub_questions=[
          PlannedSubQuestion(text=f"q{i}", query=f"query {i}") for i in range(4)])
      out = await plan_node({"run_id": run.id, "question": run.question, "round": 0, "pending_sq_ids": [], "researched": []}, cfg(deps))
      assert len(out["pending_sq_ids"]) == 4
      assert event_types(deps, run.id) >= {"node_started", "sub_questions_planned", "node_finished"}

  async def test_research_stores_sources_with_tiers_and_chunks(deps, run, sq):
      deps.search.results[sq.query] = [SearchResult("https://www.iea.org/r", "IEA", "word " * 900),
                                        SearchResult("https://blog.example.com/p", "Blog", None)]
      deps.fetcher.pages["https://blog.example.com/p"] = "blog text " * 300
      out = await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
      assert out == {"researched": [sq.id]}
      tiers = sorted(s.tier for s in all_sources(deps, run.id))
      assert tiers == [1, 3] and chunk_count(deps, sq.id) >= 3

  async def test_research_survives_all_fetch_failures(deps, run, sq):
      deps.search.results[sq.query] = [SearchResult("https://x.com/a", "A", None)]
      deps.fetcher.pages["https://x.com/a"] = None
      out = await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
      assert out == {"researched": [sq.id]} and chunk_count(deps, sq.id) == 0
      assert "source_skipped" in event_types(deps, run.id)
  ```
  Helpers (`cfg`, `event_types`, `all_sources`, `chunk_count`, plus the `run`/`sq` fixtures) go in `tests/conftest.py`.
- [ ] **Step 2:** Run `uv run pytest tests/test_planner_researcher.py -q`. Expected: FAIL.
- [ ] **Step 3:** Write the prompts and nodes. `planner.md` must ask for 3–6 non-overlapping sub-questions, each with a web-search query.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: planner, refine and researcher nodes`.

### Task 11: Drafter and Verifier nodes

**Files:**
- Create: `app/prompts/drafter.md`, `app/prompts/verifier.md`, `app/graph/nodes/drafter.py`, `app/graph/nodes/verifier.py`, `tests/test_drafter_verifier.py`

**Interfaces:**
- Produces:
  - `async def draft_node(state, config) -> dict`: for each id in `state["pending_sq_ids"]`, retrieves `vectors.query(..., k=top_k_chunks)` and orders the chunks by source tier (1 first). It calls role `drafter` once per sub-question with those chunks labelled `[chunk:<id>]`. Claims whose `chunk_ids` aren't in the provided set are **dropped** (event `claim_dropped_bad_citation`); the rest are stored as `Claim(round=state["round"])`. It returns `{}`.
  - `async def verify_node(state, config) -> dict`: verifies every claim of the run that has no `Verdict` yet, concurrently. For each, it calls role `verifier` with **only** the claim text and the concatenated text of its cited chunks, then `quote_found = quote_in_chunk(out.quote, chunk_text, settings.quote_match_threshold)`. If the label isn't `unsupported` and `quote_found` is False, the stored label becomes `unsupported` with reason `"verifier quote not found in source"`. It stores the `Verdict` with `model=settings.model_verifier`, emits event `claim_verified` `{claim_id, label, quote_found}`, and returns `{}`.

- [ ] **Step 1:** Write the failing tests:
  ```python
  async def test_drafter_drops_claims_with_unknown_chunk_ids(deps, run, sq, chunks):  # chunks: ids [1, 2]
      deps.llm.handlers["drafter"] = lambda s, u: DraftOutput(claims=[
          DraftClaim(sub_question_id=sq.id, text="Demand rose 28%.", chunk_ids=[1]),
          DraftClaim(sub_question_id=sq.id, text="Invented.", chunk_ids=[999])])
      await draft_node(state(run, pending=[sq.id]), cfg(deps))
      assert [c.text for c in claims_of(deps, run.id)] == ["Demand rose 28%."]
      assert "claim_dropped_bad_citation" in event_types(deps, run.id)

  async def test_verifier_downgrades_fabricated_quote(deps, run, claim_on_chunk):  # chunk text = CHUNK from Task 6
      deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(label="supported", quote="demand fell 12% to 0.9 TWh", reason="r")
      await verify_node(state(run), cfg(deps))
      v = verdict_of(deps, claim_on_chunk.id)
      assert v.label == "unsupported" and v.quote_found is False and v.reason == "verifier quote not found in source"

  async def test_verifier_keeps_real_quote(deps, run, claim_on_chunk):
      deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(label="supported", quote="demand rose 28% to 1.1 TWh", reason="r")
      await verify_node(state(run), cfg(deps))
      assert verdict_of(deps, claim_on_chunk.id).label == "supported"

  async def test_verifier_sees_only_cited_chunk(deps, run, claim_on_chunk, other_chunk):
      deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(label="unsupported", quote="", reason="r")
      await verify_node(state(run), cfg(deps))
      role, user = deps.llm.calls[-1]
      assert other_chunk.text not in user

  async def test_verifier_skips_already_verified(deps, run, claim_on_chunk):
      deps.llm.handlers["verifier"] = lambda s, u: VerdictOutput(label="unsupported", quote="", reason="r")
      await verify_node(state(run), cfg(deps)); await verify_node(state(run), cfg(deps))
      assert sum(1 for r, _ in deps.llm.calls if r == "verifier") == 1
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_drafter_verifier.py -q`. Expected: FAIL.
- [ ] **Step 3:** Write the prompts and nodes. `drafter.md`: one fact per claim, cite only the given chunk ids, no outside knowledge. `verifier.md`: judge only against the given text; `quote` must be copied verbatim from it; use `partial` when the source supports only part of the claim.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: drafter with citation check and verifier with quote check`.

### Task 12: Synthesizer, report, graph wiring with loop, runner, CLI

**Files:**
- Create: `app/prompts/synthesizer.md`, `app/graph/nodes/synthesizer.py`, `app/report.py`, `app/graph/build.py`, `app/graph/runner.py`, `app/cli.py`, `tests/test_graph.py`, `tests/test_report.py`

**Interfaces:**
- Produces:
  - `async def synthesize_node(state, config) -> dict`: passes only claims labelled `supported`/`partial` (with ids and source tiers) to role `synthesizer` → `SynthesisOutput`. It drops paragraphs that cite claim ids outside that set (event `paragraph_dropped`), renders with `render_report`, and stores `Report(markdown, metrics)`, where `metrics` = `{"claims_drafted", "supported", "partial", "unsupported", "rounds", "sources", "tier_counts"}`.
  - `render_report(question: str, synthesis: SynthesisOutput, claims: dict[int, Claim], verdicts: dict[int, Verdict], chunk_sources: dict[int, Source], thin_sq_texts: list[str], conflicts: list[Conflict]) -> str`. The markdown has a title, a summary, sections where each paragraph ends with `[n]` citation markers, `⚠ partially verified` markers on partial claims, a `## Limited evidence` section listing `thin_sq_texts` (only if non-empty), `## Rejected claims` (claim text, reason), and `## Sources` (numbered `[n] title (url), tier T`).
  - `def route_after_verify(state, config) -> Literal["refine", "synthesize"]`: uses supported counts per sub-question from the DB with `thin_sub_questions`/`should_loop`. It sets `status="thin"` on thin sub-questions and `status="ok"` on the rest (so a sub-question fixed in a later round ends up `ok`), and emits `loop_decision {thin_ids, round, looping}`. The synthesizer's `thin_sq_texts` = sub-questions still `thin` at the end.
  - `def dispatch_research(state) -> list[Send]`: one `Send("research", {...})` per id in `pending_sq_ids`.
  - `build_graph() -> CompiledStateGraph`. Edges: `START → plan →(dispatch_research) research → draft → verify →(route_after_verify) refine | synthesize`; `refine →(dispatch_research) research`; `synthesize → END`. Task 17 inserts `contradictions` before `synthesize`.
  - `create_run(session, question: str, settings: Settings) -> Run` (status `queued`, `models` = the role → model map).
  - `async def run_research(run_id: str, deps: Deps) -> None`: sets status `running`; `await build_graph().ainvoke(initial_state, {"configurable": {"deps": deps}, "recursion_limit": 50})`; sets `completed` + `finished_at`. On **any** exception it sets `failed`, stores `error=str(e)`, and emits `run_failed`. Token totals are summed from `llm_call` events.
  - `app/cli.py`: `python -m app.cli "question"` creates a run, runs it with `build_deps(get_settings())`, and prints the report path and the metrics.

- [ ] **Step 1:** Write the failing tests (all with fakes):
  ```python
  async def test_loop_runs_once_for_thin_subquestion(deps):
      # Planner → sq A, B, C. In round 0 the verifier supports 2 claims for A and C but only 1 for B.
      # In round 1 (after refine) B gets 2 supported claims.
      script_happy_path(deps, thin_in_round0={"B"})
      run = create_run(session_of(deps), "EV battery risks 2026?", deps.settings)
      await run_research(run.id, deps)
      ev = events(deps, run.id)
      assert [e.payload["looping"] for e in ev if e.type == "loop_decision"] == [True, False]
      assert {e.payload["sub_question_id"] for e in ev if e.type == "search_done" and e.payload["round"] == 1} == {sq_id(deps, run.id, "B")}
      assert get_run(deps, run.id).status == "completed"

  async def test_loop_stops_at_cap_and_reports_limited_evidence(deps):
      script_happy_path(deps, thin_in_round0={"B"}, always_thin={"B"})
      run = create_run(session_of(deps), "q", deps.settings); await run_research(run.id, deps)
      assert sum(1 for e in events(deps, run.id) if e.type == "loop_decision" and e.payload["looping"]) == 2
      assert "## Limited evidence" in report_of(deps, run.id).markdown

  async def test_zero_search_results_gives_insufficient_report_not_crash(deps):
      script_no_results(deps)
      run = create_run(session_of(deps), "obscure q", deps.settings); await run_research(run.id, deps)
      r = get_run(deps, run.id)
      assert r.status == "completed" and "## Limited evidence" in report_of(deps, run.id).markdown

  async def test_llm_failure_marks_run_failed(deps):
      deps.llm.handlers["planner"] = lambda s, u: LlmOutputError("mesh 503 after retries")
      run = create_run(session_of(deps), "q", deps.settings); await run_research(run.id, deps)
      r = get_run(deps, run.id)
      assert r.status == "failed" and "503" in r.error and "run_failed" in event_types(deps, run.id)

  async def test_synthesizer_cannot_cite_rejected_claims(deps):
      script_happy_path(deps, synth_cites_rejected=True)
      run = create_run(session_of(deps), "q", deps.settings); await run_research(run.id, deps)
      md = report_of(deps, run.id).markdown
      rejected_text = rejected_claim_text(deps, run.id)
      body = md.split("## Rejected claims")[0]
      assert rejected_text not in body and rejected_text in md
  ```
  `tests/test_report.py`: given a hand-built `SynthesisOutput`, one supported, one partial and one unsupported claim, and two sources with tiers 1 and 3, assert the markdown has `[1]`, `⚠ partially verified`, `## Rejected claims` with the unsupported text and reason, `## Sources` with `tier 1`, and **no** `## Limited evidence` when `thin_sq_texts == []`.

  `script_happy_path` / `script_no_results` go in `tests/fakes.py`: scripted handlers keyed on the sub-question text found in the `user` prompt.
- [ ] **Step 2:** Run `uv run pytest tests/test_graph.py tests/test_report.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the synthesizer node, `render_report`, `build.py`, `runner.py` and `cli.py`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Real end-to-end check with the real keys: `uv run python -m app.cli "What are the main risks of investing in the EV battery market in 2026?"`. Expected: status `completed`; the report has sources and at least one claim; `claims_drafted > 0`; the event log contains `claim_verified`. Note any rejected claims; if there are none, try a question with conflicting coverage. Record the run id in `docs/DESIGN_NOTES.md` as the first real run.
- [ ] **Step 6:** Commit: `feat: synthesizer, report rendering, LangGraph wiring with research loop, CLI`.

---

## Phase D: API and UI (day 7)

### Task 13: Runs API, SSE stream, access gate, rate limit, moderation screen

**Files:**
- Create: `app/api/guard.py`, `app/api/runs.py`, `tests/test_api_runs.py`; modify `app/main.py` (include the router, store `Deps` on `app.state.deps`, create the semaphore)

**Interfaces:**
- Consumes: `create_run`, `run_research`, `Deps`, `LlmClient.moderate`.
- Produces:
  - `require_access(request) -> None`: if `demo_access_code` is set, requires a cookie `dd_access` equal to it (the HTML gate posts the code to `/access`, which sets the cookie); otherwise 401.
  - `POST /api/runs {"question": str}` (length 10–500) → `202 {"run_id"}`. Order: access check → moderation (flagged → `400 {"detail": "question rejected by safety screen"}`) → rate limit (runs created in the last hour ≥ `runs_per_hour` → 429) → `create_run` → `asyncio.create_task` running `run_research` inside `async with app.state.run_slots` (`Semaphore(max_concurrent_runs)`).
  - `GET /api/runs/{id}` → `{"id", "status", "error", "report_markdown" | null, "metrics" | null}`
  - `GET /api/runs/{id}/trail` → `{"run", "sub_questions", "searches", "sources", "claims": [{..., "verdict": {...}|null}], "events"}`
  - `GET /api/runs/{id}/events` → SSE (sse-starlette). Replays `Event`s with `id > Last-Event-ID` (default 0), then polls the DB every 0.5 s for new ones. Each SSE event has `id=<event.id>`, `event=<event.type>`, `data=<json>`. When the run is `completed` or `failed` and no events remain, it sends `event: done` and closes.

- [ ] **Step 1:** Write the failing tests (TestClient; `run_research` monkeypatched to a fake that logs two events and completes):
  ```python
  def test_gate_blocks_without_cookie(client_with_code): assert client_with_code.post("/api/runs", json={"question": "What is the EV market?"}).status_code == 401
  def test_create_run_returns_202(client): r = client.post("/api/runs", json={"question": "What is the EV battery market size?"}); assert r.status_code == 202 and r.json()["run_id"]
  def test_moderation_rejects(client, deps): deps.llm.flagged = True; assert client.post("/api/runs", json={"question": "something harmful here"}).status_code == 400
  def test_rate_limit(client_rate_1):
      assert client_rate_1.post("/api/runs", json={"question": "first question here"}).status_code == 202
      assert client_rate_1.post("/api/runs", json={"question": "second question here"}).status_code == 429
  def test_sse_replays_finished_run_then_done(client, finished_run_with_events):
      with client.stream("GET", f"/api/runs/{finished_run_with_events}/events") as r:
          body = "".join(r.iter_text())
      assert body.index("event: node_started") < body.index("event: done")
  def test_sse_resumes_after_last_event_id(client, finished_run_with_events, first_event_id):
      with client.stream("GET", f"/api/runs/{finished_run_with_events}/events", headers={"Last-Event-ID": str(first_event_id)}) as r:
          body = "".join(r.iter_text())
      assert f"id: {first_event_id}\n" not in body and "event: done" in body
  def test_failed_run_status_exposes_error(client, failed_run):
      assert client.get(f"/api/runs/{failed_run}").json()["status"] == "failed"
  def test_trail_shape(client, finished_run_with_events):
      t = client.get(f"/api/runs/{finished_run_with_events}/trail").json()
      assert set(t) == {"run", "sub_questions", "searches", "sources", "claims", "events"}
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_api_runs.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `guard.py` and `runs.py`, and wire up `main.py`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: runs API with SSE, access gate, rate limit, moderation screen`.

### Task 14: HTML pages (Jinja2 + HTMX)

**Files:**
- Create: `app/api/pages.py`, `app/templates/{base,gate,ask,run,report,trail,eval}.html`, `app/static/app.css`, `app/static/print.css`, `tests/test_pages.py`

**Interfaces:**
- Consumes: the Task 13 endpoints, `Report`, `Verdict`, `Source`.
- Produces routes:
  - `GET /` (ask form; if gated and there's no cookie → `gate.html`; `POST /access` sets the cookie)
  - `GET /runs/{id}` (live timeline via `hx-ext="sse"` connected to `/api/runs/{id}/events`; on `done`, HTMX loads the report)
  - `GET /runs/{id}/report` (report HTML: markdown rendered server-side; each `[n]` citation is a `<span class="cite" title="<verbatim quote>">` with a link to the source; tier badges; partial markers; the Rejected claims section; a "Save as PDF" button calling `window.print()`)
  - `GET /runs/{id}/trail` (tables plus a "Download JSON" link to the API trail)
  - `GET /eval` (Task 16 fills it in; until then it shows "No eval results yet")
- Pico.css and htmx (+ the sse extension) load from cdnjs or jsDelivr with exact pinned versions. Add `markdown-it-py` to the deps for server-side rendering.

- [ ] **Step 1:** Write the failing tests (seeded DB fixture with one completed run containing a supported, a partial and a rejected claim):
  ```python
  def test_ask_page(client): assert 'name="question"' in client.get("/").text
  def test_gate_page_when_code_set(client_with_code): assert 'name="code"' in client_with_code.get("/").text
  def test_report_shows_quote_on_hover_and_rejected(client, seeded_run):
      html = client.get(f"/runs/{seeded_run.id}/report").text
      assert 'class="cite"' in html and seeded_run.supported_quote in html
      assert "Rejected claims" in html and seeded_run.rejected_reason in html
  def test_trail_page_lists_verdicts(client, seeded_run):
      assert "unsupported" in client.get(f"/runs/{seeded_run.id}/trail").text
  def test_run_page_wires_sse(client, seeded_run):
      assert f'sse-connect="/api/runs/{seeded_run.id}/events"' in client.get(f"/runs/{seeded_run.id}").text
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_pages.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the templates and routes. `print.css` hides the navigation and buttons and expands citations as footnotes.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Use the `run` skill: `docker compose up --build`, ask a real question in the browser, and watch the live timeline, report hover quotes, trail page and print preview. Expected: the timeline updates live, and the report renders with citations and a Rejected claims section.
- [ ] **Step 6:** Commit and push to `main` (this deploys). Expected: the same flow works on the EC2 URL behind the access code.
  ```bash
  git commit -m "feat: Jinja2/HTMX UI with live timeline, report, audit trail, PDF print"
  ```

---

## Phase E: Self-evaluation (days 8–9)

### Task 15: Test set and claim mutation generator

**Files:**
- Create: `eval/__init__.py`, `eval/questions.jsonl`, `eval/mutate.py`, `tests/test_mutate.py`

**Interfaces:**
- Produces:
  - `questions.jsonl`: 24 lines `{"id": "q01", "domain": "finance|tech|science|current|comparison", "question": "..."}` (≥ 4 per domain).
  - `MutationKind = Literal["number", "year", "negation", "entity"]`
  - `mutate_claim(text: str, kind: MutationKind, rng: random.Random) -> str | None` (deterministic kinds; `None` when it doesn't apply)
  - `async def entity_swap(text: str, llm: LlmClient) -> str | None` (role `planner`; returns `None` if the output equals the input)
  - Rules: **number** changes the first non-year number by ×(1.3–2.0) or ÷ the same factor, keeping its format (`28%` → `47%`, `1.1 TWh` → `1.9 TWh`). **year** shifts the first 19xx/20xx year by ±1–3. **negation** inserts or removes `not` after the first auxiliary (`is`/`are`/`was`/`were`/`will`/`can`/`has`/`have`) or swaps `rose`↔`fell`, `increased`↔`decreased`, `higher`↔`lower`.

- [ ] **Step 1:** Write the failing tests:
  ```python
  rng = lambda: random.Random(7)
  def test_number_changes_value_keeps_unit():
      out = mutate_claim("Demand rose 28% to 1.1 TWh in 2025.", "number", rng())
      assert out != "Demand rose 28% to 1.1 TWh in 2025." and "%" in out and "2025" in out
  def test_year_shifts_only_year():
      out = mutate_claim("Demand rose 28% in 2025.", "year", rng())
      assert "28%" in out and "2025" not in out and re.search(r"20(2[2-8])", out)
  def test_negation_flips_direction(): assert mutate_claim("Battery prices fell in 2024.", "negation", rng()) == "Battery prices rose in 2024."
  def test_not_applicable_returns_none(): assert mutate_claim("Battery makers face supply risk.", "number", rng()) is None
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_mutate.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `mutate.py` and write the 24 questions (realistic and answerable from the public web, including ≥ 3 where sources are known to disagree, e.g. market-size estimates).
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat(eval): test question set and claim mutation generator`.

### Task 16: Eval runner, metrics, independent judge, hand-label set, RESULTS.md, Eval page

**Files:**
- Create: `app/prompts/judge.md`, `eval/metrics.py`, `eval/judge.py`, `eval/label.py`, `eval/run.py`, `eval/gold/claims_gold.jsonl`, `eval/RESULTS.md` (generated), `tests/test_metrics.py`; modify `app/api/pages.py` + `app/templates/eval.html`

**Interfaces:**
- Produces:
  - `metrics.py`, all pure functions:
    - `catch_rate(mutant_labels: list[str]) -> float` (share labelled `unsupported`)
    - `wrongful_rejection_rate(original_labels: list[str]) -> float` (share of originally-supported claims re-verified as `unsupported`)
    - `quote_validity_rate(verdicts: list[Verdict]) -> float` (share of verdicts with a non-empty `quote` whose `quote_found` is True)
    - `agreement(pred: list[bool], gold: list[bool]) -> tuple[float, float]` (accuracy, Cohen's kappa)
    - `summarize(values: list[float]) -> dict` (mean, p50, p95)
  - `judge.py`: `async def judge_claim(llm, claim_text: str, chunk_text: str) -> JudgeOutput` (role `judge`, prompt `judge.md`)
  - `label.py`: `python -m eval.label --runs <ids> --n 50` samples 50 claims across runs (stratified across labels) into `eval/gold/claims_gold.jsonl` as `{"claim_id", "claim", "source_text", "gold_supported": null}` for the owner to fill in.
  - `run.py`: `python -m eval.run [--limit N] [--skip-research --runs-file eval/results/runs.json]`. Steps:
    1. Run research for each question (or reuse the stored run ids).
    2. Collect per-run metrics, cost/latency (from `llm_call` events) and loop counts.
    3. Re-verify each supported claim once more as the "original" control.
    4. Create one mutant per supported claim (cycling kinds; entity via LLM), verify it against the same chunk, and compute the catch rate per kind.
    5. Have the judge check up to 100 sampled final-report claims (independent groundedness = share judged supported).
    6. If gold labels are filled in, compute agreement.
    7. Write `eval/results/<UTC timestamp>.json` and regenerate `eval/RESULTS.md`: a table of every metric, the model configuration used, the date, the question count, and a "How to reproduce" section.
  - `/eval` page: reads the latest results JSON and shows stat tiles plus one bar chart of catch rate by mutation kind (follow the `dataviz` skill).

- [ ] **Step 1:** Write the failing tests:
  ```python
  def test_catch_rate(): assert catch_rate(["unsupported", "unsupported", "supported", "partial"]) == 0.5
  def test_wrongful_rejection(): assert wrongful_rejection_rate(["supported", "unsupported", "supported", "supported"]) == 0.25
  def test_agreement_perfect(): assert agreement([True, False, True], [True, False, True]) == (1.0, 1.0)
  def test_agreement_kappa_known_value():
      acc, kappa = agreement([True, True, False, False], [True, False, False, False])
      assert acc == 0.75 and round(kappa, 3) == 0.5
  def test_summarize(): assert summarize([1.0, 2.0, 3.0, 4.0])["p50"] == 2.5
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_metrics.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `metrics.py`, `judge.py`, `label.py`, `run.py` and the Eval page.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Smoke run: `uv run python -m eval.run --limit 2`. Expected: a results JSON and `RESULTS.md` written, every metric field populated (gold agreement shows "not labelled yet").
- [ ] **Step 6:** Full run: `uv run python -m eval.run`. Expected: 24 runs finished (failed runs are counted and reported, not hidden), and a total cost printed (expected ≲ $2).
- [ ] **Step 7 (owner, about 1 hour):** `uv run python -m eval.label --runs-file eval/results/runs.json --n 50`, then fill in `gold_supported` (true/false) for all 50 lines by reading `source_text`.
- [ ] **Step 8:** `uv run python -m eval.run --skip-research --runs-file eval/results/runs.json`. Expected: RESULTS.md now includes accuracy and kappa against the hand labels.
- [ ] **Step 9:** Commit the results, gold set, RESULTS.md and code: `feat(eval): metrics, independent judge, gold-label agreement, results`.

---

## Phase F: Bonuses (day 10)

### Task 17: Contradiction detection

**Files:**
- Create: `app/prompts/contradictions.md`, `app/graph/nodes/contradictions.py`, `tests/test_contradictions.py`; modify `app/graph/build.py` (insert the node: `route_after_verify` returns `"contradictions"` instead of `"synthesize"`, then `contradictions → synthesize`), `app/graph/nodes/synthesizer.py` (pass conflicts to `render_report`), `app/report.py` (a `## Conflicting sources` section, rendered only when there are conflicts)

**Interfaces:**
- Produces: `async def contradictions_node(state, config) -> dict`. Per sub-question with ≥ 2 supported claims, it calls role `contradictions` with those claims (ids, text and source domain) → `ContradictionOutput`. It keeps only conflicts whose ids belong to that sub-question's supported set, stores them as event `conflict_found` (payload = the Conflict), and returns `{}`. The synthesizer reads the conflicts from events.

- [ ] **Step 1:** Write the failing tests:
  ```python
  async def test_conflict_flagged_in_report(deps):
      script_happy_path(deps, conflict_between=("A", 0, 1))   # claims 0 and 1 of sub-question A conflict
      run = create_run(session_of(deps), "q", deps.settings); await run_research(run.id, deps)
      md = report_of(deps, run.id).markdown
      assert "## Conflicting sources" in md and "conflict_found" in event_types(deps, run.id)
  async def test_conflict_with_foreign_ids_ignored(deps):
      script_happy_path(deps, conflict_ids=(9999, 1))
      run = create_run(session_of(deps), "q", deps.settings); await run_research(run.id, deps)
      assert "## Conflicting sources" not in report_of(deps, run.id).markdown
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_contradictions.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement and rewire. Also update the existing `test_graph.py` routing assertions if they name the next node.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Re-run the eval (`uv run python -m eval.run`) so RESULTS.md reflects the final graph. Commit: `feat: cross-source contradiction detection`.

---

## Phase G: Finish (day 11; day 12 is buffer only)

### Task 18: Docs, reviews, going public, submission

**Files:**
- Create/modify: `README.md`, `docs/DESIGN_NOTES.md`, `eval/RESULTS.md` (already generated)

- [ ] **Step 1:** Write `README.md`:
  - what it is (one paragraph, the trust problem)
  - an architecture diagram (mermaid) of the agent flow
  - how the agents cooperate (one line per agent)
  - the verification design (the quote check in code, the loop rule)
  - a self-eval headline numbers table, linking `eval/RESULTS.md`
  - how to run: `uv sync`, `.env`, `uv run uvicorn …`, `docker compose up`, `python -m app.cli`, `python -m eval.run`
  - deployment (link `docs/DEPLOY.md`)
  - the live URL (if deployed)
  - the Mesh model table
  - known limitations (be honest)
- [ ] **Step 2:** Write `docs/DESIGN_NOTES.md`, the "key design decisions" note for the submission: the Drafter/Synthesizer split, the cross-family verifier, the quote check in code, why the eval uses a third-model judge and hand labels, SQLite/Chroma/Tavily choices, the Python 3.11 constraint, and any model swaps made in Task 3.
- [ ] **Step 3:** Run the `code-review` and `simplify` skills on the whole repo; fix what they find; `uv run pytest -q`. Expected: all pass.
- [ ] **Step 4:** Run the `security-review` skill. Then scan the full history for secrets: `git log -p | grep -nE "rsk_[A-Za-z0-9]{8,}|tvly-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16}"`. Expected: no matches. If anything appears, **rotate the key** and rewrite history before going public.
- [ ] **Step 5:** Make the repo public (GitHub → Settings → Danger Zone), **no later than 29 Oct**. Push, and check that Agent Arena Checks shows all critical checks green on the latest commit.
- [ ] **Step 6:** Final deploy via push to `main`. On the live URL: run one question end to end, and open the report, trail and eval pages.
- [ ] **Step 7 (optional, owner):** Record a 2–3 minute demo video: ask a question, show the live timeline with a loop and a rejected claim, hover a citation, open the trail, then the eval page.
- [ ] **Step 8 (owner):** Submit on the hackathon dashboard **by midday 30 Oct IST** with the repo URL (plus the video and live URL if available).
