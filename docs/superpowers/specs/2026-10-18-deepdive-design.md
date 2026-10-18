# DeepDive Research Agent: Design

- **Date:** 2026-10-18
- **Hackathon:** Agent Arena 2026 (Krish Naik), Track 01 "DeepDive: Build a Self-Verifying Research Analyst"
- **Builder:** solo (knows RAG and LangGraph from courses, first hands-on project)
- **Repo:** `deepdive-research-agent` (public GitHub)

Items marked **[my call]** were proposed by Claude during design and not explicitly chosen by the owner. They can be changed freely.

---

## 1. Intent: why this exists

Research tools that cite sources already exist. Their known weakness is **trust**: the cited source often doesn't support the sentence attached to it. DeepDive's value is **not** gathering information. It's guaranteeing that every claim in the final report is true to the source it cites, and showing the evidence for that.

The owner's goals for the project, in priority order:
1. **Learn** real agentic engineering: LangGraph orchestration, RAG, LLM verification, evaluation, Docker/AWS deployment.
2. **Win** (₹20k / ₹10k / ₹5k). Submissions are screened by an automated AI code reader, then by human judges.
3. **Resume:** something worth explaining in an interview.

### Judging criteria (from the hackathon page), and where this design covers each

| # | Criterion | Covered by |
|---|---|---|
| 1 | Multi-agent orchestration | §4: LangGraph graph with 5 distinct agents and parallel researchers |
| 2 | Core loop (system checks/improves its own work) | §5: verifier, quote check in code, loop back to research |
| 3 | Core domain capability | §4–5: grounded, cited research reports |
| 4 | Supporting domain capability | §5.6–5.7: source-quality tiers, contradiction detection, PDF export |
| 5 | Foundation, data model, real ingestion | §6: SQLModel schema, Tavily, fetching, chunking, embeddings |
| 6 | Self-evaluation (test set and measured numbers) | §7: planted false claims, independent judge, hand-labelled gold set |
| 7 | Auditable decision trail | §6 `events`/`verdicts` tables, §8 audit trail page |
| 8 | Code quality and reproducibility | §10: uv lockfile, ruff, pytest, Docker, CI |

## 2. Goals, non-goals, anti-goals

**Goals**
- A user asks a research question and gets a structured report in which **every claim is cited and verified** against the exact source text.
- Unsupported claims are visibly **rejected, with reasons**.
- Thin sub-questions trigger **more research automatically** (a real loop that runs).
- Every step of every run is **recorded and viewable**.
- **Honest, reproducible metrics** committed to the repo.
- Deployed on AWS (EC2) through an automated GitHub → ECR → EC2 pipeline.

**Non-goals (out of scope unless time is left; see §12 stretch)**
- Multi-user accounts and authentication (a single demo access code is enough).
- User document upload (the brief marks it optional).
- Conversational follow-up chat (a brief bonus, treated as stretch).
- Horizontal scaling or a job queue (one instance, in-process background tasks).

**Anti-goals (failure even if it "works")**
- A report whose citations *look* right but don't support the sentence: the exact problem we exist to solve.
- Anything that looks faked to the judges: hardcoded outputs, an agent that's never called, a loop that never loops, a test set or metrics that don't exist. The rules say these "will score poorly".
- Any LLM or embedding call that does **not** go through Mesh (disqualifying).
- Secrets committed to git or baked into the Docker image.
- The public URL draining Mesh or Tavily credits.
- Infrastructure work eating the time needed for the verifier and eval.

## 3. Constraints

| Constraint | Source |
|---|---|
| Backend must be Python with **Flask or FastAPI** | Rules |
| **Every LLM/AI call through Mesh API** (`https://api.meshapi.ai/v1`, OpenAI-compatible, key `MESH_API_KEY`, prefix `rsk_`) | Rules, critical check |
| Public GitHub repo; README covering what was built, how the agents cooperate, design decisions, self-eval results | Rules |
| `requirements.txt` or `pyproject.toml` listing a web framework and an LLM client | Critical check |
| **All `.py` files must compile on Python 3.11** (the checks workflow uses `python-version: '3.11'`) | `agent-arena-2026-checks.yml` |
| `.gitignore` includes `.env`; no committed secrets | Advisory check, rules |
| Checks workflow committed **unchanged** at `.github/workflows/agent-arena-2026-checks.yml`, with GitHub secrets `MESH_API_KEY` and `SUBMISSION_TOKEN` | Hackathon setup |
| Registration closes **24 Oct 2026, 23:00 IST**; submissions close **30 Oct 2026, 23:00 IST**; results 3 Nov | Timeline |
| Solo builder, about 12 days | Owner |

## 4. Architecture

### 4.1 Stack (decided by owner)

| Layer | Choice |
|---|---|
| Python | **3.11** everywhere (local, Docker, CI) to match the judges' compile check |
| Packages | **uv** (`pyproject.toml` + `uv.lock`) plus an exported `requirements.txt` |
| Web framework | **FastAPI** |
| Orchestration | **LangGraph** |
| LLM gateway | **Mesh API** through the official `openai` Python SDK |
| Database | **SQLite** through **SQLModel** |
| Vector store | **Chroma**, embedded/persistent mode; **embeddings computed through Mesh** |
| Web search | **Tavily** (raw results only, **never** its LLM "answer" feature) |
| UI | **Jinja2 + HTMX** (SSE extension) served by FastAPI; Pico.css [my call] |
| Runtime | **Docker** (one app container) + docker-compose; deployed to **AWS EC2** |

### 4.2 Agent flow (approved by owner)

```
User question
  → Planner            : splits into 3–6 sub-questions + search queries
  → Researcher × N      : (parallel, LangGraph Send) search → fetch → chunk → embed → store evidence
  → Drafter             : writes ATOMIC claims as JSON, each citing evidence chunk id(s)
  → Verifier            : judges each claim against ONLY its cited chunk → label + exact quote + reason
  → quote check (code)  : quote must exist in the chunk, else claim → unsupported
  → "Enough support?"   : rule in code, per sub-question
        ├─ no  → back to Researcher for thin sub-questions (refined query + failure context), max 2 extra rounds
        └─ yes → Synthesizer : readable report using ONLY verified claims
  → every step logged to SQLite (events) → audit trail
```

- **Coordinator** = the LangGraph `StateGraph`. It holds shared state and routes between nodes.
- **Five distinct agents**, each with its own prompt file and its own input/output schema: Planner, Researcher, Drafter, Verifier, Synthesizer. Researcher is the only one with tools (search, fetch).
- **Drafter and Synthesizer are deliberately separate** (owner agreed). The Synthesizer only sees verified claims, so it can't introduce unverified facts, and the split gives a clean draft-vs-final metric.
- **Verifier uses a different model family from the Drafter** [my call], e.g. the Drafter on an OpenAI model and the Verifier on a Claude or Gemini model, both through Mesh. This avoids a model approving its own output.
- **Loop cap:** 2 extra research rounds (configurable). The cap and each loop are recorded in events, so judges can see the loop really run.

### 4.3 Graph state (sketch)

`run_id, question, sub_questions[{id, text, query, round, status}], evidence_ids_by_subq, claims[], verdicts[], round, report_md`

Large data (chunk text, page text) lives in SQLite/Chroma. The state carries IDs only.

### 4.4 Mesh client (`app/llm.py`)

- One module wraps all model calls. The base URL `https://api.meshapi.ai/v1` appears literally in config, since the automated check looks for it.
- Per-agent model names come from env (`MODEL_PLANNER`, `MODEL_DRAFTER`, `MODEL_VERIFIER`, `MODEL_JUDGE`, `MODEL_EMBED`, …).

### 4.5 Model choice and budget

The only **free** model on Mesh (checked 2026-10-18 on meshapi.ai/models, Plan = Free) is `openai/omni-moderation-latest`. It is a **moderation classifier** and can't generate text, so it can't run any agent. All agent and embedding models are therefore paid, but the low-cost tier is very cheap.

Starting lineup [my call; confirm IDs and endpoint support on day 1 with the real key]. Prices are per 1M input/output tokens from the Mesh model list:

| Role | Model (Mesh ID) | Price in / out | Why |
|---|---|---|---|
| Planner, Researcher extraction, Drafter | `openai/gpt-5-nano` | $0.05 / $0.40 | Cheapest capable model in the OpenAI family |
| Verifier | `anthropic/claude-haiku-5.5` | $0.09 / $0.45 | **Different family** from the Drafter (§4.2) |
| Synthesizer | `openai/gpt-5-nano` (upgrade if prose quality is weak) | $0.05 / $0.40 | Only rewrites verified claims |
| Independent eval judge | `google/gemini-2.5-flash-lite` | $0.10 / $0.40 | A **third** family, independent of the verifier |
| Embeddings | `google/embeddinggemma-300m` (fallback `amazon/titan-embed-text-v2`) | $0.002 / – | Cheap; must confirm it works on the `/v1/embeddings` endpoint |
| Input safety screen (optional) | `openai/omni-moderation-latest` | Free | The owner's free model, used for what it does: screening user questions before a run |

**Budget estimate (rough):** one research run ≈ 40 calls ≈ 120k input + 20k output tokens ≈ **$0.01–0.03**. A full eval (25 questions, plus mutation tests and the judge) ≈ **$1–2**. A **$5 top-up is expected to cover the whole hackathon**, development included. Track actual spend through the per-call token logging.

**Rule:** the published `eval/RESULTS.md` numbers must come from the **same model configuration that's deployed**. If models are upgraded for the deployed demo, the eval is re-run on that configuration so the numbers describe the real system.
- **Structured output:** request a JSON schema/JSON mode, validate with Pydantic, and do **one repair retry** on validation failure. Not every model behind Mesh supports strict structured outputs.
- Retries with backoff on 429/5xx; every call logs model, tokens and latency (feeds the cost metric).

## 5. Verification design (the core)

1. **Atomic claims.** The Drafter returns `[{claim_id, sub_question_id, text, chunk_ids[1..2]}]`. One fact per claim.
2. **Isolated judging.** The Verifier gets the claim and the cited chunk text only. It returns `{label: supported|partial|unsupported, quote, reason}`.
3. **Quote check in code** (`app/tools/quote_match.py`): normalize whitespace, quotes and case, then do an exact substring match, then a fuzzy match (e.g. `rapidfuzz.partial_ratio ≥ 90`; threshold chosen by builder and unit-tested). Quote not found → claim becomes `unsupported`, with reason `"verifier quote not found in source"`.
4. **What each label means for the report:**
   - `supported` → in the report
   - `partial` → in the report, with a "partially verified" marker
   - `unsupported` → removed, listed in the report's **Rejected claims** section with its reason
5. **Thinness rule** (`app/graph/rules.py`, plain code): a sub-question with fewer than `MIN_SUPPORTED_CLAIMS = 2` supported claims → re-research (if rounds remain) with a refined query and the list of rejected claims. If still thin after the cap, the report says the sub-question has limited evidence instead of padding it.
6. **Source quality tiers** (bonus): readable rules by domain/type. Tier 1 = official, `.gov`/`.edu`, filings, journals; tier 2 = major news; tier 3 = blogs and forums. Used to rank chunks for the Drafter and shown as badges.
7. **Contradiction detection** (bonus): among *supported* claims within a sub-question, an LLM check for conflicting pairs (same entity or metric, different value). Conflicts are flagged in the report, not silently resolved.

## 6. Data

### 6.1 Ingestion
- Tavily search per sub-question query (≈5 results), raw content requested; fall back to `httpx` + `trafilatura` when the raw content is missing.
- Chunking ≈ 800 tokens with ≈ 100 overlap [builder may tune]. Each chunk is stored with its source ID and position.
- Embeddings through **Mesh** (the homepage lists embedding models, e.g. Mistral Embed; exact model ID to confirm at setup). Stored in Chroma with `run_id`/`sub_question_id` metadata; top-k retrieval per sub-question, then tier-aware ordering.
- **Caching** [my call]: search and fetch results are cached (keyed by query/URL). This makes eval reruns reproducible and cheaper, and protects the Tavily quota.

### 6.2 Tables (SQLModel)

| Table | Key fields |
|---|---|
| `runs` | id, question, status, models used (json), started/finished, token/cost totals |
| `sub_questions` | id, run_id, text, query, round, status |
| `searches` | id, sub_question_id, query, round, provider, n_results |
| `sources` | id, url, domain, title, tier, fetched_at |
| `evidence_chunks` | id, source_id, run_id, sub_question_id, text, position |
| `claims` | id, run_id, sub_question_id, text, cited chunk ids, round |
| `verdicts` | id, claim_id, label, quote, quote_found (bool), reason, verifier model |
| `events` | id, run_id, node, type, ts, payload (json): drives SSE and the audit trail |
| `reports` | run_id, markdown, metrics (json) |

Persistence: SQLite file and Chroma directory under `/app/data`, mounted as a Docker volume.

## 7. Self-evaluation

- **Test set:** 20–25 questions in `eval/questions.jsonl` covering finance, tech, science, current events and comparisons.
- **Metrics** (all produced by `uv run python -m eval.run`):

| Metric | How |
|---|---|
| Verifier catch rate | Take supported claims and generate altered versions (number, date, entity swap, negation) via `eval/mutate.py` with the same citation; % flagged unsupported |
| Wrongful rejection rate | % of real supported claims the verifier rejects (guards against "reject everything") |
| Independent groundedness | A **separate judge model** (not the verifier) re-checks a sample of final-report claims. Avoids grading our own homework |
| Quote validity rate | % of verifier quotes found in the source |
| Draft → final | Unsupported claim share in the draft vs the final report |
| Coverage and loops | % of sub-questions sufficiently supported; loop trigger rate |
| Cost and latency | Tokens, estimated cost and seconds per run |
| **Agreement with hand labels** | Owner hand-labels **~50 claims** (owner committed ~1 hour) in `eval/gold/claims_gold.jsonl`; report the verifier's accuracy (and Cohen's kappa) against them |

- **Outputs:** `eval/results/<timestamp>.json` plus a human-readable `eval/RESULTS.md` (committed), and the Eval page in the UI.
- **Unit tests** (`pytest`, LLM mocked): quote matching, thinness rule, tier rules, mutation generator, schema validation.

## 8. API and UI

**Endpoints [my call on exact shape]**
- `POST /api/runs` `{question}` → `{run_id}` (starts a background task)
- `GET /api/runs/{id}/events`: SSE stream of graph events
- `GET /api/runs/{id}`: status and report
- `GET /api/runs/{id}/trail`: full audit JSON
- `GET /api/eval/latest`
- `GET /healthz`
- FastAPI's automatic `/docs` stays enabled.

**Pages (Jinja2 + HTMX)**
1. **Ask:** question box, then a live timeline of agent steps (Planner → sub-questions, Researcher N found X sources, Verifier rejected claim #k because…, looping back on sub-question 2…).
2. **Report:** report with citations; hovering shows the exact supporting quote. Tier badges, contradiction flags, Rejected claims section.
3. **Audit trail:** full run record; JSON download.
4. **Eval:** metric charts from `eval/results/`.
5. **PDF export:** print stylesheet plus browser "Save as PDF".

**Protecting the public URL [my call]:** `DEMO_ACCESS_CODE` (cookie after entry), `RUNS_PER_HOUR` limit counted from `runs`, and at most 2 concurrent runs (semaphore).

## 9. Deployment (owner wants Docker on EC2)

- **Local = prod:** the same `Dockerfile` (multi-stage, `python:3.11-slim`, uv) and `docker-compose.yml` with a `./data:/app/data` volume.
- **Pipeline (`.github/workflows/deploy.yml`)**, separate from the hackathon checks workflow: push to `main` → ruff + pytest → build image → push to **ECR** tagged with the commit SHA → deploy to EC2 through **AWS SSM Run Command** (`docker compose pull && up -d`). Plain SSH is the fallback.
- **AWS auth from GitHub:** **OIDC** role (no long-lived AWS keys in GitHub).
- **EC2:** t3.small, **Elastic IP**, region **ap-south-1 (Mumbai)** [my call]. Optional **Caddy** reverse proxy for automatic HTTPS through a `nip.io`-style hostname.
- **Secrets on the instance:** `.env` on the host or SSM Parameter Store; never in the image or the repo.
- **AWS account is new:** the owner creates it personally, then immediately sets a **$10 budget alert** and root MFA, and uses an IAM identity for daily work.
- **Timing:** a "hello world" deploy on day 2–3, then time-boxed. Deployment isn't a judging criterion; a deployed URL is only reviewed for shortlisted teams.

### 9.1 Repo visibility (owner decided)

**Private while building; public once the first working version is verified.** The hackathon checks run on private repos too (GitHub Actions free minutes are plenty). Before switching to public:
- Run the `security-review` skill and scan the **full git history** for secrets. A key committed once and deleted later is still visible in history once the repo is public. If one is found, rotate the key; don't just delete the file.
- Switch to public **no later than 28–29 Oct**, so judges can read it and the submission link works well before the deadline.

## 10. Repo layout and tooling

```
deepdive-research-agent/
├─ app/
│  ├─ main.py            # FastAPI app
│  ├─ config.py          # pydantic-settings; Mesh base URL lives here
│  ├─ llm.py             # the only place that calls Mesh
│  ├─ db.py, models.py   # SQLModel engine + tables
│  ├─ graph/             # state.py, build.py, rules.py, nodes/{planner,researcher,drafter,verifier,synthesizer}.py
│  ├─ tools/             # search.py, fetch.py, chunking.py, vectorstore.py, quote_match.py, tiers.py
│  ├─ prompts/           # one prompt file per agent
│  ├─ api/               # runs.py, pages.py, sse.py
│  ├─ templates/, static/
├─ eval/                 # questions.jsonl, mutate.py, judge.py, run.py, gold/, results/, RESULTS.md
├─ tests/
├─ docs/                 # superpowers/specs/, DESIGN_NOTES.md (submission "key design decisions")
├─ deploy/               # Caddyfile, ec2-bootstrap.sh
├─ .github/workflows/    # agent-arena-2026-checks.yml (unchanged), ci.yml, deploy.yml
├─ Dockerfile, docker-compose.yml, .dockerignore
├─ pyproject.toml, uv.lock, requirements.txt (uv export)
├─ .env.example, .gitignore, .pre-commit-config.yaml
├─ README.md, CLAUDE.md
```

- **ruff** with `target-version = "py311"`; **pre-commit** runs ruff and `python -m compileall` (same as the judges' check).
- **CLAUDE.md** (via the `init` skill) records the hard rules: Python 3.11 syntax only, all model calls through `app/llm.py` → Mesh, never commit `.env`, no stubbed features.
- A Claude Code hook (via `update-config`) runs the compile and lint check after edits.
- Skills planned for the build: `writing-plans` → `executing-plans`/`subagent-driven-development`, `test-driven-development`, `systematic-debugging`, `dataviz` (Eval page), `code-review` + `simplify` + `security-review` before going public, `verification-before-completion`.

## 11. Timeline (today is Sat 18 Oct; deadline Thu 30 Oct 23:00 IST)

| Day | Date | Work |
|---|---|---|
| 1 | 18–19 Oct | Approve design, write plan. **Register for the hackathon.** Create accounts: AWS (budget alert, MFA), Mesh key, Tavily key, GitHub repo. Scaffold the repo; hackathon checks green |
| 2 | 20 Oct | Docker + compose; AWS OIDC, ECR, EC2, SSM; "hello world" deployed through the pipeline |
| 3–4 | 21–22 Oct | Models/DB, Mesh client, Planner, Researcher (search, fetch, chunk, embed, cache), event logging; CLI run through research |
| 5–6 | 23–24 Oct | Drafter, Verifier + quote check, thinness loop, Synthesizer; full pipeline from the CLI |
| 7 | 25 Oct | FastAPI endpoints, SSE, UI pages, access code and rate limit |
| 8–9 | 26–27 Oct | Eval: question set, mutation generator, independent judge, **hand-label ~50 claims**, RESULTS.md, Eval page |
| 10 | 28 Oct | Bonuses: tiers, contradiction detection, PDF export |
| 11 | 29 Oct | Hardening, README, DESIGN_NOTES, code/security review, final deploy, optional demo video. **Repo goes public** once checks pass (§9.1) |
| 12 | 30 Oct | **Buffer only.** Submit by midday, not at 23:00 |

## 12. Stretch (only after §11 day 10 is done)
User document upload (PDF) into the same evidence store; conversational follow-up that reuses a run's verified claims; source-quality weighting shown in the report's methodology section.

## 13. Risks

| Risk | Mitigation |
|---|---|
| A Mesh model lacks strict JSON output | JSON mode + Pydantic validation + repair retry; pick models that behave well during setup |
| Mesh embeddings not available or named differently | Confirm on day 1; fallback = BM25 keyword retrieval (no AI call, so no rule issue) |
| Mesh credits run out during eval | Caching; cheap model lineup (§4.5, ~$5 expected total); per-call token logging to watch spend |
| A cheap model is too weak (bad JSON, poor verification) | Swap that one role to a stronger model via env; the verifier is the first to upgrade |
| Tavily free quota | Caching; DuckDuckGo + trafilatura fallback |
| AWS setup eats time | Hello-world deploy early and time-boxed; deployment is optional for scoring |
| Python 3.12-only syntax slips in | ruff `py311` target + pre-commit compile check |

## 14. Left to the builder
Exact prompts; chunk size and top-k; fuzzy-match threshold (unit-tested); exact Mesh model IDs per agent (chosen on day 1 from Mesh's model list); HTML/CSS details; test structure.

## 15. Owner answers (2026-10-18)
- **Credits:** the owner hoped to use a free Mesh model for development and pay only for the deployed version. The only free model is moderation-only (§4.5), so development uses the cheap paid lineup instead (a ~$5 top-up). The free moderation model is kept as an optional input-safety screen.
- **Repo visibility:** private first, public after the first working version is verified (§9.1).
