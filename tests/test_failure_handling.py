"""Failure handling: one failing step or service must not sink the run or the eval."""

import asyncio
import random
import re
import time

import httpx
import openai
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.graph.runner import create_run, run_research
from app.llm import LlmOutputError, MeshLlm
from app.main import create_app
from tests.fakes import FakeOpenAI
from tests.helpers import claims_of, event_types, events, get_run, report_of, sign_in
from tests.scripts import script_happy_path

_REQ = httpx.Request("POST", "https://api.meshapi.ai/v1/chat/completions")


def _conn_error():
    return openai.APIConnectionError(request=_REQ)


class Out(BaseModel):
    answer: str


# --- a failing sub-question must not leave siblings running after run_failed ------------------


class _SlowSiblings:
    """Drafter for 'Sub-question A' fails at once; B and C would take a while."""

    def __init__(self, inner, error):
        self.inner, self.error = inner, error

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def chat_json(self, *, role, user, **kw):
        if role == "drafter":
            if "Sub-question A" in user:
                raise self.error
            await asyncio.sleep(0.3)
        return await self.inner.chat_json(role=role, user=user, **kw)


async def test_drafter_output_error_affects_only_its_subquestion(deps):
    script_happy_path(deps)
    deps.llm = _SlowSiblings(deps.llm, LlmOutputError("bad json twice"))
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    assert get_run(deps, run.id).status == "completed"
    assert "draft_failed" in event_types(deps, run.id)
    assert "## Limited evidence" in report_of(deps, run.id).markdown


async def test_transport_error_cancels_sibling_drafts_and_reports_clearly(deps):
    script_happy_path(deps)
    deps.llm = _SlowSiblings(deps.llm, _conn_error())
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    r = get_run(deps, run.id)
    assert r.status == "failed" and "APIConnectionError" in r.error and "TaskGroup" not in r.error
    await asyncio.sleep(0.5)  # siblings would have finished by now if still running
    assert claims_of(deps, run.id) == []
    failed_at = [e.id for e in events(deps, run.id) if e.type == "run_failed"][0]
    assert all(e.id < failed_at for e in events(deps, run.id) if e.type == "claims_drafted")


# --- contradiction check never fails a finished run -------------------------------------------


async def test_contradiction_transport_error_does_not_fail_run(deps):
    script_happy_path(deps)
    deps.llm.handlers["contradictions"] = lambda s, u: _conn_error()
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    assert get_run(deps, run.id).status == "completed"
    assert "contradiction_check_failed" in event_types(deps, run.id)


# --- responsive run creation, sane retry budgets ----------------------------------------------


def test_slow_moderation_does_not_block_run_creation(deps, monkeypatch):
    async def _noop_research(run_id, deps):
        return None

    async def slow_moderate(text):
        await asyncio.sleep(3)
        return False

    monkeypatch.setattr("app.api.runs.run_research", _noop_research)
    deps.settings.moderation_timeout_s = 0.2
    deps.llm.moderate = slow_moderate
    deps.settings.google_client_id, deps.settings.google_client_secret = "cid", "s"
    client = TestClient(create_app(deps.settings, deps=deps))
    sign_in(client)
    started = time.perf_counter()
    r = client.post("/api/runs", json={"question": "What is the EV battery market size in 2026?"})
    assert r.status_code == 202 and time.perf_counter() - started < 2


async def test_timeouts_get_a_smaller_retry_budget_than_429s(settings):
    timeout = openai.APITimeoutError(request=_REQ)
    fake = FakeOpenAI(chat=[timeout] * 4 + ['{"answer": "late"}'])
    llm = MeshLlm(settings, client=fake, retry_base_delay=0, rate_limit_wait=0)
    try:
        await llm.chat_json(role="planner", system="s", user="u", schema=Out)
        raise AssertionError("expected the timeout to propagate")
    except openai.APITimeoutError:
        pass
    assert len(fake.chat_calls) == settings.llm_timeout_retries + 1


async def test_rate_window_is_consumed_only_when_a_slot_is_held(settings):
    llm = MeshLlm(settings, client=FakeOpenAI(chat=['{"answer": "x"}']))
    seen = []
    original = llm._limiter.acquire

    async def spy():
        seen.append(llm._slots._value < settings.llm_concurrency)
        await original()

    llm._limiter.acquire = spy
    await llm.chat_json(role="planner", system="s", user="u", schema=Out)
    assert seen == [True]


# --- very long "words" can't explode the chunk count ------------------------------------------


def test_long_words_do_not_collapse_overlap():
    from app.tools.chunking import chunk_text

    text = " ".join("w" * 39 + str(i % 10) for i in range(2000))  # 40-char words
    chunks = chunk_text(text, max_words=600, overlap_words=75, max_chars=3000)
    assert len(chunks) < 60
    assert chunks[-1].split()[-1] == text.split()[-1]


# --- eval robustness and wording --------------------------------------------------------------


async def test_eval_tolerates_transport_errors_from_judge_and_entity_swap(deps, tmp_path):
    from eval.run import evaluate, research_questions

    script_happy_path(deps)
    run_ids = await research_questions(deps, [{"id": "q01", "question": "q"}])
    deps.llm.handlers["judge"] = lambda s, u: _conn_error()
    original_planner = deps.llm.handlers["planner"]

    def planner(system, user):
        if system.startswith("Rewrite the sentence"):
            return _conn_error()
        return original_planner(system, user)

    deps.llm.handlers["planner"] = planner
    results = await evaluate(
        deps, run_ids, random.Random(1), kinds=("entity",), gold_path=tmp_path / "x"
    )
    assert results["independent_judge"]["errors"] > 0


async def test_probe_does_not_hide_programming_errors(deps):
    from eval.run import verifier_probe

    script_happy_path(deps)
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    deps.llm.handlers["verifier"] = lambda s, u: TypeError("bug in our code")
    try:
        await verifier_probe(deps, [run.id], random.Random(1), kinds=())
        raise AssertionError("programming errors must propagate")
    except TypeError:
        pass


def test_results_page_wording_reflects_labels():
    from eval.run import render_results_md

    base = {
        "generated_at": "t",
        "n_questions": 1,
        "models": {},
        "runs": {"total": 1, "completed": 1, "failed": 0, "failed_ids": []},
        "verifier_probe": {
            "catch_rate": 1.0,
            "n_mutants": 1,
            "catch_rate_by_kind": {},
            "n_mutants_by_kind": {},
            "wrongful_rejection_rate": 0.0,
            "n_originals": 1,
            "errors": 0,
        },
        "independent_judge": {"model": "m", "n": 1, "groundedness": 1.0, "errors": 0},
        "quote_validity_rate": 1.0,
        "claims": {
            "supported": 1,
            "partial": 0,
            "unsupported": 0,
            "drafted": 1,
            "draft_rejected_rate": 0.0,
            "report_paragraphs_dropped": 0,
        },
        "loops": {"runs_with_loop_rate": 0.0, "mean_extra_rounds": 0.0},
        "coverage": {"ok_rate": 1.0, "sub_questions": 1},
        "cost": {"per_run_usd": {"mean": 0.01, "p95": 0.01}},
        "latency_s": {"p50": 1, "p95": 1},
    }
    labelled = render_results_md(
        {**base, "gold": {"n_labelled": 50, "accuracy": 0.86, "kappa": 0.7}}
    )
    unlabelled = render_results_md(
        {**base, "gold": {"n_labelled": 0, "accuracy": None, "kappa": None}}
    )
    assert "n/a until" not in labelled and "n/a until" in unlabelled


# --- colour highlighting must not corrupt its own escape codes --------------------------------


def test_colour_highlight_keeps_text_intact():
    from eval.annotate import highlight

    claim = "Sales rose 12.5% to 1 billion units in 2023 across 33 markets."
    source = "In 2023 sales rose 12.5% to 1 billion units in 33 markets, analysts said."
    out = highlight(claim, source, color=True)
    assert "\x1b[\x1b" not in out and re.sub(r"\x1b\[[0-9;]*m", "", out) == source
