"""Regression tests for findings from the whole-branch review (one test per finding)."""

import random
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from app.deps import SearchResult
from app.graph.nodes.researcher import research_node
from app.graph.runner import create_run, run_research
from app.graph.schemas import JudgeOutput, Paragraph, Section, SynthesisOutput, VerdictOutput
from app.llm import LlmOutputError, MeshLlm
from app.main import create_app
from app.models import Run
from app.tools.fetch import HttpFetcher
from app.tools.quote_match import quote_in_chunk
from tests.fakes import FakeOpenAI
from tests.helpers import cfg, chunk_count, claims_of, event_types, events, get_run, report_of
from tests.scripts import _drafter, script_happy_path

# --- I1: re-drafted duplicates must not satisfy the loop rule ---------------------------


async def test_redrafted_duplicate_claims_do_not_satisfy_loop(deps):
    script_happy_path(deps)

    def drafter_without_round(system, user):
        out = _drafter(system, user)
        for c in out.claims:
            c.text = re.sub(r" round \d", "", c.text)  # same facts every round
        return out

    def verifier(system, user):
        letter, i = re.search(r"Claim:\n(\w) fact (\d)", user).groups()
        if letter == "B" and i == "2":
            return VerdictOutput(label="unsupported", quote="", reason="absent")
        return VerdictOutput(label="supported", quote=f"Evidence for {letter} round", reason="ok")

    deps.llm.handlers.update(drafter=drafter_without_round, verifier=verifier)
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    looping = [e.payload["looping"] for e in events(deps, run.id) if e.type == "loop_decision"]
    assert looping == [True, True, False]
    b_claims = [c for c in claims_of(deps, run.id) if c.text.startswith("B ")]
    assert len(b_claims) == 2 and "claim_dropped_duplicate" in event_types(deps, run.id)


# --- I2: synthesizer prose is checked (numbers must come from cited claims) ------------


async def test_paragraph_with_number_not_in_cited_claims_is_dropped(deps):
    script_happy_path(deps)

    def synth(system, user):
        ids = [int(x) for x in re.findall(r"\[claim:(\d+)\]", user)]
        return SynthesisOutput(
            title="T",
            summary="Demand will grow 99% by 2031.",
            sections=[
                Section(
                    heading="H",
                    paragraphs=[
                        Paragraph(text="Topic A has two findings.", claim_ids=[ids[0]]),
                        Paragraph(text="Market grew 99% in 2031.", claim_ids=[ids[0]]),
                    ],
                )
            ],
        )

    deps.llm.handlers["synthesizer"] = synth
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    md = report_of(deps, run.id).markdown
    assert "Topic A has two findings." in md
    assert "99%" not in md and "2031" not in md
    dropped = [e for e in events(deps, run.id) if e.type == "paragraph_dropped"]
    assert any("numbers" in e.payload.get("reason", "") for e in dropped)


async def test_synthesizer_failure_falls_back_to_verified_claims(deps):
    script_happy_path(deps)
    deps.llm.handlers["synthesizer"] = lambda s, u: LlmOutputError("bad json twice")
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    assert get_run(deps, run.id).status == "completed"
    md = report_of(deps, run.id).markdown
    assert "A fact 1 round 0" in md and "synthesis_fallback" in event_types(deps, run.id)


def _api_error(kind):
    import httpx
    import openai

    req = httpx.Request("POST", "https://api.meshapi.ai/v1/chat/completions")
    if kind == "timeout":
        return openai.APITimeoutError(request=req)
    return openai.InternalServerError(
        "upstream 502", response=httpx.Response(502, request=req), body=None
    )


@pytest.mark.parametrize("kind", ["timeout", "server_error"])
async def test_synthesizer_api_failure_still_produces_a_report(deps, kind):
    """A live run timed out twice at this last step and lost 61 verified claims."""
    script_happy_path(deps)
    deps.llm.handlers["synthesizer"] = lambda s, u: _api_error(kind)
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    assert get_run(deps, run.id).status == "completed"
    assert "A fact 1 round 0" in report_of(deps, run.id).markdown
    fallback = [e for e in events(deps, run.id) if e.type == "synthesis_fallback"]
    assert fallback and "Error" in fallback[0].payload["reason"]


# --- I3: runs orphaned by a restart are marked failed on startup -----------------------


def test_startup_marks_interrupted_runs_failed(deps):
    with deps.session_factory() as s:
        s.add(Run(id="orphan", question="q", status="running", models={}))
        s.add(Run(id="waiting", question="q", status="queued", models={}))
        s.add(Run(id="done", question="q", status="completed", models={}))
        s.commit()
    TestClient(create_app(deps.settings, deps=deps))
    assert get_run(deps, "orphan").status == "failed"
    assert "interrupted" in get_run(deps, "orphan").error
    assert get_run(deps, "waiting").status == "failed"
    assert get_run(deps, "done").status == "completed"
    assert "run_failed" in event_types(deps, "orphan")


# --- I4 / d / I8: LLM client hardening -------------------------------------------------


def test_default_client_has_bounded_timeout_and_no_sdk_retries(settings):
    llm = MeshLlm(settings)
    assert llm.client.max_retries == 0
    assert llm.client.timeout == settings.llm_timeout_s


async def test_chat_json_tolerates_code_fences(settings):
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    fake = FakeOpenAI(chat=['```json\n{"answer": "hi"}\n```'])
    out = await MeshLlm(settings, client=fake).chat_json(
        role="verifier", system="s", user="u", schema=Out
    )
    assert out.answer == "hi" and len(fake.chat_calls) == 1


async def test_embed_is_batched_and_keeps_order(settings):
    fake = FakeOpenAI()
    texts = ["x" * (i % 7 + 1) for i in range(150)]
    vecs = await MeshLlm(settings, client=fake).embed(texts)
    assert len(fake.embed_calls) == 3
    assert [v[0] for v in vecs] == [float(len(t)) for t in texts]


# --- I5: fuzzy quote matching can't flip meaning ---------------------------------------

SRC = "The policy was effective at reducing emissions in the region."


@pytest.mark.parametrize(
    "quote",
    [
        "The policy was not effective at reducing emissions",
        "The policy was ineffective at reducing emissions",
    ],
)
def test_fuzzy_quote_cannot_add_negation(quote):
    assert not quote_in_chunk(quote, SRC)


def test_fuzzy_quote_cannot_drop_negation():
    src = "The policy was not effective at reducing emissions in the region."
    assert not quote_in_chunk("The policy was effective at reducing emissions", src)


def test_dropped_punctuation_still_matches():
    assert quote_in_chunk("The policy was effective at reducing emissions in the region", SRC)


# --- I6: latency is measured from when the run started, not when it was queued ---------


async def test_run_records_started_at(deps):
    script_happy_path(deps)
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    r = get_run(deps, run.id)
    assert r.started_at is not None and r.finished_at >= r.started_at


async def test_eval_latency_excludes_queue_time(deps):
    from datetime import timedelta

    from eval.run import run_stats

    script_happy_path(deps)
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    with deps.session_factory() as s:
        r = s.get(Run, run.id)
        r.created_at = r.created_at - timedelta(hours=1)
        s.add(r)
        s.commit()
    assert run_stats(deps, [run.id])["latency_s"]["p50"] < 3600


# --- a / f: fetch robustness and caching ----------------------------------------------


async def test_fetch_unexpected_exception_returns_none(session_factory):
    def handler(request):
        raise ValueError("weird url")

    f = HttpFetcher(session_factory, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await f.fetch_text("https://x.org") is None


ARTICLE = (
    "<html><body><article>"
    + "".join(
        f"<p>Paragraph {i}: cell production and prices changed in segment {i} this year.</p>"
        for i in range(12)
    )
    + "</article></body></html>"
)


async def test_transient_fetch_failure_is_not_cached(session_factory):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"})

    f = HttpFetcher(session_factory, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await f.fetch_text("https://y.org") is None
    assert await f.fetch_text("https://y.org") is not None
    assert len(calls) == 2


async def test_permanent_fetch_failure_is_cached(session_factory):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, text="gone", headers={"content-type": "text/html"})

    f = HttpFetcher(session_factory, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await f.fetch_text("https://z.org")
    await f.fetch_text("https://z.org")
    assert len(calls) == 1


# --- b: short Tavily stubs fall back to fetching; fetch errors are isolated -----------


async def test_short_raw_content_falls_back_to_fetch(deps, run, sq):
    deps.search.results[sq.query] = [SearchResult("https://a.org/p", "A", "Enable JavaScript.")]
    deps.fetcher.pages["https://a.org/p"] = "real article text " * 100
    await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
    with deps.session_factory() as s:
        from sqlmodel import select

        from app.models import EvidenceChunk

        texts = s.exec(select(EvidenceChunk.text)).all()
    assert texts and "real article text" in texts[0]


async def test_fetcher_exception_is_skipped_not_fatal(deps, run, sq):
    class Boom:
        async def fetch_text(self, url):
            raise RuntimeError("boom")

    deps.fetcher = Boom()
    deps.search.results[sq.query] = [SearchResult("https://b.org/p", "B", None)]
    out = await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
    assert out == {"researched": [sq.id]} and chunk_count(deps, sq.id) == 0
    assert "source_skipped" in event_types(deps, run.id)


# --- e: one bad judge reply doesn't abort the eval ------------------------------------


async def test_eval_survives_judge_errors(deps, tmp_path):
    from eval.run import evaluate, research_questions

    script_happy_path(deps)
    replies = iter([LlmOutputError("bad")] + [JudgeOutput(supported=True, reason="ok")] * 50)
    deps.llm.handlers["judge"] = lambda s, u: next(replies)
    run_ids = await research_questions(deps, [{"id": "q01", "question": "q"}])
    results = await evaluate(deps, run_ids, random.Random(1), kinds=(), gold_path=tmp_path / "x")
    j = results["independent_judge"]
    assert j["errors"] == 1 and j["n"] == 5 and j["groundedness"] == 1.0


# --- wording: thin sub-questions may still have some verified evidence -----------------


def test_limited_evidence_wording_is_accurate():
    from app.report import render_report

    md = render_report(
        "q", SynthesisOutput(title="T", summary="S", sections=[]), {}, {}, {}, ["X?"], []
    )
    assert "says nothing about them" not in md and "limited verified evidence" in md


# --- live run 1: per-sub-question LLM stages must run concurrently ----------------------


class _ConcurrencyProbe:
    """Wraps a ScriptedLlm and records the max number of in-flight calls per role."""

    def __init__(self, inner):
        self.inner, self.in_flight, self.peak = inner, {}, {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def chat_json(self, *, role, **kw):
        import asyncio

        self.in_flight[role] = self.in_flight.get(role, 0) + 1
        self.peak[role] = max(self.peak.get(role, 0), self.in_flight[role])
        await asyncio.sleep(0.01)
        try:
            return await self.inner.chat_json(role=role, **kw)
        finally:
            self.in_flight[role] -= 1


async def test_drafter_and_contradictions_run_subquestions_concurrently(deps):
    script_happy_path(deps)
    probe = _ConcurrencyProbe(deps.llm)
    deps.llm = probe
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)
    assert get_run(deps, run.id).status == "completed"
    assert probe.peak["drafter"] == 3 and probe.peak["contradictions"] == 3
