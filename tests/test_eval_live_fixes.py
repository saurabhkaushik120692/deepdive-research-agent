"""Regression tests for failures seen in the first full eval run (Mesh 429s, dense chunks)."""

import httpx
import openai
import pytest
from pydantic import BaseModel

from app.deps import SearchResult
from app.graph.nodes.researcher import research_node
from app.graph.runner import create_run, run_research
from app.llm import MeshLlm, RateLimiter
from app.tools.chunking import is_noise
from tests.fakes import FakeOpenAI
from tests.helpers import cfg, chunk_count


class Out(BaseModel):
    answer: str


def _rate_limited():
    req = httpx.Request("POST", "https://api.meshapi.ai/v1/chat/completions")
    return openai.RateLimitError(
        "RPM limit exceeded", response=httpx.Response(429, request=req), body=None
    )


async def test_rate_limiter_waits_when_window_is_full():
    now = [0.0]
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        now[0] += s

    limiter = RateLimiter(rpm=2, clock=lambda: now[0], sleep=fake_sleep)
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()  # third request in the same minute must wait for the window
    assert len(sleeps) == 1 and 59.9 <= sleeps[0] <= 60.1


async def test_429_is_retried_long_enough(settings):
    fake = FakeOpenAI(chat=[_rate_limited()] * 4 + ['{"answer": "ok"}'])
    llm = MeshLlm(settings, client=fake, retry_base_delay=0, rate_limit_wait=0)
    out = await llm.chat_json(role="planner", system="s", user="u", schema=Out)
    assert out.answer == "ok" and len(fake.chat_calls) == 5


async def test_embed_input_is_truncated(settings):
    fake = FakeOpenAI()
    await MeshLlm(settings, client=fake).embed(["x" * 5000])
    sent = fake.embed_calls[0]["input"][0]
    assert len(sent) == settings.embed_max_chars


@pytest.mark.parametrize(
    "text,noisy",
    [
        ("-0.47 | 0.24 | -0.24 | -0.24 | | December 2013 | -0.12 | 0.06 | -0.18 | " * 20, True),
        ("Preface xi Foreword " + "-" * 200 + " v Preface " + "-" * 200, True),
        ("%3E%3Cpath d='M15.9953 7.9996c0 4.418-3.5816 7.9996-7.9996' " * 20, True),
        (
            "Global EV battery demand rose 28% to 1.1 TWh in 2025, according to the IEA. " * 10,
            False,
        ),
    ],
)
def test_noise_detection(text, noisy):
    assert is_noise(text) is noisy


async def test_researcher_drops_noise_chunks(deps, run, sq):
    prose = "Battery demand rose sharply this year across major markets and segments. " * 60
    table = "| -0.47 | 0.24 | -0.24 | 0.06 | -0.18 | 2014 | 0.12 | " * 120
    deps.search.results[sq.query] = [
        SearchResult("https://a.org/p", "A", prose),
        SearchResult("https://b.org/t", "B", table),
    ]
    await research_node({"run_id": run.id, "sub_question_id": sq.id, "round": 0}, cfg(deps))
    assert chunk_count(deps, sq.id) == 2  # prose only (~4500 chars → 2 chunks); table dropped


async def test_retry_failed_reruns_only_failed_questions(deps):
    from app.models import Run
    from eval.run import retry_failed
    from tests.scripts import script_happy_path

    script_happy_path(deps)
    with deps.session_factory() as s:
        s.add(Run(id="good", question="Q1", status="completed", models={}))
        s.add(Run(id="bad", question="Q2", status="failed", models={}))
        s.commit()
    stored = [{"question_id": "q01", "run_id": "good"}, {"question_id": "q02", "run_id": "bad"}]
    questions = [{"id": "q01", "question": "Q1"}, {"id": "q02", "question": "Q2"}]
    new = await retry_failed(deps, stored, questions)
    assert new[0] == {"question_id": "q01", "run_id": "good"}
    assert new[1]["question_id"] == "q02" and new[1]["run_id"] != "bad"
    with deps.session_factory() as s:
        assert s.get(Run, new[1]["run_id"]).status == "completed"


async def test_repair_never_sends_empty_assistant_message(settings):
    fake = FakeOpenAI(chat=["", '{"answer": "x"}'])
    out = await MeshLlm(settings, client=fake).chat_json(
        role="verifier", system="s", user="u", schema=Out
    )
    assert out.answer == "x"
    repair_messages = fake.chat_calls[1]["messages"]
    assert all(m["content"].strip() for m in repair_messages)


async def test_verifier_probe_counts_api_errors_instead_of_crashing(deps):
    import random

    from eval.run import verifier_probe
    from tests.scripts import script_happy_path

    script_happy_path(deps)
    run = create_run(deps.session_factory(), "q", deps.settings)
    await run_research(run.id, deps)

    def broken_verifier(system, user):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.meshapi.ai/v1"))

    deps.llm.handlers["verifier"] = broken_verifier
    probe = await verifier_probe(deps, [run.id], random.Random(1), kinds=("number",))
    assert probe["errors"] > 0 and probe["n_originals"] == 6


# --- embedding coverage: embed most of each chunk, fall back when an input is too long ------


async def test_embedding_covers_most_of_a_chunk(settings):
    fake = FakeOpenAI()
    await MeshLlm(settings, client=fake).embed(["x" * 3000])
    assert settings.embed_max_chars >= 2400
    assert len(fake.embed_calls[0]["input"][0]) == settings.embed_max_chars


async def test_too_long_embedding_input_falls_back_to_shorter_text(settings):
    fake = FakeOpenAI(embed_max_len=2000)
    vecs = await MeshLlm(settings, client=fake).embed(["short text", "y" * 3000])
    assert len(vecs) == 2
    retry = fake.embed_calls[-1]["input"]
    assert all(len(t) <= settings.embed_fallback_chars for t in retry)
    assert vecs[0] == [10.0, 1.0]  # the short input is unchanged in the retry


async def test_embedding_fallback_gives_up_after_one_retry(settings):
    fake = FakeOpenAI(embed_max_len=5)
    with pytest.raises(openai.BadRequestError):
        await MeshLlm(settings, client=fake).embed(["this is longer than five"])
    assert len(fake.embed_calls) == 2
