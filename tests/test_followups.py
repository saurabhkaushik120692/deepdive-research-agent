"""Follow-up questions on a finished report."""

import re
import sqlite3
from datetime import UTC, datetime

import openai
import pytest
from sqlalchemy import inspect
from sqlmodel import create_engine, select

from app.db import init_db
from app.deps import SearchResult
from app.events import log_event
from app.graph.schemas import (
    DraftClaim,
    DraftOutput,
    FollowUpAnswer,
    Paragraph,
    ScopeOutput,
    VerdictOutput,
)
from app.models import (
    Claim,
    Event,
    EvidenceChunk,
    FollowUp,
    Report,
    Run,
    Source,
    SubQuestion,
    User,
    Verdict,
)

# --- data, settings, vectors ------------------------------------------------------------------


def test_followup_columns_migrate_on_old_db(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE run (id VARCHAR PRIMARY KEY, question VARCHAR NOT NULL)")
    con.execute(
        "CREATE TABLE subquestion (id INTEGER PRIMARY KEY, run_id VARCHAR NOT NULL,"
        " text VARCHAR NOT NULL, query VARCHAR NOT NULL, round INTEGER, status VARCHAR)"
    )
    con.execute(
        "CREATE TABLE claim (id INTEGER PRIMARY KEY, run_id VARCHAR NOT NULL,"
        " sub_question_id INTEGER NOT NULL, text VARCHAR NOT NULL, chunk_ids JSON,"
        " round INTEGER, in_draft BOOLEAN)"
    )
    con.execute(
        "CREATE TABLE event (id INTEGER PRIMARY KEY, run_id VARCHAR NOT NULL, node VARCHAR,"
        " type VARCHAR, ts DATETIME, payload JSON)"
    )
    con.commit()
    con.close()
    engine = create_engine(f"sqlite:///{path}")
    init_db(engine)
    init_db(engine)  # idempotent
    insp = inspect(engine)
    for table in ("subquestion", "claim", "event"):
        assert "followup_id" in {c["name"] for c in insp.get_columns(table)}
    assert "followup" in insp.get_table_names()


def test_log_event_tags_followup_id(session, run):
    tagged = log_event(session, run.id, "followup_scope", "scope_checked", {"followup_id": 7})
    plain = log_event(session, run.id, "planner", "planned", {})
    assert tagged.followup_id == 7
    assert plain.followup_id is None


def test_followup_drafter_family_must_differ():
    from app.config import Settings

    with pytest.raises(ValueError):
        Settings(_env_file=None, model_followup_drafter="anthropic/x", model_verifier="anthropic/y")


def test_followup_settings_defaults(settings):
    assert settings.followups_per_report == 10
    assert settings.followup_max_age_days == 30
    for role in ("scope", "followup_drafter", "followup_answer"):
        assert getattr(settings, f"model_{role}") == "openai/gpt-5-nano"
        assert getattr(settings, f"reasoning_effort_{role}") == "low"


async def test_memory_vectors_query_run():
    from tests.fakes import MemoryVectors

    v = MemoryVectors()
    await v.add("r1", 1, [10, 11], ["a", "b"])
    await v.add("r1", 2, [20], ["c"])
    await v.add("r2", 3, [30], ["d"])
    assert await v.query_run("r1", "anything", 5) == [10, 11, 20]
    assert await v.query_run("r1", "anything", 2) == [10, 11]


# --- the follow-up graph -----------------------------------------------------------------------

CUT_FEB = "The MPC cut the repo rate by 25 basis points to 6.25% in February 2025."
CUT_APR = "The MPC cut the repo rate again by 25 basis points to 6.0% in April 2025."
INFLATION = "Retail inflation eased to 3.2% in March 2025."
FOOD = "Food prices fell sharply in early 2025, pulling inflation down."
REJECTED = "The MPC moved to an accommodative stance in March 2025."
WEB_FACT_1 = "Analysts expected one more cut of 25 basis points in June 2025."
WEB_FACT_2 = "Bank lending rates fell by about 40 basis points after the cuts."
QUERY = "RBI repo rate cut reasons 2025"


def _chunk(s, run_id, sq_id, url, text, tier=1):
    src = Source(run_id=run_id, url=url, domain=url.split("/")[2], tier=tier)
    s.add(src)
    s.commit()
    s.refresh(src)
    ch = EvidenceChunk(source_id=src.id, run_id=run_id, sub_question_id=sq_id, text=text)
    s.add(ch)
    s.commit()
    s.refresh(ch)
    return ch


def seed_report(deps, *, user_id=None, finished_at=None, is_example=False, status="completed"):
    """A finished report: one sub-question, two saved chunks, a supported and a rejected claim.

    Returns the ids. Vectors are added by the async fixture below."""
    with deps.session_factory() as s:
        if user_id is None:
            user = User(google_sub="g-owner", email="owner@example.com")
            s.add(user)
            s.commit()
            s.refresh(user)
            user_id = user.id
        run = Run(
            question="How did the RBI repo rate change in 2025, and why?",
            user_id=user_id,
            status=status,
            is_example=is_example,
            finished_at=finished_at or datetime.now(UTC),
            models={},
        )
        s.add(run)
        s.commit()
        s.refresh(run)
        sq = SubQuestion(run_id=run.id, text="How did the repo rate change?", query="repo rate")
        s.add(sq)
        s.commit()
        s.refresh(sq)
        a = _chunk(s, run.id, sq.id, "https://rbi.org.in/a", f"{CUT_FEB} {CUT_APR}")
        b = _chunk(s, run.id, sq.id, "https://reuters.com/b", f"{INFLATION} {FOOD}", tier=2)
        good = Claim(run_id=run.id, sub_question_id=sq.id, text=CUT_FEB, chunk_ids=[a.id])
        bad = Claim(run_id=run.id, sub_question_id=sq.id, text=REJECTED, chunk_ids=[a.id])
        s.add_all([good, bad])
        s.commit()
        s.refresh(good)
        s.refresh(bad)
        s.add(Verdict(claim_id=good.id, label="supported", quote=CUT_FEB, quote_found=True))
        s.add(Verdict(claim_id=bad.id, label="unsupported", reason="not in source"))
        synthesis = {
            "title": "RBI repo rate in 2025",
            "summary": "The RBI cut the repo rate in 2025.",
            "sections": [
                {"heading": "Rate cuts", "paragraphs": [{"text": CUT_FEB, "claim_ids": [good.id]}]}
            ],
        }
        s.add(Report(run_id=run.id, markdown="# r", metrics={}, synthesis=synthesis))
        s.commit()
        return {
            "run": run.id,
            "user": user_id,
            "sq": sq.id,
            "a": a.id,
            "b": b.id,
            "good": good.id,
            "bad": bad.id,
        }


@pytest.fixture
async def report_run(deps):
    ids = seed_report(deps)
    await deps.vectors.add(ids["run"], ids["sq"], [ids["a"], ids["b"]], ["a", "b"])
    return ids


def _new_followup(deps, ids, question="Why did the RBI cut rates in 2025?", status="queued"):
    with deps.session_factory() as s:
        f = FollowUp(run_id=ids["run"], user_id=ids["user"], question=question, status=status)
        s.add(f)
        s.commit()
        s.refresh(f)
        return f.id


def _followup(deps, fid):
    with deps.session_factory() as s:
        return s.get(FollowUp, fid)


def _chunk_ids(user):
    return [int(i) for i in re.findall(r"\[chunk:(\d+)\]", user)]


def _claim_ids(user):
    return [int(i) for i in re.findall(r"\[claim:(\d+)\]", user)]


def _in_scope(s, u):
    return ScopeOutput(
        in_scope=True,
        reason="asks about the report's topic",
        standalone_question="Why did the RBI cut the repo rate in 2025?",
        search_query=QUERY,
    )


def _verifier(s, u):
    claim = u.split("Claim:\n", 1)[1].split("\n\nSource text:", 1)[0]
    source = u.split("Source text:\n", 1)[1]
    if claim in source:
        return VerdictOutput(label="supported", quote=claim, reason="stated")
    return VerdictOutput(label="unsupported", quote="", reason="not stated")


def _drafter_by_text(table):
    """Draft every sentence in `table` that appears in a chunk the drafter was given."""

    def handler(s, u):
        evidence = u.split("Evidence:", 1)[1]
        claims = []
        for text in table:
            for cid in _chunk_ids(u):
                block = evidence.split(f"[chunk:{cid}]", 1)[1].split("[chunk:", 1)[0]
                if text in block:
                    claims.append(DraftClaim(sub_question_id=0, text=text, chunk_ids=[cid]))
                    break
        return DraftOutput(claims=claims)

    return handler


def _answer_citing_all(s, u):
    text = "The RBI cut rates as inflation eased."
    return FollowUpAnswer(paragraphs=[Paragraph(text=text, claim_ids=_claim_ids(u))])


def _happy(deps, drafted=(INFLATION, FOOD)):
    deps.llm.handlers |= {
        "scope": _in_scope,
        "followup_drafter": _drafter_by_text(list(drafted)),
        "verifier": _verifier,
        "followup_answer": _answer_citing_all,
    }


async def test_followup_declined_off_topic(deps, report_run):
    from app.graph.followup import run_followup

    deps.llm.handlers["scope"] = lambda s, u: ScopeOutput(
        in_scope=False,
        reason="This is about stocks, not the RBI repo rate.",
        standalone_question="",
        search_query="",
    )
    fid = _new_followup(deps, report_run, "What's the best EV stock?")
    await run_followup(fid, deps)
    f = _followup(deps, fid)
    assert f.status == "declined" and "stocks" in f.decline_reason
    assert [r for r, _ in deps.llm.calls] == ["scope"]
    with deps.session_factory() as s:
        assert s.exec(select(SubQuestion).where(SubQuestion.followup_id == fid)).all() == []


async def test_followup_answers_from_saved_evidence_without_search(deps, report_run):
    from app.graph.followup import run_followup

    _happy(deps)
    fid = _new_followup(deps, report_run)
    await run_followup(fid, deps)
    f = _followup(deps, fid)
    assert f.status == "answered" and f.searched_web is False
    assert deps.search.queries == []
    [p] = f.answer["paragraphs"]
    with deps.session_factory() as s:
        texts = {s.get(Claim, i).text for i in p["claim_ids"]}
    assert texts == {CUT_FEB, INFLATION, FOOD}  # the report's verified claim is reused
    assert f.standalone_question == "Why did the RBI cut the repo rate in 2025?"


async def test_followup_searches_web_when_thin(deps, report_run):
    from app.graph.followup import run_followup

    page = f"{WEB_FACT_1} {WEB_FACT_2} " + "Context about monetary policy in India. " * 20
    deps.search.results[QUERY] = [SearchResult("https://livemint.com/x", "Mint", page)]
    _happy(deps, drafted=(INFLATION, WEB_FACT_1, WEB_FACT_2))
    fid = _new_followup(deps, report_run)
    await run_followup(fid, deps)
    f = _followup(deps, fid)
    assert f.status == "answered" and f.searched_web is True
    assert deps.search.queries == [QUERY]
    with deps.session_factory() as s:
        texts = {s.get(Claim, i).text for i in f.answer["paragraphs"][0]["claim_ids"]}
    assert {WEB_FACT_1, WEB_FACT_2} <= texts


async def test_followup_cannot_redraft_report_claims(deps, report_run):
    from app.graph.followup import run_followup

    _happy(deps)
    deps.llm.handlers["followup_drafter"] = lambda s, u: DraftOutput(
        claims=[DraftClaim(sub_question_id=0, text=REJECTED, chunk_ids=[report_run["a"]])]
    )
    fid = _new_followup(deps, report_run)
    await run_followup(fid, deps)
    with deps.session_factory() as s:
        redrafted = s.exec(select(Claim).where(Claim.followup_id == fid)).all()
        dropped = s.exec(
            select(Event).where(Event.followup_id == fid, Event.type == "claim_dropped_duplicate")
        ).all()
    assert redrafted == [] and dropped


async def test_followup_answer_drops_paragraph_with_unsupported_number(deps, report_run):
    from app.graph.followup import run_followup

    _happy(deps)
    deps.llm.handlers["followup_answer"] = lambda s, u: FollowUpAnswer(
        paragraphs=[
            Paragraph(text="Inflation fell to 1.5% so the RBI cut.", claim_ids=_claim_ids(u)[:1])
        ]
    )
    fid = _new_followup(deps, report_run)
    await run_followup(fid, deps)
    f = _followup(deps, fid)
    texts = [p["text"] for p in f.answer["paragraphs"]]
    assert all("1.5%" not in t for t in texts)
    assert len(texts) == 3  # every allowed claim is still listed, one line each
    with deps.session_factory() as s:
        assert s.exec(
            select(Event).where(Event.followup_id == fid, Event.type == "paragraph_dropped")
        ).all()


async def test_followup_none_found(deps, report_run):
    from app.graph.followup import run_followup

    deps.vectors.store.clear()
    _happy(deps, drafted=())
    fid = _new_followup(deps, report_run)
    await run_followup(fid, deps)
    f = _followup(deps, fid)
    assert f.status == "answered" and f.answer["none_found"] is True
    assert f.answer["paragraphs"] == []
    assert "followup_answer" not in [r for r, _ in deps.llm.calls]


async def test_followup_mesh_error_marks_failed(deps, report_run):
    from app.graph.followup import run_followup

    def boom(s, u):
        raise openai.APIConnectionError(request=None)

    _happy(deps)
    deps.llm.handlers["followup_answer"] = boom
    fid = _new_followup(deps, report_run)
    await run_followup(fid, deps)  # never raises
    f = _followup(deps, fid)
    assert f.status == "failed" and f.error and f.finished_at is not None


def test_fail_interrupted_followups_on_startup(deps, report_run):
    from app.graph.followup import fail_interrupted_followups

    fid = _new_followup(deps, report_run, status="running")
    assert fail_interrupted_followups(deps) == 1
    assert _followup(deps, fid).status == "failed"


async def test_followup_events_tagged(deps, report_run):
    from app.graph.followup import run_followup

    with deps.session_factory() as s:
        log_event(s, report_run["run"], "planner", "planned", {})
    _happy(deps)
    fid = _new_followup(deps, report_run)
    await run_followup(fid, deps)
    with deps.session_factory() as s:
        evs = s.exec(select(Event).where(Event.run_id == report_run["run"])).all()
    before = [e for e in evs if e.type == "planned"]
    after = [e for e in evs if e.type not in ("planned", "llm_call")]
    assert before[0].followup_id is None
    assert after and all(e.followup_id == fid for e in after)


# --- rules and API -----------------------------------------------------------------------------

from datetime import timedelta  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from tests.helpers import sign_in  # noqa: E402

ASK = "Why did the RBI cut rates twice?"


async def _no_followup(followup_id, deps):
    pass


@pytest.fixture
def owner(deps, monkeypatch):
    """A signed-in owner client; follow-ups are recorded but not run."""
    monkeypatch.setattr("app.api.followups.run_followup", _no_followup)
    deps.settings.google_client_id, deps.settings.google_client_secret = "cid", "s"
    deps.settings.contact_email = "admin@example.com"
    client = TestClient(create_app(deps.settings, deps=deps))
    client.user_id = sign_in(client)
    return client


def _ask(client, run_id, question=ASK):
    return client.post(f"/api/runs/{run_id}/followups", json={"question": question})


def _count(deps, run_id):
    with deps.session_factory() as s:
        return len(s.exec(select(FollowUp).where(FollowUp.run_id == run_id)).all())


def test_followup_post_ok_returns_202(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    r = _ask(owner, ids["run"])
    assert r.status_code == 202
    assert _followup(deps, r.json()["followup_id"]).question == ASK


def test_followup_refused_on_example_report(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id, is_example=True)
    r = _ask(owner, ids["run"])
    assert r.status_code == 403 and "your own finished reports" in r.json()["detail"]


def test_followup_refused_when_older_than_30_days(deps, owner):
    old = datetime.now(UTC) - timedelta(days=31)
    ids = seed_report(deps, user_id=owner.user_id, finished_at=old)
    r = _ask(owner, ids["run"])
    assert r.status_code == 403 and "older than 30 days" in r.json()["detail"]


def test_followup_refused_after_10(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    for status in ["answered"] * 8 + ["declined", "failed"]:
        _new_followup(deps, ids, status=status)
    assert _ask(owner, ids["run"]).status_code == 202  # 9 counted: the failed one doesn't
    _update_status_all(deps, ids["run"], "answered", only="queued")
    r = _ask(owner, ids["run"])
    assert r.status_code == 403
    assert r.json()["detail"] == "You've used all 10 follow-ups on this report."


def _update_status_all(deps, run_id, status, only):
    with deps.session_factory() as s:
        for f in s.exec(select(FollowUp).where(FollowUp.run_id == run_id)).all():
            if f.status == only:
                f.status = status
                s.add(f)
        s.commit()


def test_followup_refused_for_blocked_user(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    with deps.session_factory() as s:
        u = s.get(User, owner.user_id)
        u.blocked = True
        s.add(u)
        s.commit()
    r = _ask(owner, ids["run"])
    assert r.status_code == 403 and "can't start new reports" in r.json()["detail"]


def test_followup_refused_while_report_running(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id, status="running")
    assert _ask(owner, ids["run"]).status_code == 403


def test_followup_other_users_run_404(deps, owner):
    ids = seed_report(deps)  # owned by someone else
    assert _ask(owner, ids["run"]).status_code == 404


def test_followup_second_post_while_running_409(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    assert _ask(owner, ids["run"]).status_code == 202
    r = _ask(owner, ids["run"])
    assert r.status_code == 409 and "already running" in r.json()["detail"]
    assert _count(deps, ids["run"]) == 1


def test_followup_question_too_short_422(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    assert _ask(owner, ids["run"], "a").status_code == 422


def test_followup_safety_screen_400(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    deps.llm.flagged = True
    assert _ask(owner, ids["run"]).status_code == 400
    assert _count(deps, ids["run"]) == 0


def test_followup_status_get_owner_only(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    fid = _ask(owner, ids["run"]).json()["followup_id"]
    url = f"/api/runs/{ids['run']}/followups/{fid}"
    r = owner.get(url)
    assert r.status_code == 200
    assert r.json() | {"stage": "x"} == {
        "id": fid,
        "status": "queued",
        "stage": "x",
        "decline_reason": "",
        "searched_web": False,
    }
    sign_in(owner, sub="someone-else", email="other@example.com")
    assert owner.get(url).status_code == 404
    sign_in(owner, sub="the-admin", email="admin@example.com")
    assert owner.get(url).status_code == 200


# --- the report stays as it was ----------------------------------------------------------------


def _verifier_rejecting_food(s, u):
    if FOOD in u.split("Claim:\n", 1)[1].split("\n\nSource text:", 1)[0]:
        return VerdictOutput(label="unsupported", quote="", reason="not stated")
    return _verifier(s, u)


async def _answered_followup(deps, ids):
    """A follow-up with one supported and one rejected claim of its own."""
    from app.graph.followup import run_followup

    _happy(deps)
    deps.llm.handlers["verifier"] = _verifier_rejecting_food
    fid = _new_followup(deps, ids)
    await run_followup(fid, deps)
    assert _followup(deps, fid).status == "answered"
    return fid


def _view(deps, run_id):
    from app.api.pages import _report_view

    with deps.session_factory() as s:
        return _report_view(s, s.get(Run, run_id))


async def test_report_view_unchanged_by_followup(deps, report_run):
    before = _view(deps, report_run["run"])
    await _answered_followup(deps, report_run)
    after = _view(deps, report_run["run"])
    assert after["counts"] == before["counts"]
    assert after["rejected"] == before["rejected"]
    assert after["thin"] == before["thin"]


async def test_sse_excludes_followup_events(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    await deps.vectors.add(ids["run"], ids["sq"], [ids["a"], ids["b"]], ["a", "b"])
    with deps.session_factory() as s:
        log_event(s, ids["run"], "planner", "planned", {})
    await _answered_followup(deps, ids)
    body = owner.get(f"/api/runs/{ids['run']}/events").text
    assert "event: planned" in body
    assert "scope_checked" not in body and "followup_started" not in body


async def test_verdict_samples_ignore_followup_claims(deps, report_run):
    from app.api.pages import _verdict_samples

    await _answered_followup(deps, report_run)
    with deps.session_factory() as s:
        samples = _verdict_samples(s, report_run["run"], pick=lambda rows: rows[-1])
    assert {x["text"] for x in samples} == {CUT_FEB, REJECTED}


async def test_trail_lists_followups_separately(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    await deps.vectors.add(ids["run"], ids["sq"], [ids["a"], ids["b"]], ["a", "b"])
    fid = await _answered_followup(deps, ids)
    trail = owner.get(f"/api/runs/{ids['run']}/trail").json()
    assert {c["text"] for c in trail["claims"]} == {CUT_FEB, REJECTED}
    assert [sq["id"] for sq in trail["sub_questions"]] == [ids["sq"]]
    assert all(e["followup_id"] is None for e in trail["events"])
    [f] = trail["followups"]
    assert f["followup"]["id"] == fid
    assert {c["text"] for c in f["claims"]} == {INFLATION, FOOD}
    assert {c["verdict"]["label"] for c in f["claims"]} == {"supported", "unsupported"}
    assert any(e["type"] == "scope_checked" for e in f["events"])


# --- report page -------------------------------------------------------------------------------


def _page(client, run_id):
    r = client.get(f"/runs/{run_id}/report")
    assert r.status_code == 200
    return r.text


def test_report_page_shows_followup_box_for_owner(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    html = _page(owner, ids["run"])
    assert 'id="followups"' in html and 'id="followup-form"' in html
    assert "10 of 10 left" in html


async def test_example_report_has_no_followup_section_for_public(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    await deps.vectors.add(ids["run"], ids["sq"], [ids["a"], ids["b"]], ["a", "b"])
    await _answered_followup(deps, ids)
    with deps.session_factory() as s:
        run = s.get(Run, ids["run"])
        run.is_example = True  # marked as an example after the follow-up was asked
        s.add(run)
        s.commit()
    public = TestClient(owner.app)
    public.cookies.clear()
    html = _page(public, ids["run"])
    assert 'id="followups"' not in html and "Why did the RBI cut rates" not in html
    assert "Why did the RBI cut rates" in _page(owner, ids["run"])  # the owner still sees it
    assert 'id="followup-form"' not in _page(owner, ids["run"])  # but can't ask more


def test_old_report_shows_closed_message(deps, owner):
    old = datetime.now(UTC) - timedelta(days=45)
    ids = seed_report(deps, user_id=owner.user_id, finished_at=old)
    html = _page(owner, ids["run"])
    assert "Follow-ups are closed for reports older than 30 days." in html
    assert 'id="followup-form"' not in html


async def test_answered_followup_renders_with_citation_quote(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    await deps.vectors.add(ids["run"], ids["sq"], [ids["a"], ids["b"]], ["a", "b"])
    await _answered_followup(deps, ids)
    html = _page(owner, ids["run"])
    assert "The RBI cut rates as inflation eased." in html
    assert f"<q>{INFLATION}</q>" in html  # the verifier's quote, shown on hover
    assert "9 of 10 left" in html


async def test_pending_followup_shows_its_stage(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    fid = _new_followup(deps, ids, status="running")
    with deps.session_factory() as s:
        f = s.get(FollowUp, fid)
        f.stage = "searching"
        s.add(f)
        s.commit()
    html = _page(owner, ids["run"])
    assert f'data-followup="{fid}"' in html and "Searching the web for more…" in html


def test_followup_fragment_route_owner_only(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    r = owner.get(f"/runs/{ids['run']}/followups")
    assert r.status_code == 200 and 'id="followups"' in r.text and "<html" not in r.text
    sign_in(owner, sub="someone-else", email="other@example.com")
    assert owner.get(f"/runs/{ids['run']}/followups").status_code == 404


async def test_trail_page_lists_followups(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    await deps.vectors.add(ids["run"], ids["sq"], [ids["a"], ids["b"]], ["a", "b"])
    await _answered_followup(deps, ids)
    html = owner.get(f"/runs/{ids['run']}/trail").text
    assert "Follow-ups" in html and "scope_checked" in html and FOOD in html


async def test_public_trail_json_hides_followups_on_example(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id, is_example=True)
    await deps.vectors.add(ids["run"], ids["sq"], [ids["a"], ids["b"]], ["a", "b"])
    await _answered_followup(deps, ids)
    public = TestClient(owner.app)
    public.cookies.clear()
    assert public.get(f"/api/runs/{ids['run']}/trail").json()["followups"] == []
    assert len(owner.get(f"/api/runs/{ids['run']}/trail").json()["followups"]) == 1


# --- from the live check -----------------------------------------------------------------------


async def test_uncited_claims_listed_only_when_no_paragraph_survives(deps, report_run):
    from app.graph.followup import run_followup

    _happy(deps)

    def cites_one(s, u):
        [first, *_] = _claim_ids(u)
        with deps.session_factory() as sess:
            text = sess.get(Claim, first).text
        return FollowUpAnswer(paragraphs=[Paragraph(text=text, claim_ids=[first])])

    deps.llm.handlers["followup_answer"] = cites_one
    fid = _new_followup(deps, report_run)
    await run_followup(fid, deps)
    assert len(_followup(deps, fid).answer["paragraphs"]) == 1


def test_paragraph_cites_one_marker_per_source():
    from app.api.followup_views import paragraph_view
    from app.report import Citations

    src = Source(id=1, run_id="r", url="https://rbi.org.in/a", domain="rbi.org.in", tier=1)
    claims = {
        1: Claim(id=1, run_id="r", sub_question_id=1, text="a", chunk_ids=[10]),
        2: Claim(id=2, run_id="r", sub_question_id=1, text="b", chunk_ids=[11]),
    }
    view = paragraph_view("a and b", [1, 2], Citations(claims, {10: src, 11: src}), {})
    assert [c["n"] for c in view["cites"]] == [1]


def test_ask_box_hidden_while_a_followup_runs(deps, owner):
    ids = seed_report(deps, user_id=owner.user_id)
    _new_followup(deps, ids, status="running")
    html = _page(owner, ids["run"])
    assert 'id="followup-form"' not in html and "data-followup=" in html


async def test_public_trail_json_unchanged_by_followup_with_web_search(deps, owner):
    """Sources fetched and model calls made by a follow-up never reach the report's trail."""
    from app.graph.followup import run_followup

    ids = seed_report(deps, user_id=owner.user_id, is_example=True)
    await deps.vectors.add(ids["run"], ids["sq"], [ids["a"], ids["b"]], ["a", "b"])
    public = TestClient(owner.app)
    public.cookies.clear()
    url = f"/api/runs/{ids['run']}/trail"
    before = public.get(url).json()

    page = f"{WEB_FACT_1} {WEB_FACT_2} " + "Context about monetary policy in India. " * 20
    deps.search.results[QUERY] = [SearchResult("https://livemint.com/x", "Mint", page)]
    _happy(deps, drafted=(INFLATION, WEB_FACT_1, WEB_FACT_2))
    scope = deps.llm.handlers["scope"]

    def scope_logging_usage(s, u):
        # What MeshLlm's usage callback does for every call: an llm_call event on the run.
        with deps.session_factory() as sess:
            log_event(sess, ids["run"], "llm", "llm_call", {"role": "scope"})
        return scope(s, u)

    deps.llm.handlers["scope"] = scope_logging_usage
    fid = _new_followup(deps, ids)
    await run_followup(fid, deps)
    assert _followup(deps, fid).searched_web is True

    after = public.get(url).json()
    assert after["sources"] == before["sources"]
    assert after["events"] == before["events"]
