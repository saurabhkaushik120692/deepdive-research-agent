from sqlmodel import select

from app.graph.runner import create_run, run_research
from app.llm import LlmOutputError
from app.models import Claim, SubQuestion, Verdict
from tests.helpers import event_types, events, get_run, report_of
from tests.scripts import script_happy_path, script_no_results


def session_of(deps):
    return deps.session_factory()


def sq_id(deps, run_id, letter):
    with deps.session_factory() as s:
        return s.exec(
            select(SubQuestion.id).where(
                SubQuestion.run_id == run_id, SubQuestion.text == f"Sub-question {letter}"
            )
        ).one()


def rejected_claim_text(deps, run_id):
    with deps.session_factory() as s:
        return s.exec(
            select(Claim.text)
            .join(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.run_id == run_id, Verdict.label == "unsupported")
        ).first()


async def test_loop_runs_once_for_thin_subquestion(deps):
    script_happy_path(deps, thin_in_round0={"B"})
    run = create_run(session_of(deps), "EV battery risks 2026?", deps.settings)
    await run_research(run.id, deps)
    ev = events(deps, run.id)
    assert [e.payload["looping"] for e in ev if e.type == "loop_decision"] == [True, False]
    round1 = {
        e.payload["sub_question_id"]
        for e in ev
        if e.type == "search_done" and e.payload["round"] == 1
    }
    assert round1 == {sq_id(deps, run.id, "B")}
    assert get_run(deps, run.id).status == "completed"
    md = report_of(deps, run.id).markdown
    assert "## Limited evidence" not in md and "## Sources" in md


async def test_loop_stops_at_cap_and_reports_limited_evidence(deps):
    script_happy_path(deps, thin_in_round0={"B"}, always_thin={"B"})
    run = create_run(session_of(deps), "q", deps.settings)
    await run_research(run.id, deps)
    looping = [e.payload["looping"] for e in events(deps, run.id) if e.type == "loop_decision"]
    assert looping == [True, True, False]
    md = report_of(deps, run.id).markdown
    assert "## Limited evidence" in md
    assert "Sub-question B" in md.split("## Limited evidence")[1]


async def test_zero_search_results_gives_insufficient_report_not_crash(deps):
    script_no_results(deps)
    run = create_run(session_of(deps), "obscure q", deps.settings)
    await run_research(run.id, deps)
    r = get_run(deps, run.id)
    assert r.status == "completed"
    assert "## Limited evidence" in report_of(deps, run.id).markdown


async def test_llm_failure_marks_run_failed(deps):
    deps.llm.handlers["planner"] = lambda s, u: LlmOutputError("mesh 503 after retries")
    run = create_run(session_of(deps), "q", deps.settings)
    await run_research(run.id, deps)
    r = get_run(deps, run.id)
    assert r.status == "failed" and "503" in r.error and r.finished_at is not None
    assert "run_failed" in event_types(deps, run.id)


async def test_synthesizer_cannot_cite_rejected_claims(deps):
    script_happy_path(deps, thin_in_round0={"B"}, synth_cites_rejected=True)
    run = create_run(session_of(deps), "q", deps.settings)
    await run_research(run.id, deps)
    md = report_of(deps, run.id).markdown
    rejected_text = rejected_claim_text(deps, run.id)
    assert rejected_text == "B fact 2 round 0"
    body = md.split("## Rejected claims")[0]
    assert rejected_text not in body and rejected_text in md
    assert "paragraph_dropped" in event_types(deps, run.id)


async def test_run_records_models_and_metrics(deps):
    script_happy_path(deps)
    run = create_run(session_of(deps), "q", deps.settings)
    assert run.models["verifier"] == deps.settings.model_verifier and run.status == "queued"
    await run_research(run.id, deps)
    m = report_of(deps, run.id).metrics
    assert m["claims_drafted"] == 6 and m["supported"] == 6 and m["unsupported"] == 0
    assert m["rounds"] == 0 and m["sources"] == 3
