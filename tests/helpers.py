"""Small query helpers shared by graph tests."""

from sqlmodel import select

from app.models import Claim, Event, EvidenceChunk, Report, Run, Source, SubQuestion, Verdict


def cfg(deps):
    return {"configurable": {"deps": deps}}


def events(deps, run_id):
    with deps.session_factory() as s:
        return s.exec(select(Event).where(Event.run_id == run_id).order_by(Event.id)).all()


def event_types(deps, run_id):
    return {e.type for e in events(deps, run_id)}


def all_sources(deps, run_id):
    with deps.session_factory() as s:
        return s.exec(select(Source).where(Source.run_id == run_id)).all()


def chunk_count(deps, sq_id):
    with deps.session_factory() as s:
        return len(
            s.exec(select(EvidenceChunk).where(EvidenceChunk.sub_question_id == sq_id)).all()
        )


def sub_questions(deps, run_id):
    with deps.session_factory() as s:
        return s.exec(
            select(SubQuestion).where(SubQuestion.run_id == run_id).order_by(SubQuestion.id)
        ).all()


def claims_of(deps, run_id):
    with deps.session_factory() as s:
        return s.exec(select(Claim).where(Claim.run_id == run_id).order_by(Claim.id)).all()


def verdict_of(deps, claim_id):
    with deps.session_factory() as s:
        return s.exec(select(Verdict).where(Verdict.claim_id == claim_id)).first()


def get_run(deps, run_id):
    with deps.session_factory() as s:
        return s.get(Run, run_id)


def report_of(deps, run_id):
    with deps.session_factory() as s:
        return s.get(Report, run_id)


def state(run, pending=None, round_=0):
    return {
        "run_id": run.id,
        "question": run.question,
        "round": round_,
        "pending_sq_ids": pending or [],
        "researched": [],
    }
