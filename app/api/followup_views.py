"""Read-side views of follow-ups, shared by the JSON trail and the HTML pages."""

from sqlmodel import select

from app.models import (
    Claim,
    Event,
    EvidenceChunk,
    FollowUp,
    Run,
    SearchRecord,
    Source,
    SubQuestion,
    Verdict,
)
from app.report import Citations

STAGE_LABELS = {
    "waiting": "Waiting for a free slot…",
    "scope": "Checking it's on topic…",
    "drafting": "Checking the report's sources…",
    "searching": "Searching the web for more…",
    "writing": "Writing the answer…",
}


def paragraph_view(
    text: str, claim_ids: list[int], citations: Citations, verdicts: dict[int, Verdict]
) -> dict:
    """A paragraph with its numbered citations and the verifier's quote for each, as the report
    page shows them."""
    cites, seen = [], set()
    for cid in claim_ids:
        if cid not in citations.claims:
            continue
        for n, src in citations.sources_for(cid):
            if n in seen:  # several claims from one source: one marker
                continue
            seen.add(n)
            quote = verdicts[cid].quote if cid in verdicts else ""
            cites.append(
                {"n": n, "url": src.url, "domain": src.domain, "tier": src.tier, "quote": quote}
            )
    partial = any(verdicts[cid].label == "partial" for cid in claim_ids if cid in verdicts)
    return {"text": text, "cites": cites, "partial": partial}


def followups_view(session, run: Run) -> list[dict]:
    """Follow-ups on a report, oldest first, ready for the report page. Each answer numbers its
    own citations from 1 and lists its own sources."""
    items = []
    for f in session.exec(
        select(FollowUp).where(FollowUp.run_id == run.id).order_by(FollowUp.id)
    ).all():
        answer = f.answer or {}
        paragraphs, sources = [], []
        if f.status == "answered" and answer.get("paragraphs"):
            ids = {i for p in answer["paragraphs"] for i in p["claim_ids"]}
            claims = {c.id: c for c in session.exec(select(Claim).where(Claim.id.in_(ids))).all()}
            verdicts = {
                v.claim_id: v
                for v in session.exec(select(Verdict).where(Verdict.claim_id.in_(ids))).all()
            }
            chunk_ids = {i for c in claims.values() for i in c.chunk_ids}
            chunk_sources = {
                ch.id: src
                for ch, src in session.exec(
                    select(EvidenceChunk, Source)
                    .join(Source, Source.id == EvidenceChunk.source_id)
                    .where(EvidenceChunk.id.in_(chunk_ids))
                ).all()
            }
            citations = Citations(claims, chunk_sources)
            paragraphs = [
                paragraph_view(p["text"], p["claim_ids"], citations, verdicts)
                for p in answer["paragraphs"]
            ]
            sources = [(citations.numbers[s.url], s) for s in citations.cited]
        stage = "waiting" if f.status == "queued" else f.stage
        items.append(
            {
                "id": f.id,
                "question": f.question,
                "status": f.status,
                "stage_label": STAGE_LABELS.get(stage, STAGE_LABELS["waiting"]),
                "decline_reason": f.decline_reason,
                "searched_web": f.searched_web,
                "none_found": bool(answer.get("none_found")),
                "paragraphs": paragraphs,
                "sources": sources,
            }
        )
    return items


def followup_trail(session, run_id: str) -> list[dict]:
    """Each follow-up on a run with its own audit trail, oldest first. Report views leave these
    rows out, so the report itself never changes after it's written."""
    out = []
    for f in session.exec(
        select(FollowUp).where(FollowUp.run_id == run_id).order_by(FollowUp.id)
    ).all():
        sq = session.exec(select(SubQuestion).where(SubQuestion.followup_id == f.id)).first()
        searches = (
            session.exec(
                select(SearchRecord)
                .where(SearchRecord.sub_question_id == sq.id)
                .order_by(SearchRecord.id)
            ).all()
            if sq
            else []
        )
        claims = session.exec(
            select(Claim, Verdict)
            .outerjoin(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.followup_id == f.id)
            .order_by(Claim.id)
        ).all()
        events = session.exec(
            select(Event)
            .where(Event.followup_id == f.id, Event.type != "llm_call")
            .order_by(Event.id)
        ).all()
        out.append(
            {
                "followup": f,
                "sub_question": sq,
                "searches": searches,
                "claims": claims,
                "events": events,
            }
        )
    return out
