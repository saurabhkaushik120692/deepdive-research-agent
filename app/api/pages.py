import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from app.api.guard import has_access
from app.deps import Deps
from app.graph.schemas import Conflict, SynthesisOutput
from app.models import (
    Claim,
    Event,
    EvidenceChunk,
    Report,
    Run,
    SearchRecord,
    Source,
    SubQuestion,
    Verdict,
)
from app.report import Citations

ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))
EVAL_RESULTS = ROOT / "eval" / "results"

router = APIRouter()


def _deps(request: Request) -> Deps:
    return request.app.state.deps


def _gate(request: Request) -> RedirectResponse | None:
    return None if has_access(request) else RedirectResponse("/", status_code=303)


def _run_or_404(session, run_id: str) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.get("/", response_class=HTMLResponse)
def ask(request: Request):
    if not has_access(request):
        return templates.TemplateResponse(request, "gate.html", {})
    with _deps(request).session_factory() as session:
        recent = session.exec(select(Run).order_by(Run.created_at.desc()).limit(10)).all()
    return templates.TemplateResponse(request, "ask.html", {"recent": recent})


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(run_id: str, request: Request):
    if redirect := _gate(request):
        return redirect
    with _deps(request).session_factory() as session:
        run = _run_or_404(session, run_id)
    return templates.TemplateResponse(request, "run.html", {"run": run})


def _report_view(session, run: Run) -> dict | None:
    report = session.get(Report, run.id)
    if report is None or not report.synthesis:
        return None
    synthesis = SynthesisOutput.model_validate(report.synthesis)
    claims = {c.id: c for c in session.exec(select(Claim).where(Claim.run_id == run.id)).all()}
    verdicts = {
        v.claim_id: v
        for v in session.exec(select(Verdict).where(Verdict.claim_id.in_(list(claims)))).all()
    }
    chunk_sources = {
        ch.id: src
        for ch, src in session.exec(
            select(EvidenceChunk, Source)
            .join(Source, Source.id == EvidenceChunk.source_id)
            .where(EvidenceChunk.run_id == run.id)
        ).all()
    }
    citations = Citations(claims, chunk_sources)
    sections = []
    for section in synthesis.sections:
        paragraphs = []
        for p in section.paragraphs:
            cites = [
                {"n": n, "url": src.url, "quote": verdicts[cid].quote if cid in verdicts else ""}
                for cid in p.claim_ids
                for n, src in citations.sources_for(cid)
            ]
            partial = any(
                verdicts[cid].label == "partial" for cid in p.claim_ids if cid in verdicts
            )
            paragraphs.append({"text": p.text, "cites": cites, "partial": partial})
        sections.append({"heading": section.heading, "paragraphs": paragraphs})

    thin = session.exec(
        select(SubQuestion.text).where(SubQuestion.run_id == run.id, SubQuestion.status == "thin")
    ).all()
    conflicts = []
    for e in session.exec(
        select(Event).where(Event.run_id == run.id, Event.type == "conflict_found")
    ).all():
        c = Conflict(**e.payload)
        if c.claim_a in claims and c.claim_b in claims:
            conflicts.append(
                {
                    "a": claims[c.claim_a].text,
                    "b": claims[c.claim_b].text,
                    "explanation": c.explanation,
                }
            )
    rejected = [
        {"text": claims[cid].text, "reason": v.reason}
        for cid, v in sorted(verdicts.items())
        if v.label == "unsupported"
    ]
    return {
        "synthesis": synthesis,
        "sections": sections,
        "thin": thin,
        "conflicts": conflicts,
        "rejected": rejected,
        "sources": [(citations.numbers[s.url], s) for s in citations.cited],
        "metrics": report.metrics,
    }


@router.get("/runs/{run_id}/report", response_class=HTMLResponse)
def report_page(run_id: str, request: Request, fragment: bool = False):
    if redirect := _gate(request):
        return redirect
    with _deps(request).session_factory() as session:
        run = _run_or_404(session, run_id)
        view = _report_view(session, run)
    template = "_report_body.html" if fragment else "report.html"
    return templates.TemplateResponse(request, template, {"run": run, "view": view})


@router.get("/runs/{run_id}/trail", response_class=HTMLResponse)
def trail_page(run_id: str, request: Request):
    if redirect := _gate(request):
        return redirect
    with _deps(request).session_factory() as session:
        run = _run_or_404(session, run_id)
        sqs = session.exec(
            select(SubQuestion).where(SubQuestion.run_id == run_id).order_by(SubQuestion.id)
        ).all()
        searches = session.exec(
            select(SearchRecord)
            .where(SearchRecord.sub_question_id.in_([q.id for q in sqs]))
            .order_by(SearchRecord.id)
        ).all()
        claims = session.exec(
            select(Claim, Verdict)
            .outerjoin(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.run_id == run_id)
            .order_by(Claim.id)
        ).all()
        events = session.exec(
            select(Event).where(Event.run_id == run_id, Event.type != "llm_call").order_by(Event.id)
        ).all()
    return templates.TemplateResponse(
        request,
        "trail.html",
        {"run": run, "sqs": sqs, "searches": searches, "claims": claims, "events": events},
    )


def latest_eval_results() -> dict | None:
    files = sorted(EVAL_RESULTS.glob("*.json")) if EVAL_RESULTS.is_dir() else []
    files = [f for f in files if f.name != "runs.json"]
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


@router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request):
    return templates.TemplateResponse(request, "eval.html", {"results": latest_eval_results()})
