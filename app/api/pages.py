import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import func, select

from app.api.auth import admin_user, can_view, session_user
from app.api.quota import quota_for, start_refusal
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
from app.tools.tiers import TIER_LABELS

ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))
EVAL_RESULTS = ROOT / "eval" / "results"

# Example chips on the home page: questions from the eval set that make good demos.
EXAMPLES = [
    (
        "RBI repo rate in 2025",
        "How did the Reserve Bank of India's repo rate change during 2025, "
        "and what reasons did the RBI give?",
    ),
    (
        "India's chip fabs",
        "What is the status of India's semiconductor fabrication plants announced under "
        "the India Semiconductor Mission?",
    ),
    (
        "Gold price in 2025",
        "What drove the price of gold in 2025, and how much did it change over the year?",
    ),
]


def time_ago(ts: datetime, now: datetime | None = None) -> str:
    """'3 min ago' style age; SQLite hands timestamps back without a zone, and they are UTC."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    secs = int(((now or datetime.now(UTC)) - ts).total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} h ago"
    days = secs // 86400
    return f"{days} day{'s' if days > 1 else ''} ago"


def duration(run: Run) -> str:
    if run.started_at is None or run.finished_at is None:
        return ""
    secs = int((run.finished_at - run.started_at).total_seconds())
    return f"{secs // 60} min {secs % 60} s"


templates.env.filters["ago"] = time_ago
templates.env.globals["tier_labels"] = TIER_LABELS
templates.env.globals["is_admin"] = lambda request: admin_user(request) is not None

router = APIRouter()


def _deps(request: Request) -> Deps:
    return request.app.state.deps


class SignInRequired(Exception):
    """A signed-out visitor asked for a run that isn't public: send them to sign in."""


def _visible_run(session, run_id: str, request: Request) -> Run:
    run = session.get(Run, run_id)
    user = session_user(request)
    if run is not None and can_view(run, user, admin=admin_user(request) is not None):
        return run
    if user is None:
        raise SignInRequired()
    raise HTTPException(status_code=404, detail="run not found")


def sign_in_redirect(request: Request, exc: SignInRequired) -> RedirectResponse:
    return RedirectResponse(f"/auth/login?next={request.url.path}", status_code=303)


@router.get("/", response_class=HTMLResponse)
def ask(request: Request):
    user = session_user(request)
    with _deps(request).session_factory() as session:
        # Failed runs stay out of the home lists (admin still sees them).
        examples = session.exec(
            select(Run)
            .where(Run.is_example, Run.status != "failed")
            .order_by(Run.created_at.desc())
            .limit(6)
        ).all()
        mine, quota = [], None
        if user is not None:
            mine = session.exec(
                select(Run)
                .where(Run.user_id == user["id"], Run.status != "failed")
                .order_by(Run.created_at.desc())
                .limit(8)
            ).all()
            quota = quota_for(session, user["id"], _deps(request).settings)
        featured = session.exec(select(Run).where(Run.is_featured, Run.is_example)).first()
        done = [r for r in examples if r.status == "completed"]
        shown = featured or (done[0] if done else None)  # the admin's pick, else the newest
        samples = _verdict_samples(session, shown.id) if shown else []
    return templates.TemplateResponse(
        request,
        "ask.html",
        {
            "user": user,
            "mine": mine,
            "quota": quota,
            "refusal": quota and start_refusal(quota, _deps(request).settings),
            "samples": samples,
            "example_runs": examples,
            "examples": EXAMPLES,
            "results": latest_eval_results(),
            "signin_failed": request.query_params.get("signin") == "failed",
        },
    )


def _verdict_samples(session, run_id: str) -> list[dict]:
    """One supported and one rejected claim from a report, shortest first, for the home page."""
    samples = []
    for label in ("supported", "unsupported"):
        row = session.exec(
            select(Claim, Verdict)
            .join(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.run_id == run_id, Verdict.label == label)
            .order_by(func.length(Claim.text))
        ).first()
        if row:
            claim, verdict = row
            samples.append(
                {"run_id": run_id, "label": label, "text": claim.text, "reason": verdict.reason}
            )
    return samples


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(run_id: str, request: Request):
    with _deps(request).session_factory() as session:
        run = _visible_run(session, run_id, request)
        view = _report_view(session, run) if run.status == "completed" else None
    return templates.TemplateResponse(
        request, "run.html", {"run": run, "view": view, "duration": duration(run)}
    )


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
                {
                    "n": n,
                    "url": src.url,
                    "domain": src.domain,
                    "tier": src.tier,
                    "quote": verdicts[cid].quote if cid in verdicts else "",
                }
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
    labels = Counter(v.label for v in verdicts.values())
    sources = [(citations.numbers[s.url], s) for s in citations.cited]
    return {
        "synthesis": synthesis,
        "sections": sections,
        "thin": thin,
        "conflicts": conflicts,
        "rejected": rejected,
        "sources": sources,
        "counts": {
            "total": len(claims),
            "supported": labels["supported"],
            "partial": labels["partial"],
            "unsupported": labels["unsupported"],
            "official": sum(1 for _, s in sources if s.tier == 1),
            "rounds": (report.metrics or {}).get("rounds", 0),
        },
    }


@router.get("/runs/{run_id}/report", response_class=HTMLResponse)
def report_page(run_id: str, request: Request, fragment: bool = False):
    with _deps(request).session_factory() as session:
        run = _visible_run(session, run_id, request)
        view = _report_view(session, run)
    template = "_report_body.html" if fragment else "report.html"
    return templates.TemplateResponse(request, template, {"run": run, "view": view})


@router.get("/runs/{run_id}/trail", response_class=HTMLResponse)
def trail_page(run_id: str, request: Request):
    with _deps(request).session_factory() as session:
        run = _visible_run(session, run_id, request)
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


LEGAL_UPDATED = "22 October 2026"


@router.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request):
    return templates.TemplateResponse(request, "privacy.html", _legal_context(request))


@router.get("/terms", response_class=HTMLResponse)
def terms_page(request: Request):
    return templates.TemplateResponse(request, "terms.html", _legal_context(request))


def _legal_context(request: Request) -> dict:
    return {"contact": _deps(request).settings.contact_email, "updated": LEGAL_UPDATED}


@router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request):
    return templates.TemplateResponse(request, "eval.html", {"results": latest_eval_results()})
