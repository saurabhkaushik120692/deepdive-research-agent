import json
import random
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from app.api.auth import admin_user, can_view, session_user
from app.api.followup_views import followup_trail, followups_view, paragraph_view
from app.api.followups import NOT_YOURS, followup_refusal, followups_left
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
from app.tools.chunking import strip_controls
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
# Reports stored before strip_controls existed can hold control characters; clean every value.
templates.env.finalize = lambda v: strip_controls(v) if isinstance(v, str) else v
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
            "sample_question": shown.question if samples else "",
            "example_runs": examples,
            "examples": EXAMPLES,
            "results": latest_eval_results(),
            "signin_failed": request.query_params.get("signin") == "failed",
        },
    )


def _readability(text: str) -> int:
    """How well a claim reads on its own: a full sentence, with a figure and a name in it."""
    score = 2 if 50 <= len(text) <= 160 else 1 if 30 <= len(text) <= 200 else 0
    score += bool(re.search(r"\d", text))
    score += bool(re.search(r"\b[A-Z][A-Za-z]+", text[1:]))  # a name after the first letter
    return score


def _verdict_samples(session, run_id: str, pick=random.choice) -> list[dict]:
    """One supported and one rejected claim from a report for the home page. Each visit picks
    among the few claims that read best on their own, so the card changes between visits."""
    samples = []
    for label in ("supported", "unsupported"):
        rows = session.exec(
            select(Claim, Verdict)
            .join(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.run_id == run_id, Claim.followup_id.is_(None), Verdict.label == label)
            .order_by(Claim.id)
        ).all()
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: -_readability(row[0].text))
        best = _readability(rows[0][0].text)
        claim, verdict = pick([r for r in rows[:5] if _readability(r[0].text) >= best - 1])
        samples.append(
            {"run_id": run_id, "label": label, "text": claim.text, "reason": verdict.reason}
        )
    return samples


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(run_id: str, request: Request):
    with _deps(request).session_factory() as session:
        run = _visible_run(session, run_id, request)
        view = _report_view(session, run) if run.status == "completed" else None
        fu = _followup_context(session, run, request)
    return templates.TemplateResponse(
        request, "run.html", {"run": run, "view": view, "fu": fu, "duration": duration(run)}
    )


def _followup_context(session, run: Run, request: Request) -> dict | None:
    """The follow-up section of a finished report, for its owner (and the admin). None hides
    it: other visitors, unfinished runs, and examples that never had follow-ups."""
    user = session_user(request)
    owner = user is not None and run.user_id == user["id"]
    if run.status != "completed" or not (owner or admin_user(request) is not None):
        return None
    settings = _deps(request).settings
    items = followups_view(session, run)
    refusal = followup_refusal(session, run, user, settings)
    if not items and refusal == NOT_YOURS:
        return None
    busy = any(i["status"] in ("queued", "running") for i in items)
    return {
        "items": items,
        "can_ask": refusal is None and not busy,
        "refusal": refusal if refusal != NOT_YOURS else None,
        "left": followups_left(session, run.id, settings),
        "limit": settings.followups_per_report,
    }


def _report_view(session, run: Run) -> dict | None:
    report = session.get(Report, run.id)
    if report is None or not report.synthesis:
        return None
    synthesis = SynthesisOutput.model_validate(report.synthesis)
    claims = {
        c.id: c
        for c in session.exec(
            select(Claim).where(Claim.run_id == run.id, Claim.followup_id.is_(None))
        ).all()
    }
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
        paragraphs = [
            paragraph_view(p.text, p.claim_ids, citations, verdicts) for p in section.paragraphs
        ]
        sections.append({"heading": section.heading, "paragraphs": paragraphs})

    thin = session.exec(
        select(SubQuestion.text).where(
            SubQuestion.run_id == run.id,
            SubQuestion.followup_id.is_(None),
            SubQuestion.status == "thin",
        )
    ).all()
    conflicts = []
    for e in session.exec(
        select(Event).where(
            Event.run_id == run.id, Event.followup_id.is_(None), Event.type == "conflict_found"
        )
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
        fu = _followup_context(session, run, request)
    template = "_report_body.html" if fragment else "report.html"
    return templates.TemplateResponse(request, template, {"run": run, "view": view, "fu": fu})


@router.get("/runs/{run_id}/followups", response_class=HTMLResponse)
def followups_fragment(run_id: str, request: Request):
    """The follow-up section alone, re-rendered by the page's script after a follow-up ends."""
    with _deps(request).session_factory() as session:
        run = _visible_run(session, run_id, request)
        fu = _followup_context(session, run, request)
    if fu is None:
        raise HTTPException(status_code=404, detail="no follow-ups here")
    return templates.TemplateResponse(request, "_followups.html", {"run": run, "fu": fu})


@router.get("/runs/{run_id}/trail", response_class=HTMLResponse)
def trail_page(run_id: str, request: Request):
    with _deps(request).session_factory() as session:
        run = _visible_run(session, run_id, request)
        sqs = session.exec(
            select(SubQuestion)
            .where(SubQuestion.run_id == run_id, SubQuestion.followup_id.is_(None))
            .order_by(SubQuestion.id)
        ).all()
        searches = session.exec(
            select(SearchRecord)
            .where(SearchRecord.sub_question_id.in_([q.id for q in sqs]))
            .order_by(SearchRecord.id)
        ).all()
        claims = session.exec(
            select(Claim, Verdict)
            .outerjoin(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.run_id == run_id, Claim.followup_id.is_(None))
            .order_by(Claim.id)
        ).all()
        events = session.exec(
            select(Event)
            .where(Event.run_id == run_id, Event.followup_id.is_(None), Event.type != "llm_call")
            .order_by(Event.id)
        ).all()
        owner_or_admin = _followup_context(session, run, request) is not None
        followups = followup_trail(session, run_id) if owner_or_admin else []
    return templates.TemplateResponse(
        request,
        "trail.html",
        {
            "run": run,
            "sqs": sqs,
            "searches": searches,
            "claims": claims,
            "events": events,
            "followups": followups,
        },
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
