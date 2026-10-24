import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from sqlmodel import func, select
from sse_starlette.sse import EventSourceResponse

from app.api.auth import can_view, require_user, session_user
from app.api.quota import limit_message, quota_for
from app.deps import Deps
from app.graph.runner import create_run, run_research
from app.models import Claim, Event, Report, Run, SearchRecord, Source, SubQuestion, Verdict

router = APIRouter(prefix="/api")
POLL_SECONDS = 0.5
TERMINAL = ("completed", "failed")


class RunRequest(BaseModel):
    question: str = Field(min_length=10, max_length=500)


def _deps(request: Request) -> Deps:
    return request.app.state.deps


def _get_run_or_404(session, run_id: str, request: Request) -> Run:
    """The run, if this visitor may see it; other people's runs look like missing ones."""
    run = session.get(Run, run_id)
    if run is None or not can_view(run, session_user(request)):
        raise HTTPException(status_code=404, detail="run not found")
    return run


async def _passes_safety_screen(deps: Deps, question: str) -> bool:
    try:
        flagged = await asyncio.wait_for(
            deps.llm.moderate(question), timeout=deps.settings.moderation_timeout_s
        )
        return not flagged
    except Exception:
        # Moderation is a courtesy screen; if it's down or queued behind busy runs, don't block.
        return True


@router.post("/runs", status_code=202)
async def start_run(
    body: RunRequest, request: Request, user: Annotated[dict, Depends(require_user)]
) -> dict:
    deps = _deps(request)
    if not await _passes_safety_screen(deps, body.question):
        raise HTTPException(status_code=400, detail="question rejected by safety screen")

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    with deps.session_factory() as session:
        if quota_for(session, user["id"], deps.settings).exhausted:
            raise HTTPException(status_code=403, detail=limit_message(deps.settings))
        recent = session.exec(select(func.count()).where(Run.created_at >= cutoff)).one()
        if recent >= deps.settings.runs_per_hour:
            raise HTTPException(status_code=429, detail="hourly run limit reached, try later")
        run_id = create_run(session, body.question, deps.settings, user_id=user["id"]).id

    async def _execute() -> None:
        async with request.app.state.run_slots:
            await run_research(run_id, deps)

    task = asyncio.create_task(_execute())
    request.app.state.tasks.add(task)  # keep a reference so the task isn't garbage-collected
    task.add_done_callback(request.app.state.tasks.discard)
    return {"run_id": run_id}


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    with _deps(request).session_factory() as session:
        run = _get_run_or_404(session, run_id, request)
        report = session.get(Report, run_id)
        return {
            "id": run.id,
            "question": run.question,
            "status": run.status,
            "error": run.error,
            "report_markdown": report.markdown if report else None,
            "metrics": report.metrics if report else None,
        }


@router.get("/runs/{run_id}/trail")
def get_trail(run_id: str, request: Request) -> dict:
    with _deps(request).session_factory() as session:
        run = _get_run_or_404(session, run_id, request)
        sqs = session.exec(
            select(SubQuestion).where(SubQuestion.run_id == run_id).order_by(SubQuestion.id)
        ).all()
        searches = session.exec(
            select(SearchRecord)
            .where(SearchRecord.sub_question_id.in_([q.id for q in sqs]))
            .order_by(SearchRecord.id)
        ).all()
        sources = session.exec(select(Source).where(Source.run_id == run_id)).all()
        rows = session.exec(
            select(Claim, Verdict)
            .outerjoin(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.run_id == run_id)
            .order_by(Claim.id)
        ).all()
        events = session.exec(select(Event).where(Event.run_id == run_id).order_by(Event.id)).all()
        return jsonable_encoder(
            {
                "run": run,
                "sub_questions": sqs,
                "searches": searches,
                "sources": sources,
                "claims": [
                    {**c.model_dump(), "verdict": v.model_dump() if v else None} for c, v in rows
                ],
                "events": events,
            }
        )


def _claim_texts(session, events: list[Event]) -> dict[int, str]:
    """Claim text for verdict events, so the live feed can show what was checked."""
    ids = [
        e.payload["claim_id"]
        for e in events
        if e.type == "claim_verified" and (e.payload or {}).get("claim_id")
    ]
    if not ids:
        return {}
    return dict(session.exec(select(Claim.id, Claim.text).where(Claim.id.in_(ids))).all())


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request):
    """SSE: replay the run's events (after Last-Event-ID), follow new ones, then send `done`."""
    deps = _deps(request)
    with deps.session_factory() as session:
        _get_run_or_404(session, run_id, request)
    try:
        last_id = int(request.headers.get("last-event-id", "0"))
    except ValueError:
        last_id = 0

    async def gen():
        nonlocal last_id
        while True:
            with deps.session_factory() as session:
                new = session.exec(
                    select(Event)
                    .where(Event.run_id == run_id, Event.id > last_id)
                    .order_by(Event.id)
                ).all()
                status = session.get(Run, run_id).status
                texts = _claim_texts(session, new)
            for e in new:
                last_id = e.id
                payload = {"node": e.node, "ts": e.ts.isoformat(), **(e.payload or {})}
                if e.type == "claim_verified" and payload.get("claim_id") in texts:
                    payload.setdefault("text", texts[payload["claim_id"]])
                yield {"id": str(e.id), "event": e.type, "data": json.dumps(payload, default=str)}
            if status in TERMINAL and not new:
                yield {"event": "done", "data": json.dumps({"status": status})}
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(POLL_SECONDS)

    return EventSourceResponse(gen())
