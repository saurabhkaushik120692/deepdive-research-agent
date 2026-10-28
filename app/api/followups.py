"""Follow-up questions under a finished report: who may ask, and the two endpoints."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import func, select

from app.api.auth import admin_user, require_user, session_user
from app.api.quota import blocked_message
from app.api.runs import _deps, _get_run_or_404, _passes_safety_screen
from app.config import Settings
from app.graph.followup import run_followup
from app.models import FollowUp, Run, User

router = APIRouter(prefix="/api")

ACTIVE = ("queued", "running")
NOT_YOURS = "Follow-ups are only available on your own finished reports."


class FollowUpRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)  # SQLite returns naive datetimes


def followups_used(session, run_id: str) -> int:
    """Follow-ups that count toward the per-report limit: failed ones were our fault."""
    return session.exec(
        select(func.count()).where(FollowUp.run_id == run_id, FollowUp.status != "failed")
    ).one()


def followups_left(session, run_id: str, settings: Settings) -> int:
    return max(0, settings.followups_per_report - followups_used(session, run_id))


def followup_refusal(session, run: Run, user: dict | None, settings: Settings) -> str | None:
    """Why this visitor can't ask a follow-up on this report now, or None if they can."""
    if user is None or run.user_id != user["id"] or run.is_example or run.status != "completed":
        return NOT_YOURS
    account = session.get(User, user["id"])
    if account is not None and account.blocked:
        return blocked_message(settings)
    finished = _aware(run.finished_at or run.created_at)
    if datetime.now(UTC) - finished > timedelta(days=settings.followup_max_age_days):
        return (
            f"Follow-ups are closed for reports older than {settings.followup_max_age_days} days."
        )
    if followups_left(session, run.id, settings) == 0:
        return f"You've used all {settings.followups_per_report} follow-ups on this report."
    return None


@router.post("/runs/{run_id}/followups", status_code=202)
async def ask_followup(
    run_id: str,
    body: FollowUpRequest,
    request: Request,
    user: Annotated[dict, Depends(require_user)],
) -> dict:
    deps = _deps(request)
    with deps.session_factory() as session:
        _get_run_or_404(session, run_id, request)
    if not await _passes_safety_screen(deps, body.question):
        raise HTTPException(status_code=400, detail="question rejected by safety screen")

    # No awaits from here to the insert, so two quick clicks can't both pass the running check.
    with deps.session_factory() as session:
        run = session.get(Run, run_id)
        refusal = followup_refusal(session, run, user, deps.settings)
        if refusal:
            raise HTTPException(status_code=403, detail=refusal)
        running = session.exec(
            select(FollowUp.id).where(FollowUp.run_id == run_id, FollowUp.status.in_(ACTIVE))
        ).first()
        if running is not None:
            raise HTTPException(
                status_code=409, detail="A follow-up is already running on this report."
            )
        f = FollowUp(run_id=run_id, user_id=user["id"], question=body.question.strip())
        session.add(f)
        session.commit()
        session.refresh(f)
        followup_id = f.id

    async def _execute() -> None:
        async with request.app.state.run_slots:  # shares the run slots with full reports
            await run_followup(followup_id, deps)

    task = asyncio.create_task(_execute())
    request.app.state.tasks.add(task)
    task.add_done_callback(request.app.state.tasks.discard)
    return {"followup_id": followup_id}


@router.get("/runs/{run_id}/followups/{followup_id}")
def followup_status(run_id: str, followup_id: int, request: Request) -> dict:
    with _deps(request).session_factory() as session:
        run = _get_run_or_404(session, run_id, request)
        f = session.get(FollowUp, followup_id)
        user = session_user(request)
        owner = user is not None and run.user_id == user["id"]
        if f is None or f.run_id != run_id or not (owner or admin_user(request) is not None):
            raise HTTPException(status_code=404, detail="follow-up not found")
        return {
            "id": f.id,
            "status": f.status,
            "stage": f.stage,
            "decline_reason": f.decline_reason,
            "searched_web": f.searched_web,
        }
