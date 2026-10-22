import asyncio
from datetime import UTC, datetime

from sqlmodel import Session, select

from app.config import Settings
from app.deps import Deps
from app.events import log_event
from app.graph.build import build_graph
from app.models import Event, Run

MODEL_ROLES = ("planner", "drafter", "verifier", "synthesizer", "contradictions", "embed")


def create_run(
    session: Session, question: str, settings: Settings, user_id: int | None = None
) -> Run:
    run = Run(
        question=question,
        user_id=user_id,
        status="queued",
        models={role: getattr(settings, f"model_{role}") for role in MODEL_ROLES},
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


async def run_research(run_id: str, deps: Deps) -> None:
    """Execute one research run end to end. Never raises: failures are recorded on the run."""
    with deps.session_factory() as session:
        run = session.get(Run, run_id)
        run.status = "running"
        run.started_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        question = run.question
        log_event(session, run_id, "runner", "run_started", {"question": question})

    status, error = "completed", None
    try:
        await build_graph().ainvoke(
            {
                "run_id": run_id,
                "question": question,
                "round": 0,
                "pending_sq_ids": [],
                "researched": [],
            },
            {"configurable": {"deps": deps}, "recursion_limit": 50},
        )
    except asyncio.CancelledError:
        _finish(deps, run_id, "failed", "run cancelled: server shutting down")
        raise
    except Exception as e:  # recorded, not re-raised: runs execute as background tasks
        root = _first_leaf(e)
        status, error = "failed", f"{type(root).__name__}: {root}"
    _finish(deps, run_id, status, error)


def _first_leaf(e: BaseException) -> BaseException:
    """TaskGroup wraps failures in an ExceptionGroup; report the underlying error."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return e


def _finish(deps: Deps, run_id: str, status: str, error: str | None) -> None:
    """Record the final status, timing and token totals, plus a run_completed/run_failed event."""
    with deps.session_factory() as session:
        usage = session.exec(
            select(Event.payload).where(Event.run_id == run_id, Event.type == "llm_call")
        ).all()
        run = session.get(Run, run_id)
        run.status, run.error = status, error
        run.finished_at = datetime.now(UTC)
        run.prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage)
        run.completion_tokens = sum(u.get("completion_tokens", 0) for u in usage)
        session.add(run)
        session.commit()
        if status == "failed":
            log_event(session, run_id, "runner", "run_failed", {"error": error})
        else:
            log_event(session, run_id, "runner", "run_completed", {})


def fail_interrupted_runs(deps: Deps) -> int:
    """On startup: runs left queued/running by a restart can never finish, so close them."""
    with deps.session_factory() as session:
        stuck = session.exec(select(Run).where(Run.status.in_(["queued", "running"]))).all()
        ids = [r.id for r in stuck]
    for run_id in ids:
        _finish(deps, run_id, "failed", "interrupted by server restart")
    return len(ids)
