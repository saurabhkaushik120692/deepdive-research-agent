from contextvars import ContextVar

from sqlmodel import Session

from app.models import Event

# Set while a follow-up runs, so events logged without a followup_id in their payload (the
# llm_call usage events from app/llm.py) are still tagged as the follow-up's.
current_followup: ContextVar[int | None] = ContextVar("current_followup", default=None)


def log_event(
    session: Session, run_id: str, node: str, type: str, payload: dict | None = None
) -> Event:
    """Append one audit-trail event. Feeds both the SSE live timeline and the trail page.

    A payload with a followup_id, or a running follow-up (current_followup), tags the event as
    part of that follow-up, so report views can leave it out."""
    payload = payload or {}
    followup_id = payload.get("followup_id") or current_followup.get()
    event = Event(run_id=run_id, node=node, type=type, payload=payload, followup_id=followup_id)
    session.add(event)
    session.commit()
    session.refresh(event)
    return event
