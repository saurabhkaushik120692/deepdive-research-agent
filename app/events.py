from sqlmodel import Session

from app.models import Event


def log_event(
    session: Session, run_id: str, node: str, type: str, payload: dict | None = None
) -> Event:
    """Append one audit-trail event. Feeds both the SSE live timeline and the trail page."""
    event = Event(run_id=run_id, node=node, type=type, payload=payload or {})
    session.add(event)
    session.commit()
    session.refresh(event)
    return event
