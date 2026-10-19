"""Tiny key/value cache in SQLite: makes eval reruns reproducible and saves Tavily quota."""

from sqlmodel import Session

from app.models import CacheEntry


def cache_get(session: Session, key: str) -> str | None:
    entry = session.get(CacheEntry, key)
    return entry.value if entry else None


def cache_put(session: Session, key: str, value: str) -> None:
    entry = session.get(CacheEntry, key)
    if entry:
        entry.value = value
    else:
        session.add(CacheEntry(key=key, value=value))
    session.commit()
