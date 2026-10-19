"""SQLite schema. The graph state carries only ids; everything a run produces lives here.

Status/label columns are plain strings; allowed values are listed in the comments and
enforced by the code that writes them (Pydantic schemas in app/graph/schemas.py).
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return uuid.uuid4().hex


class Run(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    question: str
    status: str = "queued"  # queued | running | completed | failed
    models: dict = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SubQuestion(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    text: str
    query: str
    round: int = 0
    status: str = "pending"  # pending | researched | thin | ok


class SearchRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    sub_question_id: int = Field(foreign_key="subquestion.id", index=True)
    query: str
    round: int = 0
    provider: str = "tavily"
    n_results: int = 0


class Source(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    url: str
    domain: str
    title: str = ""
    tier: int = 3  # 1 official/academic, 2 major news, 3 other
    fetched_at: datetime = Field(default_factory=_now)


class EvidenceChunk(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    sub_question_id: int = Field(foreign_key="subquestion.id", index=True)
    text: str
    position: int = 0


class Claim(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    sub_question_id: int = Field(foreign_key="subquestion.id", index=True)
    text: str
    chunk_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    round: int = 0
    in_draft: bool = True


class Verdict(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    claim_id: int = Field(foreign_key="claim.id", unique=True, index=True)
    label: str  # supported | partial | unsupported
    quote: str = ""
    quote_found: bool = False
    reason: str = ""
    model: str = ""


class Event(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    node: str
    type: str
    ts: datetime = Field(default_factory=_now)
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))


class Report(SQLModel, table=True):
    run_id: str = Field(foreign_key="run.id", primary_key=True)
    markdown: str
    metrics: dict = Field(default_factory=dict, sa_column=Column(JSON))
    # Final (filtered) SynthesisOutput, so the HTML view can link each citation to its quote.
    synthesis: dict = Field(default_factory=dict, sa_column=Column(JSON))


class CacheEntry(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str
    created_at: datetime = Field(default_factory=_now)
