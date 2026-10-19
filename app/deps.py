"""Collaborators injected into graph nodes via config["configurable"]["deps"]."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from sqlmodel import Session

from app.config import Settings
from app.llm import LlmClient


@dataclass
class SearchResult:
    url: str
    title: str
    content: str | None


class SearchClient(Protocol):
    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class Fetcher(Protocol):
    async def fetch_text(self, url: str) -> str | None: ...


class VectorStore(Protocol):
    async def add(
        self, run_id: str, sub_question_id: int, chunk_ids: list[int], texts: list[str]
    ) -> None: ...

    async def query(self, run_id: str, sub_question_id: int, text: str, k: int) -> list[int]: ...


@dataclass
class Deps:
    settings: Settings
    llm: LlmClient
    search: SearchClient
    fetcher: Fetcher
    vectors: VectorStore
    session_factory: Callable[[], Session]


def build_deps(settings: Settings) -> Deps:
    """Real wiring. Imports are local because the tool modules import SearchResult from here."""
    from app.db import init_db, make_engine
    from app.events import log_event
    from app.llm import MeshLlm
    from app.tools.fetch import HttpFetcher
    from app.tools.search import TavilySearch
    from app.tools.vectorstore import ChromaStore

    engine = make_engine(settings.database_url)
    init_db(engine)

    def session_factory() -> Session:
        return Session(engine)

    def on_usage(run_id: str | None, usage: dict) -> None:
        if run_id is None:
            return
        with session_factory() as session:
            log_event(session, run_id, "llm", "llm_call", usage)

    llm = MeshLlm(settings, on_usage=on_usage)
    return Deps(
        settings=settings,
        llm=llm,
        search=TavilySearch(settings, session_factory),
        fetcher=HttpFetcher(session_factory),
        vectors=ChromaStore(settings.data_dir / "chroma", embed=llm.embed),
        session_factory=session_factory,
    )
