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
