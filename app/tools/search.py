import json
from collections.abc import Callable

from sqlmodel import Session
from tavily import AsyncTavilyClient

from app.config import Settings
from app.deps import SearchResult
from app.tools.cache import cache_get, cache_put


class TavilySearch:
    """Raw web search results only. Tavily's LLM 'answer' feature is never used (Mesh rule)."""

    def __init__(self, settings: Settings, session_factory: Callable[[], Session], client=None):
        self.client = client or AsyncTavilyClient(
            api_key=settings.tavily_api_key or "TAVILY_API_KEY-not-set"
        )
        self.session_factory = session_factory

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        key = f"search:{max_results}:{query}"
        with self.session_factory() as session:
            cached = cache_get(session, key)
        if cached is not None:
            return [SearchResult(**r) for r in json.loads(cached)]

        resp = await self.client.search(
            query=query,
            max_results=max_results,
            include_raw_content=True,
            include_answer=False,
        )
        results = [
            SearchResult(
                url=r["url"], title=r.get("title") or "", content=r.get("raw_content") or None
            )
            for r in resp.get("results", [])
        ]
        with self.session_factory() as session:
            cache_put(session, key, json.dumps([r.__dict__ for r in results]))
        return results
