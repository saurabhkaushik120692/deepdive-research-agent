import asyncio
from collections.abc import Callable

import httpx
import trafilatura
from sqlmodel import Session

from app.tools.cache import cache_get, cache_put

MIN_TEXT_CHARS = 200
_HTML_TYPES = ("text/html", "application/xhtml+xml")


class HttpFetcher:
    """Fetch a page and extract its main text. Any failure returns None (and is cached)."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        http: httpx.AsyncClient | None = None,
        timeout_s: float = 15,
    ) -> None:
        self.session_factory = session_factory
        self.http = http or httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "DeepDiveResearchAgent/0.1 (+research bot)"},
        )

    async def fetch_text(self, url: str) -> str | None:
        key = f"fetch:{url}"
        with self.session_factory() as session:
            cached = cache_get(session, key)
        if cached is not None:
            return cached or None

        text = await self._fetch(url)
        with self.session_factory() as session:
            cache_put(session, key, text or "")
        return text

    async def _fetch(self, url: str) -> str | None:
        try:
            resp = await self.http.get(url)
        except httpx.HTTPError:
            return None
        ctype = resp.headers.get("content-type", "").lower()
        if not resp.is_success or not ctype.startswith(_HTML_TYPES):
            return None
        text = await asyncio.to_thread(trafilatura.extract, resp.text)
        if not text or len(text) < MIN_TEXT_CHARS:
            return None
        return text
